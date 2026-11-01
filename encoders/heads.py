"""
encoders/heads.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import json
import os
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_question_sets import PROJECT, questions_csv, read_partition
from data_loader.build_utils import read_csv_defensively
from encoders.dinov3 import dev_csv, dinov3_dir, features_npz
from Inference.resume_utils import (ClaimedElsewhere, MissingInput, append_status, digest_file, read_build_params, run_units_resumable,
                                    status_path, write_build_params, write_jsonl_atomic, write_npz_atomic)
from Inference.serving_utils import require_packages
from Inference.stats_utils import stratified_group_folds


HEADS_VERSION = 3


def head_npz(cfg: Dict, modality: str, finding: str) -> str:
    return os.path.join(dinov3_dir(cfg), "heads", f"{modality}__{finding}.npz")


def reference_path(cfg: Dict, modality: str) -> str:
    return os.path.join(cfg["runs"]["dir"], modality, cfg["dinov3"]["model_key"], "original.jsonl")


def fit_head(X: np.ndarray, y: np.ndarray, groups: np.ndarray, dino_cfg: Dict) -> Dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    seed = int(dino_cfg["seed"])

    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=int(dino_cfg["max_iter"]), random_state=seed))

    folds = stratified_group_folds(y, groups, int(dino_cfg["cv_folds"]), seed)
    grid = [float(c) for c in dino_cfg["c_grid"]]
    scores = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for c in grid:
            aucs = [roc_auc_score(y[va], model(c).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]) for tr, va in folds
                    if len(np.unique(y[va])) == 2 and len(np.unique(y[tr])) == 2]
            scores[c] = float(np.mean(aucs)) if aucs else float("nan")
        if not any(np.isfinite(v) for v in scores.values()):
            raise ValueError("no fold holds both classes in training and validation")
        best = sorted((c for c in grid if np.isfinite(scores[c])), key=lambda c: (-scores[c], c))[0]
        final = model(best).fit(X, y)
    n_warn = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return {"model": final, "C": best, "cv_auroc": [scores[c] for c in grid], "grid": grid, "n_convergence_warnings": n_warn}


def _features_identity(cfg: Dict, modality: str) -> Dict:
    with np.load(features_npz(cfg, modality)) as z:
        return {k[4:]: str(z[k]) for k in z.files if k.startswith("_bp_")}


def _params(cfg: Dict, modality: str) -> Dict:
    dino_cfg = cfg["dinov3"]
    return {"stage": "main_dinov3_heads", "version": HEADS_VERSION, "features": _features_identity(cfg, modality), "dev": digest_file(dev_csv(cfg, modality)),
            "questions": digest_file(questions_csv(cfg, modality)),
            "fit": {"c_grid": [float(c) for c in dino_cfg["c_grid"]], "cv_folds": int(dino_cfg["cv_folds"]), "max_iter": int(dino_cfg["max_iter"]), "seed": int(dino_cfg["seed"])}}


def _rows_of(feats: np.ndarray, index: Dict[str, int], case_ids, what: str) -> np.ndarray:
    missing = [c for c in case_ids if c not in index]
    if missing:
        raise MissingInput(f"{len(missing)} {what} image(s) have no feature row (first: {missing[0]}); run main_dinov3_cache")
    return feats[[index[c] for c in case_ids]]


def _fit_finding(cfg: Dict, modality: str, finding: str, feats: np.ndarray, index: Dict[str, int], dev: pd.DataFrame, q: pd.DataFrame) -> List[Dict]:
    dino_cfg = cfg["dinov3"]
    qf = q[q["finding"] == finding].sort_values("question_id", kind="mergesort")
    if qf.empty:
        return [{"finding": finding, "status": "no_questions"}]
    fcfg = cfg["question_sets"]["modalities"][modality]["findings"][finding]
    dataset = fcfg.get("dev_dataset", fcfg["dataset"])
    if "dev" not in cfg["datasets"][dataset]["partitions"]:
        return [{"finding": finding, "status": "no_development_partition"}]
    if not (read_partition(cfg, dataset)["partition"] == "dev").any():
        raise MissingInput(f"{finding}: {dataset} has no development partition yet")
    d = dev[dev["finding"] == finding].sort_values("case_id", kind="mergesort")
    per_class = d.groupby("label")["group_id"].nunique().reindex([0, 1]).fillna(0).astype(int)
    if per_class.min() < int(dino_cfg["cv_folds"]):
        return [{"finding": finding, "status": "too_few_groups", "n_pos_groups": int(per_class[1]), "n_neg_groups": int(per_class[0])}]
    X = _rows_of(feats, index, d["case_id"].tolist(), "development")
    head = fit_head(X, d["label"].to_numpy(int), d["group_id"].to_numpy(), dino_cfg)
    p = head["model"].predict_proba(_rows_of(feats, index, qf["case_id"].tolist(), "evaluation"))[:, 1]
    scaler, clf = head["model"][0], head["model"][-1]
    write_npz_atomic(head_npz(cfg, modality, finding), coef=clf.coef_[0], intercept=clf.intercept_, mean=scaler.mean_, scale=scaler.scale_,
                     C=np.array(head["C"]), grid=np.array(head["grid"]), cv_auroc=np.array(head["cv_auroc"]),
                     n_convergence_warnings=np.array(head["n_convergence_warnings"]), n_dev=np.array([per_class[0], per_class[1]]),
                     question_ids=np.array(qf["question_id"].tolist()), p=p)
    return [{"question_id": r.question_id, "finding": finding, "case_id": r.case_id, "label": int(r.label), "p": float(pi), "C": head["C"],
             "status": "fitted"} for r, pi in zip(qf.itertuples(index=False), p)]


def _write_reference(cfg: Dict, modality: str, rows: pd.DataFrame, params: Dict) -> None:
    path = reference_path(cfg, modality)
    stamp = {"heads": params, "findings": sorted(rows["finding"].unique().tolist())}
    if os.path.exists(path) and read_build_params(path) == json.loads(json.dumps(stamp)):
        return
    digest = hashlib.blake2b(json.dumps(stamp, sort_keys=True).encode("ascii"), digest_size=8).hexdigest()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    records = [{"item_id": r.question_id, "question_id": r.question_id, "finding": r.finding, "label": int(r.label), "image_label": int(r.label),
                "image_case_id": r.case_id, "repeat": 0, "input_key": f"dinov3|{r.case_id}", "request_digest": digest, "raw_answer": "",
                "parsed_answer": int(r.p >= 0.5), "answer_class": "yes" if r.p >= 0.5 else "no", "p_yes": float(r.p), "p_yes_source": "head",
                "finished_at": now} for r in rows.sort_values("question_id", kind="mergesort").itertuples(index=False)]
    write_jsonl_atomic(path, records)
    write_build_params(path, stamp)


def main_dinov3_heads(cfg_path: str) -> None:
    require_packages(["sklearn"], "main_dinov3_heads")
    cfg = read_config(cfg_path)[PROJECT]
    status = status_path(cfg, "main_dinov3_heads")
    skipped, held = [], []

    def finish(modality, findings, feats, index, dev, q, params, wait):
        rows = run_units_resumable(os.path.join(dinov3_dir(cfg), "heads", "partials", modality), f"heads_{modality}", findings,
                                   lambda f: _fit_finding(cfg, modality, f, feats, index, dev, q), progress_desc=f"[main_dinov3_heads] {modality}",
                                   require_rows=True, use_claims=True, status_file=status, build_params=params, wait=wait)
        fitted = rows[rows["status"] == "fitted"] if len(rows) else rows
        if len(fitted):
            _write_reference(cfg, modality, fitted, params)
        if rows.attrs.get("n_skipped"):
            skipped.append(f"{modality} ({rows.attrs['n_skipped']} of {len(findings)} finding(s) without a head)")
        append_status(status, f"main_dinov3_heads {modality}: {fitted['finding'].nunique() if len(fitted) else 0} head(s)")
    for modality, mcfg in cfg["question_sets"]["modalities"].items():
        if not (os.path.exists(features_npz(cfg, modality)) and os.path.exists(dev_csv(cfg, modality))):
            skipped.append(f"{modality} (no features; run main_dinov3_cache)")
            continue
        params = _params(cfg, modality)
        with np.load(features_npz(cfg, modality)) as z:
            feats, index = z["embeddings"], {str(c): i for i, c in enumerate(z["case_ids"])}
        dev = read_csv_defensively(dev_csv(cfg, modality), dtype={"case_id": str, "group_id": str}, keep_default_na=False)
        q = read_csv_defensively(questions_csv(cfg, modality), usecols=["question_id", "finding", "case_id", "label"], dtype={"case_id": str},
                                 keep_default_na=False)
        work = (modality, list(mcfg["findings"]), feats, index, dev, q, params)
        try:
            finish(*work, False)
        except ClaimedElsewhere:
            held.append(work)
        except MissingInput as e:
            skipped.append(f"{modality} ({e})")
    for work in held:
        try:
            finish(*work, True)
        except MissingInput as e:
            skipped.append(f"{work[0]} ({e})")
    if skipped:
        raise MissingInput(f"[main_dinov3_heads] {'; '.join(skipped)}")
