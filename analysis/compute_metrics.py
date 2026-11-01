"""
analysis/compute_metrics.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import functools
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.joint_bootstrap import replicate_counts, summarize, weighted_auroc
from analysis.metrics_core import answer_counts, finding_series, macro_series
from analysis.records import file_fingerprint, reference_arrays, served_arrays, unit_files
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, manifest_csv, run_build_unit, run_build_units
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import asked_findings, external_sets, modality_sets
from encoders.heads import reference_path
from Inference.inference_runner import SetData, conditions_of
from Inference.parser import parser_version
from Inference.report_utils import metric_row
from Inference.resume_utils import ClaimedElsewhere, MissingInput, append_status, digest_file, status_path, write_npz_atomic
from Inference.serving_utils import require_packages
from Inference.stats_utils import resample_hash


METRICS_VERSION = 3


def boot_npz(cfg: Dict, set_name: str, model: str) -> str:
    return os.path.join(cfg["outputs"]["metrics_dir"], "boot", set_name, f"{model}.npz")


def metrics_csv(cfg: Dict, set_name: str, kind: str) -> str:
    return os.path.join(cfg["outputs"]["metrics_dir"], f"{kind}_{set_name}.csv")


def question_frame(sd: SetData) -> pd.DataFrame:
    q = sd.manifest.sort_values("question_id", kind="mergesort").reset_index(drop=True)
    return q.assign(cluster=q["dataset"].astype(str) + "|" + q["group_id"].astype(str))


def load_boot(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def complete_macro_auroc(cfg: Dict, set_name: str) -> Dict[str, float]:
    expected = set(asked_findings(cfg, set_name))
    out = {}
    for model in cfg["models"]:
        path = boot_npz(cfg, set_name, model)
        if not os.path.exists(path):
            continue
        z = load_boot(path)
        opened = json.loads(str(z.get("units", "{}"))).get("open", {}).get("original")
        if opened != 0 or "n_findings" not in z:
            continue
        values = [(str(s), float(p), int(nf)) for m, s, p, nf in zip(z["metrics"], z["scopes"], z["point"], z["n_findings"]) if m == "auroc"]
        findings = {s for s, p, _ in values if s != "macro" and np.isfinite(p)}
        hit = [p for s, p, nf in values if s == "macro" and nf == len(expected)]
        if expected and findings == expected and len(hit) == 1 and np.isfinite(hit[0]):
            out[model] = hit[0]
    return out


def _arrays(cfg: Dict, sd: SetData, model: str, q: pd.DataFrame) -> Optional[Dict]:
    if model == cfg["dinov3"]["model_key"]:
        return reference_arrays(cfg, sd.set_name, q)
    return served_arrays(cfg, sd, model, q, conditions_of(cfg, sd.set_name))


def stratum_series(arr: Dict, q: pd.DataFrame, counts: np.ndarray) -> List:
    p0 = arr["p"].get("original")
    if "stratum" not in q.columns or p0 is None:
        return []
    label = q["label"].to_numpy(dtype=float)
    out = []
    for f in dict.fromkeys(q["finding"]):
        for s in sorted(set(q.loc[q["finding"] == f, "stratum"].astype(str))):
            m = ((q["finding"] == f) & (q["stratum"].astype(str) == s)).to_numpy()
            if np.isfinite(p0[m]).all():
                out.append(("auroc_stratum", f"{f}|{s}", *weighted_auroc(counts, label, p0, m), 1))
    return out


def _build(cfg: Dict, sd: SetData, model: str, q: pd.DataFrame, counts: np.ndarray, rhash: str) -> None:
    arr = _arrays(cfg, sd, model, q)
    if arr is None:
        raise MissingInput(f"{sd.set_name}/{model}: the reference records do not match the current question set; run main_dinov3_heads")
    stale = {c: n for c, n in arr["stale_parser"].items() if n}
    if stale:
        raise MissingInput(f"{sd.set_name}/{model}: records of an older parser in {stale}; run main_reparse_records first")
    findings = list(dict.fromkeys(q["finding"]))
    per_finding = [s for f in findings for s in finding_series(arr, q, counts, f)]
    series = per_finding + macro_series(per_finding) + stratum_series(arr, q, counts)
    write_npz_atomic(boot_npz(cfg, sd.set_name, model), metrics=np.array([s[0] for s in series]), scopes=np.array([s[1] for s in series]),
                     point=np.array([s[2] for s in series], dtype=float), boot=np.vstack([s[3] for s in series]) if series else np.zeros((0, counts.shape[0])),
                     n=np.array([s[4] for s in series], dtype=int), n_findings=np.array([s[5] for s in series], dtype=int),
                     resample_hash=np.array(rhash), answers=np.array(json.dumps(answer_counts(arr, q))),
                     question_ids=np.array(q["question_id"].tolist()), p_original=arr["p"].get("original", np.full(len(q), np.nan)),
                     units=np.array(json.dumps({"open": arr["open"]})))


def _models_with_records(cfg: Dict, set_name: str) -> List[Tuple[str, List[str]]]:
    out = [(m, unit_files(cfg, set_name, m, conditions_of(cfg, set_name))) for m in cfg["models"]]
    out = [(m, files) for m, files in out if any(os.path.exists(p) for p in files)]
    ref = reference_path(cfg, set_name)
    if os.path.exists(ref):
        out.append((cfg["dinov3"]["model_key"], [ref]))
    return out


def _tables(cfg: Dict, set_name: str, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, answers, units = [], [], []
    total = len(asked_findings(cfg, set_name))
    for model in models:
        path = boot_npz(cfg, set_name, model)
        if not os.path.exists(path):
            continue
        z = load_boot(path)
        for metric, scope, point, boot, n, nf in zip(z["metrics"], z["scopes"], z["point"], z["boot"], z["n"], z["n_findings"]):
            s = summarize(boot)
            unit = "percent"
            row = {"set": set_name, "model": model, "metric": str(metric), "scope": str(scope), "n_findings": int(nf), "n_findings_total": total}
            row.update(metric_row(s["mean"], s["sd"], s["lo"], s["hi"], int(n), unit, f"{set_name}/{model}/{metric}/{scope}"))
            row.update({"point_raw": float(point), "n_replicates": s["n_finite"], "resample_hash": str(z["resample_hash"])})
            rows.append(row)
        answers += [dict(r, set=set_name, model=model) for r in json.loads(str(z["answers"]))]
        u = json.loads(str(z["units"]))
        units += [{"set": set_name, "model": model, "condition": c, "open_items": int(v)} for c, v in u["open"].items()]
    return pd.DataFrame(rows), pd.DataFrame(answers), pd.DataFrame(units)


def main_compute_metrics(cfg_path: str) -> None:
    require_packages(["scipy"], "main_compute_metrics")
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "main_compute_metrics")
    n_boot, seed = int(cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"])
    skipped, attempts, ready = [], [], []

    def attempt(sd, q, model, files, wait):
        groups = q["cluster"].tolist()
        params = {"version": METRICS_VERSION, "resamples": resample_hash(n_boot, seed, groups), "records": file_fingerprint(files),
                  "manifest": digest_file(manifest_csv(cfg, sd.set_name)), "parser": parser_version(cfg["parser"])}

        def build():
            counts, rhash = replicate_counts(groups, n_boot, seed)
            _build(cfg, sd, model, q, counts, rhash)
        try:
            run_build_unit(cfg, f"main_compute_metrics__{sd.set_name}__{model}", boot_npz(cfg, sd.set_name, model), params, lambda: True, build, wait=wait)
        except ClaimedElsewhere:
            raise
        except MissingInput as e:
            skipped.append(str(e))
    for set_name in modality_sets(cfg) + external_sets(cfg):
        try:
            sd = SetData(full, set_name)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
            continue
        q = question_frame(sd)
        models = _models_with_records(cfg, set_name)
        ready.append((set_name, models))
        attempts += [(f"main_compute_metrics__{set_name}__{m}", functools.partial(attempt, sd, q, m, f)) for m, f in models]
    run_build_units(attempts, "[main_compute_metrics] caches")
    for set_name, models in ready:
        metrics, answers, units = _tables(cfg, set_name, [m for m, _ in models])
        for kind, table in (("metrics", metrics), ("answers", answers), ("units", units)):
            write_csv_if_changed(table, metrics_csv(cfg, set_name, kind))
        append_status(status, f"main_compute_metrics {set_name}: {len(metrics)} metric rows from {len(models)} model(s)")
    if skipped:
        raise MissingInput(f"[main_compute_metrics] {len(skipped)} set(s) or model(s) not measured: {'; '.join(skipped)}")
