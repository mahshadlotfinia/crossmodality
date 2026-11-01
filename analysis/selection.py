"""
analysis/selection.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.compute_metrics import boot_npz, load_boot, question_frame
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, run_build_unit, run_build_units
from data_loader.build_utils import write_csv_if_changed
from Inference.inference_runner import SetData
from Inference.model_facts import facts_csv
from Inference.report_utils import metric_row, stat_row
from Inference.resume_utils import MissingInput, append_status, digest_file, status_path, write_csv_atomic
from Inference.serving_utils import require_packages
from Inference.stats_utils import auroc_batch, kendall_tau_b, stratified_group_folds


SELECTION_VERSION = 2


def panel_scores(cfg: Dict, set_name: str, q: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str]]:
    models, rows, missing = [], [], []
    for m in cfg["models"]:
        path = boot_npz(cfg, set_name, m)
        if not os.path.exists(path):
            missing.append(m)
            continue
        z = load_boot(path)
        if [str(x) for x in z["question_ids"]] != q["question_id"].tolist() or not np.isfinite(z["p_original"]).all():
            missing.append(m)
            continue
        models.append(m)
        rows.append(z["p_original"].astype(float))
    return models, (np.vstack(rows) if rows else np.zeros((0, len(q)))), missing


def macro_auroc(P: np.ndarray, y: np.ndarray, finding: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = []
    for f in np.unique(finding[idx]):
        sub = idx[finding[idx] == f]
        out.append(auroc_batch(np.tile(y[sub], (P.shape[0], 1)), P[:, sub]))
    return np.nanmean(np.vstack(out), axis=0)


def split_halves(q: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    strata = (q["finding"].astype(str) + "|" + q["label"].astype(str)).to_numpy()
    sel, held = stratified_group_folds(strata, q["cluster"].to_numpy(), 2, seed)[0]
    return np.sort(sel), np.sort(held)


def draw_questions(q: pd.DataFrame, sel: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    finding, label = q["finding"].to_numpy(), q["label"].to_numpy()
    for f in np.unique(finding[sel]):
        for lab in (1, 0):
            pool = sel[(finding[sel] == f) & (label[sel] == lab)]
            k = min(n // 2, pool.size)
            out.append(np.sort(rng.choice(pool, size=k, replace=False)) if k < pool.size else pool)
    return np.sort(np.concatenate(out))


def _best(scores: Optional[Dict[str, float]], models: List[str]) -> Optional[str]:
    if not scores or any(m not in scores for m in models):
        return None
    top = max(scores[m] for m in models)
    return next(m for m in models if scores[m] == top)


def fixed_rules(cfg: Dict, models: List[str], set_name: str, bench: Dict) -> Dict[str, Optional[str]]:
    sel_cfg = cfg["selection"]
    out: Dict[str, Optional[str]] = {"largest": None, "newest": None}
    if os.path.exists(facts_csv(cfg)):
        facts = pd.read_csv(facts_csv(cfg), low_memory=False).set_index("model")
        facts = facts.loc[[m for m in models if m in facts.index]]
        if len(facts) == len(models):
            out = {"largest": str(facts["parameters"].idxmax()), "newest": str(facts["created_at"].astype(str).idxmax())}
    out["omnimedvqa"] = _best(bench.get((sel_cfg["omnimedvqa_benchmark"], set_name)), models)
    parts = [bench.get((b, "all")) for b in sel_cfg["benchmark_mean_sets"]]
    mean = {m: float(np.mean([p[m] for p in parts])) for m in models} if all(p and all(m in p for m in models) for p in parts) else None
    out["benchmark_mean"] = _best(mean, models)
    return out


def run_draws(cfg: Dict, q: pd.DataFrame, models: List[str], P: np.ndarray, rules: Dict[str, Optional[str]]) -> pd.DataFrame:
    from tqdm import tqdm
    sel_cfg = cfg["selection"]
    y, finding = q["label"].to_numpy(dtype=float), q["finding"].to_numpy()
    rules = {k: models.index(v) for k, v in rules.items() if v is not None}
    rng = np.random.default_rng(int(sel_cfg["seed"]))
    rows = []
    for d in tqdm(range(int(sel_cfg["n_draws"])), desc="[main_selection] draws", unit="draw", leave=False):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        sel, held = split_halves(q, seed)
        held_auc = macro_auroc(P, y, finding, held)
        best = float(np.nanmax(held_auc))
        local = np.random.default_rng(seed)
        rule_regret = {f"regret_{k}": best - float(held_auc[i]) for k, i in rules.items()}
        for n in sel_cfg["n_per_finding"]:
            sel_auc = macro_auroc(P, y, finding, draw_questions(q, sel, int(n), local))
            tops = np.flatnonzero(sel_auc == np.nanmax(sel_auc))
            top = int(local.choice(tops))
            rows.append({"draw": d, "n": int(n), "top_model": models[top], "regret": best - float(held_auc[top]),
                         "tau_b": kendall_tau_b(sel_auc, held_auc), **rule_regret})
    return pd.DataFrame(rows)


def summarize_draws(cfg: Dict, set_name: str, draws: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics, stats = [], []
    for n, g in draws.groupby("n"):
        for col in [c for c in g.columns if c.startswith("regret")]:
            v = g[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                rule = "selection" if col == "regret" else col[len("regret_"):]
                metrics.append({"set": set_name, "n_per_finding": int(n), "rule": rule, **metric_row(v.mean(), v.std(ddof=1), np.percentile(v, 2.5), np.percentile(v, 97.5),
                                                                                   int(v.size), "percentage_points", f"{set_name}/{n}/{col}", draws=True)})
        tau = g["tau_b"].to_numpy(dtype=float)
        tau = tau[np.isfinite(tau)]
        stats.append({"set": set_name, "n_per_finding": int(n), "measure": "tau_b_median", **stat_row(np.median(tau), np.nan, int(tau.size))})
    return pd.DataFrame(metrics), pd.DataFrame(stats)


def selection_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["selection_dir"], f"{name}.csv")


def main_selection(cfg_path: str) -> None:
    require_packages(["scipy", "sklearn"], "main_selection")
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "main_selection")
    from analysis.benchmarks import benchmark_points
    all_metrics, all_stats, skipped, attempts, ready = [], [], [], [], []
    bench = benchmark_points(cfg, full)
    for set_name in cfg["question_sets"]["modalities"]:
        try:
            q = question_frame(SetData(full, set_name))
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
            continue
        models, P, missing = panel_scores(cfg, set_name, q)
        if missing:
            skipped.append(f"{set_name} ({len(models)} of {len(cfg['models'])} models measured; first missing: {missing[0]})")
            continue
        path = selection_csv(cfg, f"draws_{set_name}")
        rules = fixed_rules(cfg, models, set_name, bench)
        params = {"version": SELECTION_VERSION, "selection": cfg["selection"], "models": models,
                  "caches": {m: digest_file(boot_npz(cfg, set_name, m)) for m in models}, "facts": digest_file(facts_csv(cfg)), "rules": rules}

        def build(q=q, models=models, P=P, path=path, set_name=set_name, missing=missing, rules=rules):
            write_csv_atomic(run_draws(cfg, q, models, P, rules), path)
        attempts.append((f"main_selection__{set_name}", lambda wait, set_name=set_name, path=path, params=params, build=build:
                         run_build_unit(cfg, f"main_selection__{set_name}", path, params, lambda: True, build, wait=wait)))
        ready.append((set_name, path))
    run_build_units(attempts, "[main_selection] sets")
    for set_name, path in ready:
        metrics, stats = summarize_draws(cfg, set_name, pd.read_csv(path, low_memory=False))
        all_metrics.append(metrics)
        all_stats.append(stats)
        append_status(status, f"main_selection {set_name}: draws summarized")
    if all_metrics:
        metrics, stats = pd.concat(all_metrics, ignore_index=True), pd.concat(all_stats, ignore_index=True)
        write_csv_if_changed(metrics, selection_csv(cfg, "selection_regret"))
        write_csv_if_changed(stats, selection_csv(cfg, "selection_tau"))
    if skipped:
        raise MissingInput(f"[main_selection] {len(skipped)} set(s) without the selection test: {'; '.join(skipped)}")
