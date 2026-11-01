"""
analysis/benchmarks.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.compute_metrics import complete_macro_auroc
from analysis.joint_bootstrap import replicate_counts, summarize, weighted_rate
from analysis.records import unit_answers
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, single_shot_line, stage_files
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import BENCHMARK_PREFIX, benchmark_sets
from Inference.inference_runner import BENCHMARK_CONDITIONS, SetData
from Inference.report_utils import add_fdr, metric_row, stat_row
from Inference.resume_utils import MissingInput, append_status, status_path
from Inference.stats_utils import shift_reflect_p, spearman


ALL = "all"


def _frame(sd: SetData) -> pd.DataFrame:
    return sd.manifest.sort_values("question_id", kind="mergesort").reset_index(drop=True)


def _scopes(q: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    out = [(ALL, np.ones(len(q), dtype=bool))]
    for m in sorted(set(q["study_modality"].astype(str)) - {""}):
        out.append((m, (q["study_modality"].astype(str) == m).to_numpy()))
    return out


def _correct(cfg: Dict, sd: SetData, model: str, q: pd.DataFrame) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    label = q["label"].to_numpy(dtype=float)
    out = {}
    for cond in BENCHMARK_CONDITIONS:
        a = unit_answers(cfg, sd, model, q, cond)
        if a is None:
            return None, f"no {cond} records"
        if a["open"] or a["stale_parser"]:
            return None, f"{a['open']} open and {a['stale_parser']} older-parser records in {cond}"
        out[cond] = (a["ans"] == label).astype(float)
    return out, ""


def benchmark_points(cfg: Dict, full: Dict) -> Dict[Tuple[str, str], Dict[str, float]]:
    out = {}
    for set_name in benchmark_sets(cfg):
        try:
            sd = SetData(full, set_name)
        except MissingInput:
            continue
        q = _frame(sd)
        acc: Dict[str, Dict[str, float]] = {}
        for model in cfg["models"]:
            correct, reason = _correct(cfg, sd, model, q)
            if correct is None:
                acc = {}
                break
            for scope, m in _scopes(q):
                acc.setdefault(scope, {})[model] = float(correct["original"][m].mean())
        for scope, by_model in acc.items():
            out[(set_name[len(BENCHMARK_PREFIX):], scope)] = by_model
    return out


def _macro_auroc(cfg: Dict, set_name: str) -> Dict[str, float]:
    return complete_macro_auroc(cfg, set_name)


def benchmarks_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["benchmarks_dir"], f"{name}.csv")


@single_shot_line("benchmarks_analysis", lambda cfg: stage_files(cfg, "records", "caches", "manifests"))
def main_benchmarks_analysis(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "benchmarks_analysis")
    stats = cfg["stats"]
    rows, gaps, accuracy = [], [], {}
    for set_name in benchmark_sets(cfg):
        name = set_name[len(BENCHMARK_PREFIX):]
        try:
            sd = SetData(full, set_name)
        except MissingInput as e:
            gaps.append(f"{set_name} ({e})")
            continue
        q = _frame(sd)
        counts, rhash = replicate_counts(q["group_id"].astype(str).tolist(), int(stats["n_boot"]), int(stats["boot_seed"]))
        for model in cfg["models"]:
            correct, reason = _correct(cfg, sd, model, q)
            if correct is None:
                gaps.append(f"{set_name}/{model} ({reason})")
                continue
            for scope, m in _scopes(q):
                with_image, no_image = weighted_rate(counts, correct["original"], m), weighted_rate(counts, correct["no_image"], m)
                diff = (with_image[0] - no_image[0], with_image[1] - no_image[1], with_image[2])
                accuracy.setdefault((name, scope), {})[model] = with_image[0]
                for measure, (point, boot, n), unit in (("accuracy_image", with_image, "percent"), ("accuracy_no_image", no_image, "percent"),
                                                         ("accuracy_image_minus_no_image", diff, "percentage_points")):
                    s = summarize(boot)
                    row = {"benchmark": name, "scope": scope, "model": model, "measure": measure}
                    row.update(metric_row(s["mean"], s["sd"], s["lo"], s["hi"], n, unit, f"{name}/{scope}/{model}/{measure}"))
                    row.update({"point_raw": point, "p_raw": shift_reflect_p(boot, point) if unit == "percentage_points" else np.nan,
                                "n_replicates": s["n_finite"], "resample_hash": rhash})
                    rows.append(row)
    corr, matched = [], {str(b): list(ms) for b, ms in cfg["benchmarks"]["matching"]}
    aurocs = {}
    for (name, scope), by_model in sorted(accuracy.items()):
        targets = matched.get(name, []) if scope == ALL else [scope]
        if name not in matched and scope == ALL:
            continue
        for modality in targets:
            if modality not in aurocs:
                aurocs[modality] = _macro_auroc(cfg, modality)
            missing = [m for m in cfg["models"] if m not in aurocs[modality]]
            if missing:
                gaps.append(f"{modality}: macro AUROC of {len(missing)} model(s) (first {missing[0]})")
                continue
            models = list(cfg["models"])
            if any(m not in by_model for m in models):
                continue
            rho, p, n = spearman(np.array([by_model[m] for m in models]), np.array([aurocs[modality][m] for m in models]))
            corr.append({"benchmark": name, "scope": scope, "modality": modality, "measure": "spearman_accuracy_vs_macro_auroc", **stat_row(rho, p, n)})
    if gaps:
        raise MissingInput(f"[benchmarks_analysis] {len(gaps)} benchmark(s), model(s), or modality(ies) are not measured yet, so nothing is written (first: {'; '.join(gaps[:4])})")
    table = add_fdr(pd.DataFrame(rows), ["benchmark", "measure"])
    write_csv_if_changed(table, benchmarks_csv(cfg, "benchmark_accuracy"))
    write_csv_if_changed(add_fdr(pd.DataFrame(corr), ["benchmark"]), benchmarks_csv(cfg, "benchmark_correlations"))
    append_status(status, f"benchmarks_analysis: {len(table)} accuracy rows, {len(corr)} correlations")
