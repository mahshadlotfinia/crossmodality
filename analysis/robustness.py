"""
analysis/robustness.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.compute_metrics import question_frame
from analysis.joint_bootstrap import replicate_counts, summarize, weighted_auroc, weighted_rate
from analysis.records import unit_answers
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, single_shot_line, stage_files
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import modality_sets
from Inference.inference_runner import SetData
from Inference.report_utils import add_fdr, metric_row, stat_row
from Inference.resume_utils import MissingInput, append_status, status_path
from Inference.stats_utils import shift_reflect_p, spearman


Series = Dict[str, Tuple[float, np.ndarray, int]]


def _auroc_by_finding(counts: np.ndarray, q: pd.DataFrame, p: np.ndarray, subset: np.ndarray) -> Optional[Series]:
    if not np.isfinite(p[subset]).all():
        return None
    label = q["label"].to_numpy(dtype=float)
    out = {}
    for f in sorted(set(q.loc[subset, "finding"])):
        m = subset & (q["finding"] == f).to_numpy()
        out[f] = weighted_auroc(counts, label, p, m)
    return out


def _macro(series: Series) -> Tuple[float, np.ndarray, int]:
    vals = list(series.values())
    return float(np.mean([v[0] for v in vals])), np.mean(np.vstack([v[1] for v in vals]), axis=0), int(sum(v[2] for v in vals))


def _metric(axis: str, set_name: str, model: str, measure: str, value: Tuple[float, np.ndarray, int], unit: str, family: str = "") -> Dict:
    point, boot, n = value
    s = summarize(boot)
    row = {"axis": axis, "set": set_name, "model": model, "measure": measure, "family": family}
    row.update(metric_row(s["mean"], s["sd"], s["lo"], s["hi"], n, unit, f"{axis}/{set_name}/{model}/{measure}"))
    row.update({"point_raw": point, "p_raw": shift_reflect_p(boot, point) if unit == "percentage_points" else np.nan, "n_replicates": s["n_finite"]})
    return row


def _complete(a: Optional[Dict]) -> bool:
    return a is not None and not a["open"] and not a["stale_parser"]


def robustness_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["robustness_dir"], f"{name}.csv")


@single_shot_line("robustness", lambda cfg: stage_files(cfg, "records", "manifests"))
def main_robustness(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "robustness")
    stats = cfg["stats"]
    robust_models = list(cfg["runs"]["robustness_models"])
    c = cfg["contrasts"]
    metrics, corr, gaps = [], [], []
    for set_name in modality_sets(cfg):
        try:
            sd = SetData(full, set_name)
        except MissingInput as e:
            gaps.append(f"{set_name} ({e})")
            continue
        q = question_frame(sd)
        counts, _ = replicate_counts(q["cluster"].tolist(), int(stats["n_boot"]), int(stats["boot_seed"]))
        phr = q["question_id"].isin(sd.subsets["in_phrasing"]).to_numpy()
        det = q["question_id"].isin(sd.subsets["in_determinism"]).to_numpy()
        default, phrased = {}, {}
        for model in cfg["models"]:
            a0, ap = unit_answers(cfg, sd, model, q, "original"), unit_answers(cfg, sd, model, q, "original", "phrasing")
            if not (_complete(a0) and _complete(ap)):
                gaps.append(f"{set_name}/{model} phrasing")
                continue
            d, p_ = _auroc_by_finding(counts, q, a0["p"], phr), _auroc_by_finding(counts, q, ap["p"], phr)
            if d is None or p_ is None:
                gaps.append(f"{set_name}/{model} phrasing (P(Yes) missing)")
                continue
            default[model], phrased[model] = d, p_
            metrics.append(_metric("phrasing", set_name, model, "macro_auroc_second_phrasing", _macro(p_), "percent"))
            ad = unit_answers(cfg, sd, model, q, "original", "determinism")
            if _complete(ad):
                same = np.array([float(len(ad["raw"].get(qid, [])) == int(cfg["runs"]["determinism_repeats"]) and len(set(ad["raw"][qid])) == 1)
                                 if qid in ad["raw"] else np.nan for qid in q["question_id"]])
                metrics.append(_metric("determinism", set_name, model, "share_identical_outputs", weighted_rate(counts, same, det), "percent"))
            else:
                gaps.append(f"{set_name}/{model} determinism")
            if model not in robust_models:
                continue
            small = unit_answers(cfg, sd, model, q, "original_small")
            if _complete(small) and _auroc_by_finding(counts, q, small["p"], phr) is not None:
                main_auc, small_auc = _macro(d), _macro(_auroc_by_finding(counts, q, small["p"], phr))
                metrics.append(_metric("resolution", set_name, model, "macro_auroc_small_minus_main",
                                       (small_auc[0] - main_auc[0], small_auc[1] - main_auc[1], small_auc[2]), "percentage_points", "model"))
            else:
                gaps.append(f"{set_name}/{model} resolution")
        if len(default) == len(cfg["models"]):
            models = list(cfg["models"])
            rho, p, n = spearman(np.array([_macro(default[m])[0] for m in models]), np.array([_macro(phrased[m])[0] for m in models]))
            corr.append({"axis": "phrasing", "set": set_name, "measure": "spearman_macro_auroc_main_vs_second_phrasing", **stat_row(rho, p, n)})
            for family, pairs in (("B", [(e, l, kind) for e, l, kind in c["training"]]),):
                for earlier, later, label in pairs:
                    a, b = phrased[earlier], phrased[later]
                    common = sorted(set(a) & set(b))
                    if not common:
                        continue
                    diff = (float(np.mean([b[f][0] - a[f][0] for f in common])), np.mean(np.vstack([b[f][1] - a[f][1] for f in common]), axis=0),
                            int(sum(a[f][2] for f in common)))
                    metrics.append(_metric("phrasing", set_name, f"{later} - {earlier}", f"{label}, second phrasing", diff, "percentage_points", family))
    if gaps:
        raise MissingInput(f"[robustness] {len(gaps)} unit(s) still wait, so nothing is written (first: {'; '.join(gaps[:4])})")
    write_csv_if_changed(add_fdr(pd.DataFrame(metrics), ["axis", "family"]), robustness_csv(cfg, "robustness_metrics"))
    write_csv_if_changed(add_fdr(pd.DataFrame(corr)), robustness_csv(cfg, "robustness_rank_agreement"))
    append_status(status, f"robustness: {len(metrics)} metric rows, {len(corr)} rank agreements")
