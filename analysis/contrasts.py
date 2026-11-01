"""
analysis/contrasts.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from analysis.compute_metrics import boot_npz, complete_macro_auroc, load_boot
from analysis.joint_bootstrap import summarize
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, single_shot_line, stage_files
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import asked_findings
from Inference.model_facts import facts_csv
from Inference.report_utils import add_fdr, metric_row, stat_row
from Inference.resume_utils import MissingInput, append_status, status_path
from Inference.serving_utils import require_packages
from Inference.stats_utils import N_BOOT, shift_reflect_p


NEEDS = {"auroc": ("original",)}
Findings = Dict[str, Tuple[float, np.ndarray, int]]
Exclude = Callable[[str, List[str]], Set[str]]
UNDEFINED = (float("nan"), np.full(N_BOOT, np.nan), 0, 0)


def size_ladders(c: Dict) -> List:
    return list(c["ladders"]) + list(c["secondary_ladders"])


def finding_values(cfg: Dict, set_name: str, model: str, metric: str, hashes: Dict[str, str], scope_suffix: Optional[str] = None) -> Optional[Findings]:
    path = boot_npz(cfg, set_name, model)
    if not os.path.exists(path):
        return None
    z = load_boot(path)
    hashes.setdefault(set_name, str(z["resample_hash"]))
    if str(z["resample_hash"]) != hashes[set_name]:
        raise ValueError(f"[main_contrasts] {set_name}/{model} was resampled differently from the other models of the set; run main_compute_metrics again")
    return {str(sc): (float(pt), b, int(n)) for m, sc, pt, b, n in zip(z["metrics"], z["scopes"], z["point"], z["boot"], z["n"])
            if m == metric and sc != "macro" and np.isfinite(pt) and (scope_suffix is None or str(sc).endswith(scope_suffix))}


def open_items(cfg: Dict, set_name: str, model: str, metric: str) -> Optional[int]:
    path = boot_npz(cfg, set_name, model)
    if not os.path.exists(path):
        return None
    units = json.loads(str(load_boot(path)["units"]))["open"]
    return int(sum(units.get(c, 0) for c in NEEDS[metric]))


def macro_difference(a: Findings, b: Findings) -> Tuple[float, np.ndarray, int, int]:
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), np.full(N_BOOT, np.nan), 0, 0
    point = float(np.mean([b[f][0] - a[f][0] for f in common]))
    boot = np.mean(np.vstack([b[f][1] - a[f][1] for f in common]), axis=0)
    return point, boot, int(sum(a[f][2] for f in common)), len(common)


def contrast_row(family: str, metric: str, set_name: str, name: str, diff: Tuple[float, np.ndarray, int, int], models: str, total: int) -> Dict:
    point, boot, n, nf = diff
    s = summarize(boot)
    row = {"family": family, "metric": metric, "set": set_name, "contrast": name, "models": models, "n_findings": nf, "n_findings_total": total}
    row.update(metric_row(s["mean"], s["sd"], s["lo"], s["hi"], n, "percentage_points", f"{family}/{metric}/{set_name}/{name}"))
    row.update({"point_raw": point, "p_raw": shift_reflect_p(boot, point), "n_replicates": s["n_finite"],
                "reason": "" if nf else "undefined in every finding for one of the two models"})
    return row


def contrasts_for(cfg: Dict, set_name: str, metric: str, hashes: Dict[str, str], scope_suffix: Optional[str] = None,
                  exclude: Optional[Exclude] = None) -> List[Dict]:
    c = cfg["contrasts"]
    total = len(asked_findings(cfg, set_name))
    cut = len(scope_suffix or "")
    cache: Dict[str, Optional[Findings]] = {}
    rows, ladder_changes, training_changes = [], [], []

    def get(m):
        if m not in cache:
            cache[m] = finding_values(cfg, set_name, m, metric, hashes, scope_suffix)
        return cache[m]

    def add(family, name, lo, hi, models):
        a, b = get(lo), get(hi)
        if a is None or b is None:
            return None
        left_out = exclude(set_name, [lo, hi]) if exclude is not None else set()
        shared = set(a) & set(b)
        a, b = ({f: v for f, v in x.items() if f[:len(f) - cut] not in left_out} for x in (a, b))
        diff = macro_difference(a, b)
        row = contrast_row(family, metric, set_name, name, diff, models, total)
        if exclude is not None:
            row["n_findings_marked"] = len(left_out)
            if shared and not diff[3]:
                row["reason"] = "every finding both models define is marked for one of them"
        rows.append(row)
        return diff

    for family_name, members in c["ladders"]:
        diff = add("A", f"{family_name} largest minus smallest", members[0], members[-1], f"{members[-1]} - {members[0]}")
        if diff is not None:
            ladder_changes.append(diff)
        for lo, hi in zip(members[:-1], members[1:]):
            add("A", f"{family_name} step", lo, hi, f"{hi} - {lo}")
    for family_name, members in c["secondary_ladders"]:
        add("A", f"{family_name} largest minus smallest", members[0], members[-1], f"{members[-1]} - {members[0]}")
        for lo, hi in zip(members[:-1], members[1:]):
            add("A", f"{family_name} step", lo, hi, f"{hi} - {lo}")
    for earlier, later, kind in c["training"]:
        diff = add("B", kind, earlier, later, f"{later} - {earlier}")
        if diff is not None:
            training_changes.append(diff)
    all_ladders, all_training = len(ladder_changes) == len(c["ladders"]), len(training_changes) == len(c["training"])
    ladders_ok = all_ladders and all(d[3] for d in ladder_changes)
    series_name, c_name = "mean of the ladders, largest minus smallest", "mean absolute training change minus mean absolute ladder change"
    c_models = f"{len(training_changes)} contrasts, {len(ladder_changes)} ladders"
    if ladders_ok:
        point = float(np.mean([d[0] for d in ladder_changes]))
        boot = np.mean(np.vstack([d[1] for d in ladder_changes]), axis=0)
        series_row = contrast_row("series_mean", metric, set_name, series_name, (point, boot, 0, min(d[3] for d in ladder_changes)), f"{len(ladder_changes)} ladders", total)
        rows.append(dict(series_row, p_raw=np.nan))
    elif all_ladders:
        rows.append(dict(contrast_row("series_mean", metric, set_name, series_name, UNDEFINED, f"{len(ladder_changes)} ladders", total), p_raw=np.nan,
                         reason="a ladder has no finding that both of its models define"))
    if ladders_ok and all_training and all(d[3] for d in training_changes):
        point = float(np.mean([abs(d[0]) for d in training_changes]) - np.mean([abs(d[0]) for d in ladder_changes]))
        boot = (np.mean(np.abs(np.vstack([d[1] for d in training_changes])), axis=0)
                - np.mean(np.abs(np.vstack([d[1] for d in ladder_changes])), axis=0))
        rows.append(contrast_row("C", metric, set_name, c_name, (point, boot, 0, min(d[3] for d in training_changes + ladder_changes)), c_models, total))
    elif all_ladders and all_training:
        rows.append(dict(contrast_row("C", metric, set_name, c_name, UNDEFINED, c_models, total),
                         reason="a contrast of the comparison has no finding that both of its models define"))
    return rows


def _check_ladder_order(cfg: Dict) -> None:
    if not os.path.exists(facts_csv(cfg)):
        return
    params = pd.read_csv(facts_csv(cfg), low_memory=False).set_index("model")["parameters"]
    for name, members in size_ladders(cfg["contrasts"]):
        counts = [params.get(m) for m in members]
        if all(c is not None for c in counts) and any(a >= b for a, b in zip(counts[:-1], counts[1:])):
            raise ValueError(f"[main_contrasts] the {name} ladder is not listed from smallest to largest: {dict(zip(members, counts))}")


def contrasts_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["contrasts_dir"], f"{name}.csv")


def expected_sets(cfg: Dict, full: Dict, metric: str) -> List[str]:
    return list(cfg["question_sets"]["modalities"])


def family_gaps(cfg: Dict, full: Dict, metrics: Optional[List[str]] = None) -> List[str]:
    c = cfg["contrasts"]
    models = sorted({m for _, members in size_ladders(c) for m in members} | {m for pair in c["training"] for m in pair[:2]})
    gaps = []
    for metric in metrics or c["metrics"]:
        for set_name in expected_sets(cfg, full, metric):
            for m in models:
                n = open_items(cfg, set_name, m, metric)
                if n is None or n:
                    gaps.append(f"{metric}/{set_name}/{m} ({'no main_compute_metrics cache' if n is None else f'{n} open items'})")
    return gaps


@single_shot_line("contrasts", lambda cfg: stage_files(cfg, "caches", "manifests", "facts"))
def main_contrasts(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "main_contrasts")
    _check_ladder_order(cfg)
    rows, hashes = [], {}
    for set_name in cfg["question_sets"]["modalities"]:
        if not os.path.isdir(os.path.dirname(boot_npz(cfg, set_name, "x"))):
            continue
        for metric in cfg["contrasts"]["metrics"]:
            rows += contrasts_for(cfg, set_name, metric, hashes)
    if not rows:
        raise MissingInput("[main_contrasts] no set has two measured models of a contrast; run main_run and main_compute_metrics first")
    table = pd.DataFrame(rows)
    gaps = family_gaps(cfg, full)
    if gaps:
        raise MissingInput(f"[main_contrasts] {len(gaps)} model(s) of the contrasts are not measured yet, so no FDR correction is written (first: {'; '.join(gaps[:4])})")
    table = add_fdr(table, ["family", "metric"])
    write_csv_if_changed(table, contrasts_csv(cfg, "contrasts"))
    append_status(status, f"main_contrasts: {len(table)} contrast rows")


def scaling_rows(cfg: Dict, exclude: Optional[Exclude] = None, tag: str = "main_contrasts") -> Tuple[List[Dict], List[str]]:
    from statsmodels.regression.mixed_linear_model import MixedLM
    params = pd.read_csv(facts_csv(cfg), low_memory=False).set_index("model")["parameters"]
    ladder_of = {m: name for name, members in size_ladders(cfg["contrasts"]) for m in members}
    rows, waiting = [], []
    for set_name in cfg["question_sets"]["modalities"]:
        left_out = exclude(set_name, list(ladder_of)) if exclude is not None else set()
        complete = complete_macro_auroc({**cfg, "models": list(ladder_of)}, set_name)
        data, measured = [], 0
        for m, family in ladder_of.items():
            path = boot_npz(cfg, set_name, m)
            if m not in complete or m not in params.index:
                continue
            measured += 1
            z = load_boot(path)
            for metric, scope, point in zip(z["metrics"], z["scopes"], z["point"]):
                if metric == "auroc" and scope != "macro" and np.isfinite(point) and str(scope) not in left_out:
                    data.append({"model": m, "family": family, "finding": str(scope), "auroc": float(point), "log2_params": float(np.log2(params[m]))})
        d = pd.DataFrame(data)
        if measured == len(ladder_of) and d.empty and left_out:
            rows.append({"set": set_name, "measure": "mixedlm_slope_log2_params", **stat_row(np.nan, np.nan, 0), "n_models": 0, "converged": False,
                         "optimizer": "", "reason": "every finding is marked for a ladder model"})
            continue
        if d.empty or d["model"].nunique() < len(ladder_of):
            waiting.append(set_name)
            continue
        model = MixedLM.from_formula("auroc ~ log2_params + C(family) + C(finding)", d, groups=d["model"])
        fit, optimizer = model.fit(reml=True), "bfgs, lbfgs, cg"
        for alt in ("powell", "nm"):
            if fit.converged:
                break
            fit, optimizer = model.fit(reml=True, method=alt), alt
        if not fit.converged:
            raise RuntimeError(f"[{tag}] {set_name}: mixed model did not converge under any optimizer")
        rows.append({"set": set_name, "measure": "mixedlm_slope_log2_params", **stat_row(fit.params["log2_params"], fit.pvalues["log2_params"], len(d)),
                     "n_models": int(d["model"].nunique()), "converged": bool(fit.converged), "optimizer": optimizer, "reason": ""})
    return rows, waiting


@single_shot_line("scaling_mixedlm", lambda cfg: stage_files(cfg, "caches", "facts"))
def main_scaling_mixedlm(cfg_path: str) -> None:
    require_packages(["statsmodels.regression.mixed_linear_model"], "main_contrasts mixed model")
    cfg = read_config(cfg_path)[PROJECT]
    if not os.path.exists(facts_csv(cfg)):
        raise MissingInput("[main_contrasts] model_facts.csv is missing; run main_model_facts")
    rows, waiting = scaling_rows(cfg)
    sets = list(cfg["question_sets"]["modalities"])
    if waiting:
        raise MissingInput(f"[main_contrasts] the mixed model ran in {len(sets) - len(waiting)} of {len(sets)} sets; the FDR family needs every set")
    write_csv_if_changed(add_fdr(pd.DataFrame(rows)), contrasts_csv(cfg, "mixedlm"))
