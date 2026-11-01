"""
analysis/contamination.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from functools import partial
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from analysis.benchmarks import benchmark_points
from analysis.compute_metrics import boot_npz, question_frame
from analysis.contrasts import contrasts_for, family_gaps, finding_values, macro_difference, open_items, scaling_rows
from analysis.joint_bootstrap import summarize
from analysis.selection import SELECTION_VERSION, fixed_rules, panel_scores, run_draws, summarize_draws
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, run_build_unit, run_build_units, single_shot_line, stage_files
from data_loader.build_utils import read_csv_defensively, write_csv_if_changed
from data_loader.sets import modality_sets, set_config
from Inference.inference_runner import SetData
from Inference.model_facts import facts_csv
from Inference.report_utils import add_fdr, metric_row
from Inference.resume_utils import MissingInput, append_status, digest_file, status_path, write_csv_atomic
from Inference.serving_utils import require_packages
from Inference.stats_utils import shift_reflect_p


CONTAMINATION_VERSION = 2
TRAINING_COLUMNS = ["model", "source", "quote", "datasets", "modalities"]


def _split(value) -> List[str]:
    return [v.strip() for v in str(value).split(";") if v.strip()] if isinstance(value, str) else []


def training_table(cfg: Dict) -> pd.DataFrame:
    path = cfg["contamination"]["training_data_csv"]
    if not os.path.exists(path):
        raise MissingInput(f"[main_contamination] {path} does not exist; build it from the model cards and reports")
    t = read_csv_defensively(path, dtype=str, keep_default_na=False)
    missing = [c for c in TRAINING_COLUMNS if c not in t.columns]
    if missing:
        raise ValueError(f"[main_contamination] the training-data table lacks {missing}")
    lacking = [m for m in cfg["models"] if m not in set(t["model"])]
    unknown = sorted(set(t["model"]) - set(cfg["models"]))
    if lacking or unknown:
        raise ValueError(f"[main_contamination] the training-data table lacks {lacking} and has models outside the panel {unknown}")
    if (t["source"].str.strip() == "").any() or (t["quote"].str.strip() == "").any():
        raise ValueError("[main_contamination] a row of the training-data table has no source or no quote")
    bad_d = sorted({d for v in t["datasets"] for d in _split(v)} - set(cfg["datasets"]) - set(cfg["benchmarks"]["sources"]))
    bad_m = sorted({m for v in t["modalities"] for m in _split(v)} - set(modality_sets(cfg)))
    if bad_d or bad_m:
        raise ValueError(f"[main_contamination] the training-data table names datasets {bad_d} or modalities {bad_m} that are not this study's")
    return t


def marked_pairs(t: pd.DataFrame) -> Set[Tuple[str, str]]:
    return {(m, d) for m, v in zip(t["model"], t["datasets"]) for d in _split(v)}


def marked_findings(cfg: Dict, marked: Set[Tuple[str, str]], set_name: str, models: List[str]) -> Set[str]:
    return {f for f, fc in set_config(cfg, set_name)["findings"].items() if any((m, fc["dataset"]) in marked for m in models)}


def contrasts_unmarked(cfg: Dict, full: Dict, marked: Set[Tuple[str, str]], hashes: Dict[str, str]) -> Tuple[List[Dict], List[str]]:
    exclude = partial(marked_findings, cfg, marked)
    rows = []
    for set_name in modality_sets(cfg):
        if not os.path.isdir(os.path.dirname(boot_npz(cfg, set_name, "x"))):
            continue
        for metric in cfg["contamination"]["metrics"]:
            rows += contrasts_for(cfg, set_name, metric, hashes, exclude=exclude)
    return rows, family_gaps(cfg, full, cfg["contamination"]["metrics"])


def named_modality_rows(cfg: Dict, t: pd.DataFrame, hashes: Dict[str, str]) -> Tuple[List[Dict], List[str]]:
    rows, gaps = [], []
    for general, medical, kind in cfg["contrasts"]["training"]:
        if kind != cfg["contamination"]["medical_kind"]:
            continue
        named = {m for v in t.loc[t["model"] == medical, "modalities"] for m in _split(v)}
        changes = {}
        for set_name in modality_sets(cfg):
            a, b = finding_values(cfg, set_name, general, "auroc", hashes), finding_values(cfg, set_name, medical, "auroc", hashes)
            if a is None or b is None or open_items(cfg, set_name, general, "auroc") or open_items(cfg, set_name, medical, "auroc"):
                gaps.append(f"named modalities {medical} - {general}: {set_name}")
                continue
            changes[set_name] = macro_difference(a, b)
        inside, outside = [s for s in changes if s in named], [s for s in changes if s not in named]
        if len(changes) < len(modality_sets(cfg)):
            continue
        row = {"contrast": f"{medical} - {general}", "named_modalities": ";".join(sorted(inside)), "other_modalities": ";".join(sorted(outside))}
        if not inside or not outside:
            rows.append(dict(row, reason="every modality is named" if not outside else "no modality is named", p_raw=np.nan))
            continue
        point = float(np.mean([changes[s][0] for s in inside]) - np.mean([changes[s][0] for s in outside]))
        boot = np.mean(np.vstack([changes[s][1] for s in inside]), axis=0) - np.mean(np.vstack([changes[s][1] for s in outside]), axis=0)
        s = summarize(boot)
        row.update(metric_row(s["mean"], s["sd"], s["lo"], s["hi"], int(sum(changes[x][2] for x in changes)), "percentage_points", f"named/{medical}"))
        row.update({"point_raw": point, "p_raw": shift_reflect_p(boot, point), "n_replicates": s["n_finite"], "reason": ""})
        rows.append(row)
    return rows, gaps


def contamination_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["contamination_dir"], f"{name}.csv")


def selection_unmarked(cfg: Dict, full: Dict, marked: Set[Tuple[str, str]]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str]]:
    metrics, stats, gaps, attempts, ready = [], [], [], [], []
    bench = benchmark_points(cfg, full)
    for set_name in modality_sets(cfg):
        try:
            q = question_frame(SetData(full, set_name))
        except MissingInput as e:
            gaps.append(f"selection {set_name} ({e})")
            continue
        models, P, missing = panel_scores(cfg, set_name, q)
        if missing:
            gaps.append(f"selection {set_name} ({len(models)} of {len(cfg['models'])} models measured)")
            continue
        keep = ~q["finding"].isin(marked_findings(cfg, marked, set_name, models)).to_numpy()
        if not keep.any():
            continue
        q2, P2 = q[keep].reset_index(drop=True), P[:, keep]
        path = contamination_csv(cfg, f"draws_unmarked_{set_name}")
        rules = fixed_rules(cfg, models, set_name, bench)
        params = {"version": CONTAMINATION_VERSION, "selection_version": SELECTION_VERSION, "selection": cfg["selection"], "models": models, "kept": sorted(set(q2["finding"])),
                  "caches": {m: digest_file(boot_npz(cfg, set_name, m)) for m in models}, "rules": rules}

        def build(q2=q2, P2=P2, path=path, models=models, rules=rules):
            write_csv_atomic(run_draws(cfg, q2, models, P2, rules), path)
        attempts.append((f"main_contamination__{set_name}", lambda wait, set_name=set_name, path=path, params=params, build=build:
                         run_build_unit(cfg, f"main_contamination__{set_name}", path, params, lambda: True, build, wait=wait)))
        ready.append((set_name, path))
    run_build_units(attempts, "[main_contamination] sets")
    for set_name, path in ready:
        m, s = summarize_draws(cfg, set_name, pd.read_csv(path, low_memory=False))
        metrics.append(m)
        stats.append(s)
    return metrics, stats, gaps


@single_shot_line("contamination", lambda cfg: stage_files(cfg, "caches", "manifests", "facts", "training", "records"))
def main_contamination(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "main_contamination")
    t = training_table(cfg)
    marked = marked_pairs(t)
    write_csv_if_changed(pd.DataFrame(sorted(marked), columns=["model", "dataset"]), contamination_csv(cfg, "marked_pairs"))
    hashes: Dict[str, str] = {}
    rows, gaps = contrasts_unmarked(cfg, full, marked, hashes)
    if rows and not gaps:
        write_csv_if_changed(add_fdr(pd.DataFrame(rows), ["family", "metric"]), contamination_csv(cfg, "contrasts_unmarked"))
    named, named_gaps = named_modality_rows(cfg, t, hashes)
    gaps += named_gaps
    if not named_gaps and named:
        write_csv_if_changed(add_fdr(pd.DataFrame(named)), contamination_csv(cfg, "named_modalities"))
    metrics, stats, e4_gaps = selection_unmarked(cfg, full, marked)
    gaps += e4_gaps
    if metrics:
        m, s = pd.concat(metrics, ignore_index=True), pd.concat(stats, ignore_index=True)
        write_csv_if_changed(m, contamination_csv(cfg, "selection_regret_unmarked"))
        write_csv_if_changed(s, contamination_csv(cfg, "selection_tau_unmarked"))
    append_status(status, f"main_contamination: {len(marked)} marked pairs, {len(rows)} contrast rows, {len(named)} named-modality rows, {len(metrics)} selection sets")
    if gaps:
        raise MissingInput(f"[main_contamination] {len(gaps)} input(s) still wait (first: {'; '.join(gaps[:4])})")


@single_shot_line("scaling_mixedlm_unmarked", lambda cfg: stage_files(cfg, "caches", "facts", "training"))
def main_scaling_mixedlm_unmarked(cfg_path: str) -> None:
    require_packages(["statsmodels.regression.mixed_linear_model"], "main_contamination mixed model")
    cfg = read_config(cfg_path)[PROJECT]
    if not os.path.exists(facts_csv(cfg)):
        raise MissingInput("[main_contamination] model_facts.csv is missing; run main_model_facts")
    rows, waiting = scaling_rows(cfg, partial(marked_findings, cfg, marked_pairs(training_table(cfg))), "main_contamination")
    sets = modality_sets(cfg)
    if waiting:
        raise MissingInput(f"[main_contamination] the mixed model ran in {len(sets) - len(waiting)} of {len(sets)} sets; the FDR family needs every set")
    write_csv_if_changed(add_fdr(pd.DataFrame(rows)), contamination_csv(cfg, "mixedlm_unmarked"))
