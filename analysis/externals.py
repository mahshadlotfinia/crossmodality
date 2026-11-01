"""
analysis/externals.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.compute_metrics import complete_macro_auroc
from analysis.contrasts import contrasts_for, open_items, size_ladders
from config.serde import read_config
from data_loader.build_question_sets import PROJECT, manifest_csv, single_shot_line, stage_files
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import external_sets, modality_of
from Inference.report_utils import add_fdr, stat_row
from Inference.resume_utils import MissingInput, append_status, status_path
from Inference.stats_utils import spearman


STRATUM = "auroc_stratum"


def _contrast_models(cfg: Dict) -> List[str]:
    c = cfg["contrasts"]
    return sorted({m for _, members in size_ladders(c) for m in members} | {m for pair in c["training"] for m in pair[:2]})


def _family_complete(cfg: Dict, set_name: str, metric: str) -> List[str]:
    gaps = []
    for m in _contrast_models(cfg):
        n = open_items(cfg, set_name, m, "auroc" if metric == STRATUM else metric)
        if n is None or n:
            gaps.append(f"{set_name}/{m} ({'no main_compute_metrics cache' if n is None else f'{n} open items'})")
    return gaps


def _macro(cfg: Dict, set_name: str) -> Dict[str, float]:
    return complete_macro_auroc(cfg, set_name)


def externals_csv(cfg: Dict, name: str) -> str:
    return os.path.join(cfg["outputs"]["externals_dir"], f"{name}.csv")


@single_shot_line("externals", lambda cfg: stage_files(cfg, "caches", "manifests"))
def main_externals(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "externals")
    rows, corr, gaps, hashes = [], [], [], {}
    for set_name in external_sets(cfg):
        if not os.path.exists(manifest_csv(cfg, set_name)):
            gaps.append(f"{set_name} (no manifest; run main_build_external_sets)")
            continue
        family = _family_complete(cfg, set_name, "auroc")
        if family:
            gaps += family
        else:
            r = contrasts_for(cfg, set_name, "auroc", hashes)
            rows += [dict(x, family_set=set_name) for x in r if x["family"] in ("B", "C")]
        internal, external = _macro(cfg, modality_of(cfg, set_name)), _macro(cfg, set_name)
        models = [m for m in cfg["models"] if m in internal and m in external]
        if len(models) < len(cfg["models"]):
            gaps.append(f"{set_name}: internal and external macro AUROC of {len(models)} of {len(cfg['models'])} models")
        else:
            rho, p, n = spearman(np.array([internal[m] for m in models]), np.array([external[m] for m in models]))
            corr.append({"set": set_name, "modality": modality_of(cfg, set_name), "measure": "spearman_internal_vs_external_macro_auroc", **stat_row(rho, p, n)})
    if rows:
        write_csv_if_changed(add_fdr(pd.DataFrame(rows), ["family", "metric", "family_set"]), externals_csv(cfg, "external_contrasts"))
    if corr:
        table = pd.DataFrame(corr)
        write_csv_if_changed(add_fdr(table) if len(table) == len(external_sets(cfg)) else table.assign(p_fdr=np.nan, significant_fdr05=False),
                             externals_csv(cfg, "external_rank_agreement"))
    append_status(status, f"externals: {len(rows)} contrast rows, {len(corr)} rank agreements")
    if gaps:
        raise MissingInput(f"[externals] {len(gaps)} external set(s) or model(s) still wait (first: {'; '.join(gaps[:4])})")
