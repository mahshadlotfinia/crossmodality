"""
data_loader/sources/mammo.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import ast
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root, tree_signature


DATASET = "vindr_mammo"


def source_signature(dcfg: Dict) -> Dict:
    return {"findings_csv": file_signature(dcfg["findings_csv"]), "breast_csv": file_signature(dcfg["breast_csv"]),
            "image_dir": tree_signature(dcfg["image_dir"])}


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    b = read_csv_defensively(dcfg["breast_csv"], usecols=["study_id", "image_id", "split"], dtype=str)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({
        "case_id": [make_case_id(DATASET, i) for i in b["image_id"]], "dataset": DATASET, "key": b["image_id"],
        "group_id": b["study_id"], "split": b["split"],
        "partition": np.where(b["split"].isin(dcfg["eval_splits"]), "eval", "dev"),
        "image_path": [os.path.join(base, s, f"{i}.png") for s, i in zip(b["study_id"], b["image_id"])]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fa = read_csv_defensively(dcfg["findings_csv"], usecols=["image_id", "finding_categories"], dtype=str)
    cats = fa["finding_categories"].apply(ast.literal_eval)
    positive = set(fa.loc[cats.apply(lambda c: fcfg["source_label"] in c), "image_id"])
    annotated = set(fa.loc[cats.apply(lambda c: list(c) != [dcfg["no_finding_label"]]), "image_id"])
    birads = read_csv_defensively(dcfg["breast_csv"], usecols=["image_id", "breast_birads"], dtype=str).set_index("image_id")["breast_birads"]
    b = ev["key"].map(birads)
    if b.isna().any():
        raise ValueError(f"[{DATASET}] {int(b.isna().sum())} evaluation images have no breast BI-RADS")
    is_pos = ev["key"].isin(positive)
    is_neg = ~is_pos & (b.isin(dcfg["negative_birads"]) | (b.isin(dcfg["other_finding_birads"]) & ev["key"].isin(annotated)))
    return ev[is_pos].copy(), ev[is_neg].copy()
