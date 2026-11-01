"""
data_loader/sources/cxr.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root, tree_signature


DATASET = "vindr_cxr"


def source_signature(dcfg: Dict) -> Dict:
    return {"labels_csv": file_signature(dcfg["labels_csv"]), "boxes_csv": file_signature(dcfg["boxes_csv"]),
            "dicom_meta_csvs": [file_signature(p) for p in dcfg["dicom_meta_csvs"]], "image_dir": tree_signature(dcfg["image_dir"])}


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    lab = read_csv_defensively(dcfg["labels_csv"], usecols=["image_id", "split"], dtype={"image_id": str, "split": str})
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({
        "case_id": [make_case_id(DATASET, i) for i in lab["image_id"]], "dataset": DATASET, "key": lab["image_id"],
        "group_id": lab["image_id"], "split": lab["split"],
        "partition": np.where(lab["split"].isin(dcfg["eval_splits"]), "eval", "dev"),
        "image_path": [os.path.join(base, s, f"{i}.jpg") for i, s in zip(lab["image_id"], lab["split"])]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = fcfg["source_label"]
    lab = read_csv_defensively(dcfg["labels_csv"], usecols=["image_id", col], dtype={"image_id": str})
    values = lab.set_index("image_id")[col]
    odd = set(values.dropna().unique()) - {0, 1}
    if odd:
        raise ValueError(f"[{DATASET}] column {col} holds values other than 0 and 1: {sorted(odd)}")
    label = ev["key"].map(values)
    if label.isna().any():
        raise ValueError(f"[{DATASET}] {int(label.isna().sum())} evaluation images have no value for {col}")
    return ev[label == 1].copy(), ev[label == 0].copy()


def dev_finding_pools(fcfg: Dict, dev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos, neg = finding_pools(fcfg, dev, dcfg)
    ann = read_csv_defensively(dcfg["boxes_csv"], usecols=["image_id", "class_name"], dtype={"image_id": str})
    drawn = set(ann.loc[ann["class_name"] == fcfg["source_label"], "image_id"])
    return pos, neg[~neg["key"].isin(drawn)].copy()
