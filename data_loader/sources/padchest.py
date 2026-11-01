"""
data_loader/sources/padchest.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "padchest"
_CACHE: Dict[Tuple[str, Tuple[str, ...]], pd.DataFrame] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"labels_csv": file_signature(dcfg["labels_csv"])}


def _read(dcfg: Dict, columns: List[str]) -> pd.DataFrame:
    key = (dcfg["labels_csv"], tuple(columns))
    if key not in _CACHE:
        t = read_csv_defensively(dcfg["labels_csv"], usecols=sorted(set(columns) | {"view"}),
                                 dtype={"ImageID": str, "ImageDir": str, "PatientID": str, "split": str, "view": str})
        _CACHE[key] = t[t["view"].isin(dcfg["views"])].reset_index(drop=True)
    return _CACHE[key]


def _folder(value: str) -> str:
    return str(int(float(value)))


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _read(dcfg, ["ImageID", "ImageDir", "PatientID", "split"])
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, i) for i in t["ImageID"]], "dataset": DATASET, "key": t["ImageID"],
                         "group_id": t["PatientID"], "split": t["split"], "partition": np.where(t["split"].isin(dcfg["eval_splits"]), "eval", "dev"),
                         "image_path": [os.path.join(base, _folder(d), i) for d, i in zip(t["ImageDir"], t["ImageID"])]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = fcfg["source_label"]
    values = _read(dcfg, ["ImageID", col]).drop_duplicates("ImageID").set_index("ImageID")[col]
    odd = set(values.dropna().unique()) - {0, 1}
    if odd:
        raise ValueError(f"[{DATASET}] column {col} holds values other than 0 and 1: {sorted(odd)}")
    label = ev["key"].map(values)
    if label.isna().any():
        raise ValueError(f"[{DATASET}] {int(label.isna().sum())} evaluation images have no value for {col}")
    return ev[label == 1].copy(), ev[label == 0].copy()
