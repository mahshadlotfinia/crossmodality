"""
data_loader/sources/taix.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "taix"
_CACHE: Dict[Tuple[str, Tuple[str, ...]], pd.DataFrame] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"labels_csv": file_signature(dcfg["labels_csv"])}


def _read(dcfg: Dict, columns: List[str]) -> pd.DataFrame:
    key = (dcfg["labels_csv"], tuple(columns))
    if key not in _CACHE:
        _CACHE[key] = read_csv_defensively(dcfg["labels_csv"], usecols=columns, dtype={"image_id": str, "img_rel_path": str, "subject_id": str, "split": str})
    return _CACHE[key]


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _read(dcfg, ["image_id", "img_rel_path", "subject_id", "split"])
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, i) for i in t["image_id"]], "dataset": DATASET, "key": t["image_id"],
                         "group_id": t["subject_id"], "split": t["split"], "partition": np.where(t["split"].isin(dcfg["eval_splits"]), "eval", "dev"),
                         "image_path": [os.path.join(base, p) for p in t["img_rel_path"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [fcfg["positive_column"]] + list(fcfg["negative_columns"])
    lab = _read(dcfg, ["image_id"] + cols).set_index("image_id")
    rows = lab.loc[ev["key"]]
    if rows[cols].isna().any().any():
        raise ValueError(f"[{DATASET}] {int(rows[cols].isna().any(axis=1).sum())} evaluation images lack a value in {cols}")
    pos = (rows[fcfg["positive_column"]] == 1).to_numpy()
    neg = (rows[list(fcfg["negative_columns"])] == 0).all(axis=1).to_numpy()
    return ev[pos].copy(), ev[neg].copy()
