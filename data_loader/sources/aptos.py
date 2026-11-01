"""
data_loader/sources/aptos.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "aptos"


def source_signature(dcfg: Dict) -> Dict:
    return {"labels_csv": file_signature(dcfg["labels_csv"])}


def _labels(dcfg: Dict) -> pd.DataFrame:
    return read_csv_defensively(dcfg["labels_csv"], usecols=["id_code", "diagnosis"], dtype={"id_code": str})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _labels(dcfg)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, i) for i in t["id_code"]], "dataset": DATASET, "key": t["id_code"],
                         "group_id": t["id_code"], "split": "none", "partition": "eval", "image_path": [os.path.join(base, f"{i}.png") for i in t["id_code"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grade = ev["key"].map(_labels(dcfg).set_index("id_code")["diagnosis"])
    if grade.isna().any():
        raise ValueError(f"[{DATASET}] {int(grade.isna().sum())} images have no grade")
    return ev[grade.isin(fcfg["positive_grades"])].copy(), ev[grade.isin(fcfg["negative_grades"])].copy()
