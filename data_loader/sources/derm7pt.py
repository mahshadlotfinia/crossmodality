"""
data_loader/sources/derm7pt.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "derm7pt"


def source_signature(dcfg: Dict) -> Dict:
    return {"meta_csv": file_signature(dcfg["meta_csv"]), "index_csv": file_signature(dcfg["index_csv"])}


def _cases(dcfg: Dict) -> pd.DataFrame:
    t = read_csv_defensively(dcfg["meta_csv"], dtype=str)
    missing = [c for c in ["case_num", "derm", "diagnosis"] + list(dcfg["attributes"]) if c not in t.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the meta file lacks {missing}")
    t["file"] = [os.path.join(dcfg["images_out"], os.path.splitext(p)[0] + ".png") for p in t["derm"]]
    lacking = [f for f in t["file"] if not os.path.exists(os.path.join(dcfg["dir"], f))]
    if lacking:
        raise ValueError(f"[{DATASET}] {len(lacking)} dermoscopic images were not preprocessed, first {lacking[0]}")
    return t


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _cases(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["case_num"]], "dataset": DATASET, "key": t["case_num"], "group_id": t["case_num"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _cases(dcfg).set_index("case_num")
    if fcfg["kind"] == "diagnosis":
        dx = ev["key"].map(t["diagnosis"]).str.strip().str.lower()
        hit = dx.str.startswith(fcfg["positive_prefix"])
        return ev[hit].copy(), ev[~hit].copy()
    value = ev["key"].map(t[fcfg["source_label"]]).str.strip().str.lower()
    absent = value == dcfg["absent_value"]
    return ev[~absent & value.notna()].copy(), ev[absent].copy()
