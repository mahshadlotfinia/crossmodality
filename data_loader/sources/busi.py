"""
data_loader/sources/busi.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, rel_to_root
from data_loader.slices import read_index


DATASET = "busi"


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    classes = [str(c) for c in dcfg["classes"]]
    folder = [next((c for c in classes if f"{c}/" in src.replace(os.sep, "/")), None) for src in idx["source"]]
    if any(f is None for f in folder):
        raise ValueError(f"[{DATASET}] an image lies outside the class folders {classes}")
    return pd.DataFrame({"key": [os.path.splitext(f)[0] for f in idx["image"]], "image": idx["image"],
                         "mask": idx["mask"] if "mask" in idx.columns else "", "tissue_class": folder})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["image"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cls = ev["key"].map(_images(dcfg).set_index("key")["tissue_class"])
    negative = fcfg.get("negative_classes", [dcfg["normal_class"]])
    return ev[cls.isin(fcfg["positive_classes"])].copy(), ev[cls.isin(negative)].copy()
