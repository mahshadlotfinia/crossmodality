"""
data_loader/sources/breast.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, rel_to_root
from data_loader.slices import read_index


DATASET = "breast"


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "clinical_xlsx": file_signature(dcfg["clinical_xlsx"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    clin = pd.read_excel(dcfg["clinical_xlsx"], dtype=str)
    missing = [c for c in (dcfg["image_column"], dcfg["class_column"]) if c not in clin.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the clinical file lacks {missing}; its columns are {list(clin.columns)}")
    idx = read_index(dcfg)
    masks = idx["mask"] if "mask" in idx.columns else pd.Series([""] * len(idx))
    stored = {os.path.splitext(os.path.basename(f))[0]: (f, m) for f, m in zip(idx["image"], masks)}
    names = clin[dcfg["image_column"]].str.strip().str.replace(r"\.png$", "", regex=True)
    lacking = sorted(set(names) - set(stored))
    if lacking:
        raise ValueError(f"[{DATASET}] {len(lacking)} images of the clinical file were not preprocessed, first {lacking[0]}")
    return pd.DataFrame({"key": names, "tissue_class": clin[dcfg["class_column"]].str.strip().str.lower(),
                         "image": [stored[n][0] for n in names], "mask": [stored[n][1] for n in names]})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["image"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cls = ev["key"].map(_images(dcfg).set_index("key")["tissue_class"])
    return ev[cls.isin(fcfg["positive_classes"])].copy(), ev[cls.isin(fcfg["negative_classes"])].copy()
