"""
data_loader/sources/padufes.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root
from data_loader.slices import read_index


DATASET = "padufes"


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "metadata_csv": file_signature(dcfg["metadata_csv"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    meta = read_csv_defensively(dcfg["metadata_csv"], dtype=str)
    missing = [c for c in ("patient_id", "img_id", "diagnostic") if c not in meta.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the metadata file lacks {missing}")
    idx = read_index(dcfg)
    stored = {os.path.splitext(os.path.basename(f))[0]: f for f in idx["image"]}
    names = meta["img_id"].str.replace(r"\.png$", "", regex=True)
    lacking = sorted(set(names) - set(stored))
    if lacking:
        raise ValueError(f"[{DATASET}] {len(lacking)} images of the metadata were not preprocessed, first {lacking[0]}")
    return pd.DataFrame({"key": names, "patient": meta["patient_id"], "diagnostic": meta["diagnostic"].str.strip(), "file": names.map(stored)})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["patient"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dx = ev["key"].map(_images(dcfg).set_index("key")["diagnostic"])
    return ev[dx.isin(fcfg["positive_classes"])].copy(), ev[dx.isin(fcfg["negative_classes"])].copy()
