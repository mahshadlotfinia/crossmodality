"""
data_loader/sources/messidor2.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root, tree_signature


DATASET = "messidor2"


def source_signature(dcfg: Dict) -> Dict:
    return {"labels_csv": file_signature(dcfg["labels_csv"]), "pairs_csv": file_signature(dcfg["pairs_csv"]), "image_dir": tree_signature(dcfg["image_dir"])}


def _stem(name: str) -> str:
    return os.path.splitext(str(name).strip())[0].lower()


def _images(dcfg: Dict) -> pd.DataFrame:
    lab = read_csv_defensively(dcfg["labels_csv"], dtype={"image_id": str})
    lab = lab[lab["adjudicated_gradable"] == 1]
    pairs = read_csv_defensively(dcfg["pairs_csv"], sep=";", dtype=str)
    pairs.columns = [c.strip() for c in pairs.columns]
    patient = {}
    for i, (left, right) in enumerate(zip(pairs["left"], pairs["right"])):
        patient[_stem(left)] = patient[_stem(right)] = f"pair{i:04d}"
    files = {_stem(f): f for f in os.listdir(dcfg["image_dir"])}
    stems = lab["image_id"].map(_stem)
    lacking = sorted(set(stems) - set(patient)) + sorted(set(stems) - set(files))
    if lacking:
        raise ValueError(f"[{DATASET}] {len(lacking)} gradable images lack a pair or a file, first {lacking[0]}")
    return pd.DataFrame({"key": stems, "grade": lab["adjudicated_dr_grade"].astype(int).to_numpy(), "patient": stems.map(patient),
                         "file": stems.map(files)})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["patient"],
                         "split": "none", "partition": "eval", "image_path": [os.path.join(base, f) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grade = ev["key"].map(_images(dcfg).set_index("key")["grade"])
    return ev[grade.isin(fcfg["positive_grades"])].copy(), ev[grade.isin(fcfg["negative_grades"])].copy()
