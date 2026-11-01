"""
data_loader/sources/fundus.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, mask_is_empty, read_csv_defensively, rel_to_root, tree_signature


DATASET = "ddr"
_EMPTY: Dict[str, bool] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"grading": {s: file_signature(os.path.join(dcfg["grading_dir"], f"{s}.txt")) for s in dcfg["splits"]},
            "lesion_dir": tree_signature(dcfg["lesion_dir"]), "image_dir": tree_signature(dcfg["image_dir"])}


def _grades(dcfg: Dict) -> pd.DataFrame:
    parts = []
    for s in dcfg["splits"]:
        d = read_csv_defensively(os.path.join(dcfg["grading_dir"], f"{s}.txt"), sep=" ", header=None, names=["file", "grade"], dtype={"file": str})
        parts.append(pd.DataFrame({"key": d["file"].str.rsplit(".", n=1).str[0], "grade": d["grade"].astype(int), "grade_split": s}))
    g = pd.concat(parts, ignore_index=True)
    if g["key"].duplicated().any():
        raise ValueError(f"[{DATASET}] {int(g['key'].duplicated().sum())} images appear in more than one grading list")
    return g


def _lesion_files(dcfg: Dict) -> pd.DataFrame:
    rows = []
    for s, label_dir in dcfg["lesion_label_dirs"]:
        for t in dcfg["lesion_types"]:
            d = os.path.join(dcfg["lesion_dir"], s, label_dir, t)
            rows.extend((os.path.splitext(f)[0], s, t, os.path.join(d, f)) for f in sorted(os.listdir(d)))
    les = pd.DataFrame(rows, columns=["key", "lesion_split", "lesion", "mask_path"])
    per_image = les.groupby("key")["lesion"].nunique()
    if (per_image != len(dcfg["lesion_types"])).any():
        raise ValueError(f"[{DATASET}] {int((per_image != len(dcfg['lesion_types'])).sum())} lesion images lack a mask of some lesion type")
    return les


def _empty_flags(paths: List[str]) -> np.ndarray:
    from tqdm import tqdm
    todo = [p for p in paths if p not in _EMPTY]
    for p in tqdm(todo, desc=f"[{DATASET}] lesion masks", unit="mask", disable=not todo):
        _EMPTY[p] = mask_is_empty(p)
    return np.array([_EMPTY[p] for p in paths], dtype=bool)


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    g = _grades(dcfg)
    les = _lesion_files(dcfg).drop_duplicates("key")[["key", "lesion_split"]]
    t = g.merge(les, on="key", how="outer")
    t["split"] = t["lesion_split"].fillna(t["grade_split"])
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({
        "case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
        "split": t["split"], "partition": np.where(t["split"].isin(dcfg["eval_splits"]), "eval", "dev"),
        "image_path": [os.path.join(base, f"{k}.png") for k in t["key"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    les = _lesion_files(dcfg)
    in_lesion_set = ev["key"].isin(set(les["key"]))
    grades = _grades(dcfg).set_index("key")["grade"]
    other = ev[~in_lesion_set]
    other_grade = other["key"].map(grades)
    if fcfg["kind"] == "grade":
        return other[other_grade.isin(fcfg["positive_grades"])].copy(), other[other_grade.isin(fcfg["negative_grades"])].copy()
    if fcfg["kind"] != "lesion":
        raise ValueError(f"[{DATASET}] unknown finding kind {fcfg['kind']!r}")
    paths = les[les["lesion"] == fcfg["source_label"]].set_index("key")["mask_path"]
    lesion_ev = ev[in_lesion_set]
    empty = _empty_flags(lesion_ev["key"].map(paths).tolist())
    neg = pd.concat([lesion_ev[empty], other[other_grade.isin(fcfg["negative_grades"])]])
    return lesion_ev[~empty].copy(), neg.copy()
