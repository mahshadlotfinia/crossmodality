"""
data_loader/sources/idrid.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "idrid"


def source_signature(dcfg: Dict) -> Dict:
    return {"label_csvs": [file_signature(p) for _, p in dcfg["label_csvs"]]}


def _images(dcfg: Dict) -> pd.DataFrame:
    dirs = {str(s): d for s, d in dcfg["image_dirs"]}
    parts = []
    for split, path in dcfg["label_csvs"]:
        t = read_csv_defensively(path, dtype=str)
        t.columns = [c.strip() for c in t.columns]
        t = t.dropna(subset=[dcfg["name_column"]])
        parts.append(pd.DataFrame({"split": str(split), "name": t[dcfg["name_column"]].str.strip(), "grade": t[dcfg["grade_column"]].astype(int)}))
    out = pd.concat(parts, ignore_index=True)
    out["key"] = out["split"] + "/" + out["name"]
    out["file"] = [os.path.join(dirs[s], f"{n}.jpg") for s, n in zip(out["split"], out["name"])]
    return out


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": t["split"], "partition": "eval", "image_path": [rel_to_root(f, root) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grade = ev["key"].map(_images(dcfg).set_index("key")["grade"])
    return ev[grade.isin(fcfg["positive_grades"])].copy(), ev[grade.isin(fcfg["negative_grades"])].copy()
