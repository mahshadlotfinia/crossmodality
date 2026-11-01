"""
data_loader/sources/chestxdet.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, rel_to_root, tree_signature


DATASET = "chestxdet"
_CACHE: Dict[str, pd.DataFrame] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"annotations": [file_signature(p) for _, p in dcfg["annotations"]], "image_dir": tree_signature(dcfg["image_dir"])}


def _records(dcfg: Dict) -> pd.DataFrame:
    key = dcfg["image_dir"]
    if key not in _CACHE:
        folders = {str(s): str(f) for s, f in dcfg["image_subdirs"]}
        rows = []
        for split, path in dcfg["annotations"]:
            with open(path) as f:
                for rec in json.load(f):
                    if len(rec["syms"]) != len(rec["boxes"]):
                        raise ValueError(f"[{DATASET}] {rec['file_name']}: {len(rec['syms'])} findings and {len(rec['boxes'])} boxes")
                    rows.append({"split": str(split), "file": rec["file_name"], "syms": list(rec["syms"]), "boxes": list(rec["boxes"]),
                                 "image": os.path.join(dcfg["image_dir"], folders[str(split)], rec["file_name"])})
        t = pd.DataFrame(rows)
        t["key"] = t["split"] + "/" + t["file"].str.replace(r"\.png$", "", regex=True)
        _CACHE[key] = t
    return _CACHE[key]


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _records(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": t["split"], "partition": "eval", "image_path": [rel_to_root(p, root) for p in t["image"]]})


def _has(syms: List[str], names: List[str]) -> bool:
    return any(s in names for s in syms)


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    syms = _records(dcfg).set_index("key")["syms"]
    labels, excluded = list(fcfg["source_labels"]), list(fcfg.get("exclude_labels", []))
    s = ev["key"].map(syms)
    pos = np.array([_has(x, labels) for x in s])
    neg = np.array([not _has(x, labels + excluded) for x in s])
    return ev[pos].copy(), ev[neg].copy()
