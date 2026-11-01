"""
data_loader/sources/bhsd.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, rel_to_root
from data_loader.slices import decile, extent, read_index, read_volume, slice_frame, slice_rows, spaced_largest
from Inference.resume_utils import MissingInput


DATASET = "bhsd"
_STATS: Dict[str, Dict] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "label_values": dcfg["label_values"]}


def _label_values(dcfg: Dict) -> Dict[str, int]:
    if not dcfg["label_values"]:
        raise MissingInput("the label value of each hemorrhage type is not set; read it from the BHSD release and set datasets.bhsd.label_values")
    return {str(name): int(v) for name, v in dcfg["label_values"]}


def _volumes(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    labels = {os.path.basename(f): f for f in idx.loc[idx["kind"] == "label", "file"]}
    vols = idx[idx["kind"] == "volume"]
    missing = [f for f in vols["file"] if os.path.basename(f) not in labels]
    if missing or len(labels) != len(vols):
        raise ValueError(f"[{DATASET}] {len(vols)} volumes against {len(labels)} label volumes (first without a label: {missing[:1]})")
    return pd.DataFrame({"key": vols["file"].tolist(), "label": [labels[os.path.basename(f)] for f in vols["file"]],
                         "anatomy": vols["anatomy_mask"].tolist()})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    v = _volumes(dcfg)
    ev = eval_half(v["key"], seed)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in v["key"]], "dataset": DATASET, "key": v["key"], "group_id": v["key"],
                         "split": "none", "partition": ["eval" if k in ev else "dev" for k in v["key"]],
                         "image_path": [rel_to_root(os.path.join(dcfg["dir"], k), root) for k in v["key"]]})


def _stats(dcfg: Dict, key: str, label: str, anatomy: str) -> Dict:
    if key not in _STATS:
        lab, _ = read_volume(os.path.join(dcfg["dir"], label))
        head, _ = read_volume(os.path.join(dcfg["dir"], anatomy))
        if lab.shape != head.shape:
            raise ValueError(f"[{DATASET}] {label} and {anatomy} differ in shape")
        ext = extent(head > 0)
        zs = range(ext[0], ext[1] + 1) if ext else range(0)
        flat = lab.reshape(lab.shape[0], -1)
        _STATS[key] = {"deciles": {z: decile(z, *ext) for z in zs},
                       "areas": {int(v): {z: int((flat[z] == v).sum()) for z in zs} for v in np.unique(lab) if v > 0},
                       "any": {z: bool((flat[z] > 0).any()) for z in zs}}
    return _STATS[key]


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    value = _label_values(dcfg)[fcfg["source_label"]]
    vols = _volumes(dcfg).set_index("key")
    pos: List[Dict] = []
    neg: List[Dict] = []
    for _, case in ev.iterrows():
        st = _stats(dcfg, case["key"], vols.at[case["key"], "label"], vols.at[case["key"], "anatomy"])
        areas = st["areas"].get(value, {})
        pos += slice_rows(case, spaced_largest(areas, int(dcfg["max_positive_slices"]), int(dcfg["min_slice_distance"])), st["deciles"], case["image_path"])
        neg += slice_rows(case, [z for z, hit in st["any"].items() if not hit], st["deciles"], case["image_path"])
    return slice_frame(pos), slice_frame(neg)
