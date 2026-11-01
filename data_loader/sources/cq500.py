"""
data_loader/sources/cq500.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root
from data_loader.slices import decile, extent, lps_box, read_index, read_volume, slice_frame, slice_rows


DATASET = "cq500"
_STATS: Dict[str, Dict] = {}
_BOXES: Dict[str, pd.DataFrame] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "reads_csv": file_signature(dcfg["reads_csv"]), "bhx_csv": file_signature(dcfg["bhx_csv"]),
            "pools": "positives from scans read as hemorrhage, negatives from clean scans without a box"}


def _scans(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    vols = idx[idx["kind"] == "volume"]
    return pd.DataFrame({"key": vols["source"].to_numpy(), "file": vols["file"].to_numpy(), "anatomy": vols["anatomy_mask"].to_numpy()})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _scans(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["file"]]})


def _box_numbers(text: str) -> List[float]:
    named = dict(re.findall(r"(x|y|width|height)['\"]?\s*[:=]\s*(-?[0-9.]+(?:[eE]-?[0-9]+)?)", str(text)))
    if all(k in named for k in ("x", "y", "width", "height")):
        return [float(named[k]) for k in ("x", "y", "width", "height")]
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?", str(text))
    if len(nums) < 4:
        raise ValueError(f"[{DATASET}] cannot read a box from {text!r}")
    return [float(v) for v in nums[:4]]


def _boxes(dcfg: Dict) -> pd.DataFrame:
    if dcfg["bhx_csv"] in _BOXES:
        return _BOXES[dcfg["bhx_csv"]]
    bhx = read_csv_defensively(dcfg["bhx_csv"], dtype=str)
    cols = dcfg["bhx_columns"]
    missing = [cols[k] for k in ("sop", "label", "type", "box") if cols[k] not in bhx.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the BHX file lacks {missing}; its columns are {list(bhx.columns)}")
    bhx = bhx[bhx[cols["type"]] == dcfg["hand_drawn_type"]]
    scans = _scans(dcfg).set_index("key")
    where = {}
    for scan in scans.index:
        path = os.path.join(dcfg["slices_dir"], f"{scan}.csv")
        if os.path.exists(path):
            s = read_csv_defensively(path, dtype={"sop_uid": str})
            where.update({sop: (scan, int(k)) for sop, k in zip(s["sop_uid"], s["slice_index"])})
    rows = []
    for sop, label, data in zip(bhx[cols["sop"]], bhx[cols["label"]], bhx[cols["box"]]):
        if sop not in where:
            continue
        scan, k = where[sop]
        z, corners = lps_box(os.path.join(dcfg["dir"], scans.at[scan, "file"]), k, _box_numbers(data))
        rows.append({"scan": scan, "z": z, "label": str(label).strip(), "x_min": corners[0], "y_min": corners[1], "x_max": corners[2], "y_max": corners[3]})
    _BOXES[dcfg["bhx_csv"]] = pd.DataFrame(rows, columns=["scan", "z", "label", "x_min", "y_min", "x_max", "y_max"])
    return _BOXES[dcfg["bhx_csv"]]


def _stats(dcfg: Dict, key: str, anatomy: str) -> Dict:
    if key not in _STATS:
        head, _ = read_volume(os.path.join(dcfg["dir"], anatomy))
        ext = extent(head > 0)
        zs = range(ext[0], ext[1] + 1) if ext else range(0)
        _STATS[key] = {"deciles": {z: decile(z, *ext) for z in zs}}
    return _STATS[key]


def _clean_scans(dcfg: Dict) -> set:
    reads = read_csv_defensively(dcfg["reads_csv"], dtype={dcfg["reads_name_column"]: str})
    missing = [c for c in [dcfg["reads_name_column"]] + list(dcfg["reader_columns"]) if c not in reads.columns]
    if missing:
        raise ValueError(f"[{DATASET}] reads.csv lacks {missing}")
    clean = (reads[list(dcfg["reader_columns"])].astype(float) == 0).all(axis=1)
    return set(reads.loc[clean, dcfg["reads_name_column"]].str.strip())


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scans = _scans(dcfg).set_index("key")
    boxes = _boxes(dcfg)
    typed = boxes[boxes["label"].isin(fcfg["source_labels"])]
    clean = _clean_scans(dcfg)
    boxed = set(boxes["scan"])
    pos: List[Dict] = []
    neg: List[Dict] = []
    for _, case in ev.iterrows():
        st = _stats(dcfg, case["key"], scans.at[case["key"], "anatomy"])
        if case["key"] not in clean:
            hit = sorted({int(z) for z in typed.loc[typed["scan"] == case["key"], "z"]} & set(st["deciles"]))
            pos += slice_rows(case, hit, st["deciles"], case["image_path"])
        elif case["key"] not in boxed:
            neg += slice_rows(case, list(st["deciles"]), st["deciles"], case["image_path"])
    return slice_frame(pos), slice_frame(neg)
