"""
data_loader/sources/cbis_ddsm.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root
from data_loader.slices import read_index


DATASET = "cbis_ddsm"
_VIEW = re.compile(r"^(?P<collection>Mass|Calc)-Test_(?P<patient>P_\d+)_(?P<side>LEFT|RIGHT)_(?P<view>CC|MLO)$")


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "case_csvs": [file_signature(p) for _, p in dcfg["case_csvs"]]}


def _view_key(dicom_id: str) -> Tuple[str, str, str]:
    m = _VIEW.match(dicom_id)
    if m is None:
        raise ValueError(f"[{DATASET}] {dicom_id} is not a full-mammogram id of the test sets")
    return f"{m['patient']}_{m['side']}_{m['view']}", m["patient"], m["collection"]


def _first_part(path: str) -> str:
    return str(path).strip().replace("\\", "/").split("/")[0]


def _views(dcfg: Dict) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, str]]]]:
    idx = read_index(dcfg)
    rows: Dict[str, Dict] = {}
    for pid, f in zip(idx.loc[idx["kind"] == "image", "patient_id"], idx.loc[idx["kind"] == "image", "file"]):
        key, patient, collection = _view_key(pid)
        if key not in rows or collection == "Mass":
            rows[key] = {"key": key, "patient": patient, "file": f}
    masks = dict(zip(idx.loc[idx["kind"] == "mask", "patient_id"], idx.loc[idx["kind"] == "mask", "file"]))
    types: Dict[str, set] = {k: set() for k in rows}
    rois: Dict[str, List[Tuple[str, str]]] = {k: [] for k in rows}
    for kind, path in dcfg["case_csvs"]:
        c = read_csv_defensively(path, dtype=str)
        missing = [col for col in (dcfg["image_path_column"], dcfg["roi_path_column"]) if col not in c.columns]
        if missing:
            raise ValueError(f"[{DATASET}] {os.path.basename(path)} lacks {missing}")
        for image_path, roi_path in zip(c[dcfg["image_path_column"]], c[dcfg["roi_path_column"]]):
            key = _view_key(_first_part(image_path))[0]
            if key not in rows:
                raise ValueError(f"[{DATASET}] {key} of {os.path.basename(path)} has no preprocessed full mammogram")
            roi = masks.get(_first_part(roi_path))
            if roi is None:
                raise ValueError(f"[{DATASET}] the ROI mask {_first_part(roi_path)} was not preprocessed")
            types[key].add(str(kind))
            rois[key].append((str(kind), roi))
    t = pd.DataFrame(list(rows.values()))
    for kind, _ in dcfg["case_csvs"]:
        t[str(kind)] = [str(kind) in types[k] for k in t["key"]]
    return t, rois


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t, _ = _views(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["patient"],
                         "split": "test", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _views(dcfg)[0].set_index("key")
    has = ev["key"].map(t[fcfg["source_label"]]).astype(bool).to_numpy()
    other = ev["key"].map(t[fcfg["negative_label"]]).astype(bool).to_numpy()
    return ev[has].copy(), ev[~has & other].copy()
