"""
data_loader/sources/brats.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, rel_to_root
from data_loader.slices import decile, extent, read_index, read_volume, slice_frame, slice_rows


DATASET = "brats2021"
_STATS: Dict[Tuple[str, str], Dict] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"])}


def _subjects(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    rows: Dict[str, Dict] = {}
    for f in idx["file"]:
        m = re.search(dcfg["file_regex"], os.path.basename(f))
        if m:
            rows.setdefault(m.group("subject"), {"subject": m.group("subject")})[m.group("sequence")] = f
    t = pd.DataFrame(sorted(rows.values(), key=lambda r: r["subject"]))
    lacking = t[t[["flair", "t1ce", "seg"]].isna().any(axis=1)] if len(t) and {"flair", "t1ce", "seg"} <= set(t.columns) else t
    if len(t) == 0 or len(lacking):
        raise ValueError(f"[{DATASET}] {len(lacking)} of {len(t)} subjects lack a FLAIR, T1ce, or label volume in the preprocess index")
    return t


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    s = _subjects(dcfg)
    ev = eval_half(s["subject"], seed)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in s["subject"]], "dataset": DATASET, "key": s["subject"], "group_id": s["subject"],
                         "split": "none", "partition": ["eval" if k in ev else "dev" for k in s["subject"]],
                         "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in s["flair"]]})


def _stats(dcfg: Dict, subject: Dict, sequence: str) -> Dict:
    key = (subject["subject"], sequence)
    if key not in _STATS:
        img, _ = read_volume(os.path.join(dcfg["dir"], subject[sequence]))
        seg, _ = read_volume(os.path.join(dcfg["dir"], subject["seg"]))
        if img.shape != seg.shape:
            raise ValueError(f"[{DATASET}] {subject['subject']}: the {sequence} and label volumes differ in shape")
        ext = extent(img > 0)
        zs = range(ext[0], ext[1] + 1) if ext else range(0)
        flat, lab = img.reshape(img.shape[0], -1), seg.reshape(seg.shape[0], -1)
        tumor, enh = [int(v) for v in dcfg["tumor_labels"]], int(dcfg["enhancing_label"])
        _STATS[key] = {"deciles": {z: decile(z, *ext) for z in zs}, "brain": {z: int((flat[z] > 0).sum()) for z in zs},
                       "tumor": {z: int(np.isin(lab[z], tumor).sum()) for z in zs}, "enhancing": {z: int((lab[z] == enh).sum()) for z in zs}}
    return _STATS[key]


def _sequence(fcfg: Dict) -> str:
    return "flair" if fcfg["kind"] == "tumor" else "t1ce"


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = _subjects(dcfg).set_index("subject", drop=False)
    seq = _sequence(fcfg)
    pos: List[Dict] = []
    neg: List[Dict] = []
    for _, case in ev.iterrows():
        subj = subjects.loc[case["key"]].to_dict()
        st = _stats(dcfg, subj, seq)
        if not case["image_path"].endswith(subj["flair"]):
            raise ValueError(f"[{DATASET}] {case['image_path']} does not end in {subj['flair']}")
        volume = case["image_path"][: len(case["image_path"]) - len(subj["flair"])] + subj[seq]
        brain = st["brain"]
        if fcfg["kind"] == "tumor":
            p = [z for z in st["deciles"] if brain[z] and st["tumor"][z] >= float(dcfg["tumor_min_fraction"]) * brain[z]]
            n = [z for z in st["deciles"] if brain[z] and st["tumor"][z] == 0]
        else:
            p = [z for z in st["deciles"] if brain[z] and st["enhancing"][z] >= float(dcfg["enhancing_min_fraction"]) * brain[z]]
            n = [z for z in st["deciles"] if brain[z] and st["enhancing"][z] == 0 and st["tumor"][z] > 0]
        pos += slice_rows(case, p, st["deciles"], volume)
        neg += slice_rows(case, n, st["deciles"], volume)
    return slice_frame(pos), slice_frame(neg)
