"""
data_loader/sources/lidc.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, rel_to_root
from data_loader.slices import decile, extent, read_index, read_volume, slice_frame, slice_rows


DATASET = "lidc"
_STATS: Dict[str, Dict] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "lungs": "largest component without the table core", "lung_slice_min_mm2": float(dcfg["lung_slice_min_mm2"]),
            "table_area_ratio": float(dcfg["table_area_ratio"]), "positives": "largest outline per nodule"}


def _series(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    s = idx[idx["kind"] == "volume"].copy()
    missing = [c for c in ("anatomy_mask", "nodule_mask", "n_marked_3mm") if c not in s.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the preprocess index lacks {missing}; run main_preprocess_download for lidc again")
    return s.reset_index(drop=True)


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    s = _series(dcfg)
    ev = eval_half(s["source"], seed)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in s["file"]], "dataset": DATASET, "key": s["file"], "group_id": s["source"],
                         "split": "none", "partition": ["eval" if p in ev else "dev" for p in s["source"]],
                         "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in s["file"]]})


def _lungs(dcfg: Dict, row: pd.Series) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    from scipy import ndimage
    mask, spacing = read_volume(os.path.join(dcfg["dir"], row["anatomy_mask"]))
    lab, n = ndimage.label(mask > 0)
    sizes = np.bincount(lab.ravel())[1:]
    objs = ndimage.find_objects(lab)
    for k in np.argsort(-sizes, kind="stable"):
        sub = lab[objs[k]] == k + 1
        area = sub.reshape(sub.shape[0], -1).sum(axis=1)
        if sub.shape[0] == mask.shape[0] and area.min() >= float(dcfg["table_area_ratio"]) * area.max():
            continue
        return lab == k + 1, spacing
    return np.zeros(mask.shape, dtype=bool), spacing


def _largest_slices(nod: np.ndarray) -> List[int]:
    flat = nod.reshape(nod.shape[0], -1)
    zi, pi = np.nonzero(flat)
    labels = flat[zi, pi].astype(np.int64)
    counts = np.zeros((int(labels.max()) + 1 if labels.size else 1, nod.shape[0]), dtype=np.int64)
    np.add.at(counts, (labels, zi), 1)
    return sorted({int(np.argmax(counts[k])) for k in range(1, counts.shape[0]) if counts[k].any()})


def _stats(dcfg: Dict, key: str, row: pd.Series) -> Dict:
    if key not in _STATS:
        lungs, (sx, sy, _) = _lungs(dcfg, row)
        nod, _ = read_volume(os.path.join(dcfg["dir"], row["nodule_mask"]))
        area = lungs.reshape(lungs.shape[0], -1).sum(axis=1) * sx * sy
        ext = extent((area >= float(dcfg["lung_slice_min_mm2"]))[:, None])
        zs = range(ext[0], ext[1] + 1) if ext else range(0)
        _STATS[key] = {"deciles": {z: decile(z, *ext) for z in zs}, "nodule_slices": [z for z in _largest_slices(nod) if z in zs]}
    return _STATS[key]


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    series = _series(dcfg).set_index("file")
    pos: List[Dict] = []
    neg: List[Dict] = []
    for _, case in ev.iterrows():
        row = series.loc[case["key"]]
        st = _stats(dcfg, case["key"], row)
        pos += slice_rows(case, st["nodule_slices"], st["deciles"], case["image_path"])
        if int(float(row["n_marked_3mm"])) == 0:
            neg += slice_rows(case, list(st["deciles"]), st["deciles"], case["image_path"])
    return slice_frame(pos), slice_frame(neg)
