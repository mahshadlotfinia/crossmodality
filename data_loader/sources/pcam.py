"""
data_loader/sources/pcam.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "pcam"
PATCH = 96


def source_signature(dcfg: Dict) -> Dict:
    return {"x_h5": file_signature(dcfg["x_h5"]), "y_h5": file_signature(dcfg["y_h5"]), "meta_csv": file_signature(dcfg["meta_csv"])}


def _labels(dcfg: Dict) -> np.ndarray:
    import h5py
    with h5py.File(dcfg["y_h5"], "r") as f:
        return np.asarray(f["y"]).reshape(-1).astype(int)


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    import h5py
    y = _labels(dcfg)
    meta = read_csv_defensively(dcfg["meta_csv"], usecols=["wsi", "center_tumor_patch"], dtype={"wsi": str})
    with h5py.File(dcfg["x_h5"], "r") as f:
        n_x, shape = int(f["x"].shape[0]), tuple(f["x"].shape[1:])
    if not (len(meta) == len(y) == n_x) or shape != (PATCH, PATCH, 3):
        raise ValueError(f"[{DATASET}] {len(meta)} metadata rows, {len(y)} labels, {n_x} patches of {shape}: the files do not belong together")
    agree = (meta["center_tumor_patch"].astype(str).str.lower() == "true").astype(int).to_numpy() == y
    if not agree.all():
        raise ValueError(f"[{DATASET}] the metadata's center label differs from the label file at {int((~agree).sum())} rows")
    keys = [f"{i:05d}" for i in range(n_x)]
    base = rel_to_root(dcfg["x_h5"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in keys], "dataset": DATASET, "key": keys,
                         "group_id": meta["wsi"], "split": dcfg["release_split"], "partition": "eval",
                         "image_path": [f"{base}#{i}" for i in range(n_x)]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = _labels(dcfg)
    label = y[ev["key"].astype(int).to_numpy()]
    return ev[label == 1].copy(), ev[label == 0].copy()
