"""
data_loader/sources/camelyon17.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "camelyon17"
_CACHE: Dict[str, pd.DataFrame] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"metadata_csv": file_signature(dcfg["metadata_csv"])}


def _patches(dcfg: Dict) -> pd.DataFrame:
    if dcfg["metadata_csv"] not in _CACHE:
        t = read_csv_defensively(dcfg["metadata_csv"], index_col=0, dtype={"patient": str})
        missing = [c for c in ("patient", "node", "x_coord", "y_coord", "tumor", "slide", "center") if c not in t.columns]
        if missing:
            raise ValueError(f"[{DATASET}] metadata.csv lacks {missing}")
        p, n = t["patient"].astype(str), t["node"].astype(int).astype(str)
        rel = ("patches/patient_" + p + "_node_" + n + "/patch_patient_" + p + "_node_" + n + "_x_" + t["x_coord"].astype(int).astype(str)
               + "_y_" + t["y_coord"].astype(int).astype(str) + ".png")
        _CACHE[dcfg["metadata_csv"]] = pd.DataFrame({"key": rel.to_numpy(), "patient": p.to_numpy(), "slide": t["slide"].astype(int).astype(str).to_numpy(),
                                                     "center": t["center"].astype(int).astype(str).to_numpy(), "tumor": t["tumor"].astype(int).to_numpy()})
    return _CACHE[dcfg["metadata_csv"]]


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _patches(dcfg)
    base = rel_to_root(dcfg["dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k[len("patches/"):-len(".png")]) for k in t["key"]], "dataset": DATASET, "key": t["key"],
                         "group_id": t["patient"], "split": t["center"], "partition": np.where(t["center"].isin([str(c) for c in dcfg["eval_centers"]]), "eval", "dev"),
                         "image_path": [os.path.join(base, k) for k in t["key"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _patches(dcfg).set_index("key")
    rows = ev[ev["split"].isin([str(c) for c in fcfg["centers"]])] if "centers" in fcfg else ev
    rows = rows.assign(cap_id=rows["key"].map(t["slide"]).to_numpy())
    tumor = rows["key"].map(t["tumor"]).to_numpy()
    return rows[tumor == 1].copy(), rows[tumor == 0].copy()
