"""
data_loader/sources/cmmd.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, rel_to_root
from data_loader.slices import read_index


DATASET = "cmmd"
_NOTED: Dict[str, bool] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "clinical_xlsx": file_signature(dcfg["clinical_xlsx"])}


def _breasts(dcfg: Dict) -> Dict[Tuple[str, str], str]:
    clin = pd.read_excel(dcfg["clinical_xlsx"], dtype=str)
    need = [c for c in (dcfg["side_column"], dcfg["abnormality_column"]) if c not in clin.columns]
    ids = [c for c in dcfg["id_columns"] if c in clin.columns]
    if need or not ids:
        raise ValueError(f"[{DATASET}] the clinical file lacks {need or dcfg['id_columns']}; its columns are {list(clin.columns)}")
    out = {}
    for _, r in clin.iterrows():
        pid = next((str(r[c]).strip() for c in ids if isinstance(r[c], str) and r[c].strip()), None)
        if pid is None:
            raise ValueError(f"[{DATASET}] a clinical row has no patient id in {ids}")
        out[(pid, str(r[dcfg["side_column"]]).strip().upper()[:1])] = str(r[dcfg["abnormality_column"]]).strip().lower()
    return out


def _images(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    img = idx[idx["kind"] == "image"]
    breasts = _breasts(dcfg)
    side = img["laterality"].astype(str).str.strip().str.upper().str[:1]
    abnormality = pd.Series([breasts.get((p, s)) for p, s in zip(img["patient_id"], side)], index=img.index)
    keep = abnormality.notna()
    if not keep.any():
        raise ValueError(f"[{DATASET}] no image has a clinical row for its patient and side")
    if not _NOTED.get(dcfg["index_csv"]):
        _NOTED[dcfg["index_csv"]] = True
    img = img[keep]
    return pd.DataFrame({"key": [os.path.splitext(os.path.basename(f))[0] for f in img["file"]], "patient": img["patient_id"].to_numpy(),
                         "file": img["file"].to_numpy(), "abnormality": abnormality[keep].to_numpy()})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["patient"],
                         "split": "none", "partition": "eval", "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kind = ev["key"].map(_images(dcfg).set_index("key")["abnormality"])
    return ev[kind.isin(fcfg["positive_values"])].copy(), ev[kind.isin(fcfg["negative_values"])].copy()
