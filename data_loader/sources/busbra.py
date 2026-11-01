"""
data_loader/sources/busbra.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, read_csv_defensively, rel_to_root
from data_loader.slices import read_index
from Inference.resume_utils import MissingInput


DATASET = "busbra"


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "metadata_csv": file_signature(dcfg["metadata_csv"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    unset = [k for k in ("id_column", "pathology_column", "malignant_value", "patient_regex") if not dcfg.get(k)]
    if unset:
        raise MissingInput(f"datasets.busbra lacks {unset}; read them from the BUS-BRA metadata file and set them in config")
    idx = read_index(dcfg)
    meta = read_csv_defensively(dcfg["metadata_csv"], dtype=str).set_index(dcfg["id_column"])
    rows = []
    for r in idx.itertuples(index=False):
        name = os.path.splitext(os.path.basename(r.image))[0]
        if name not in meta.index:
            raise ValueError(f"[{DATASET}] {name} is not in the metadata file")
        m = re.search(dcfg["patient_regex"], name)
        if m is None:
            raise ValueError(f"[{DATASET}] patient_regex does not match {name}")
        rows.append({"key": name, "image": r.image, "mask": getattr(r, "mask", ""), "patient": m.group(1),
                     "malignant": meta.at[name, dcfg["pathology_column"]] == dcfg["malignant_value"]})
    return pd.DataFrame(rows)


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    ev = eval_half(t["patient"], seed)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["patient"],
                         "split": "none", "partition": ["eval" if p in ev else "dev" for p in t["patient"]],
                         "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in t["image"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    malignant = _images(dcfg).set_index("key")["malignant"]
    hit = ev["key"].map(malignant).astype(bool)
    return ev[hit].copy(), ev[~hit].copy()
