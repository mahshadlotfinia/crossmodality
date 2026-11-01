"""
data_loader/sources/ddi.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root


DATASET = "ddi"


def source_signature(dcfg: Dict) -> Dict:
    return {"metadata_csv": file_signature(dcfg["metadata_csv"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    t = read_csv_defensively(dcfg["metadata_csv"], dtype={"DDI_file": str})
    tones = {int(code): str(name) for code, name in dcfg["skin_tone_groups"]}
    unknown = sorted(set(t["skin_tone"].astype(int)) - set(tones))
    if unknown:
        raise ValueError(f"[{DATASET}] skin tone codes {unknown} are not in datasets.ddi.skin_tone_groups")
    malignant = t["malignant"].astype(str).str.strip().str.lower()
    if not malignant.isin(["true", "false"]).all():
        raise ValueError(f"[{DATASET}] malignant holds values other than True and False")
    return pd.DataFrame({"key": t["DDI_file"].str.replace(r"\.png$", "", regex=True), "file": t["DDI_file"],
                         "stratum": t["skin_tone"].astype(int).map(tones), "malignant": malignant == "true"})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": "none", "partition": "eval", "image_path": [os.path.join(base, f) for f in t["file"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _images(dcfg).set_index("key")
    out = ev.assign(stratum=ev["key"].map(t["stratum"]).to_numpy())
    hit = ev["key"].map(t["malignant"]).astype(bool).to_numpy()
    return out[hit].copy(), out[~hit].copy()
