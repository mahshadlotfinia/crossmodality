"""
data_loader/sources/fitzpatrick17k.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, Tuple

import pandas as pd

from data_loader.build_utils import file_signature, make_case_id, read_csv_defensively, rel_to_root, tree_signature


DATASET = "fitzpatrick17k"


def source_signature(dcfg: Dict) -> Dict:
    return {"metadata_csv": file_signature(dcfg["metadata_csv"]), "image_dir": tree_signature(dcfg["image_dir"])}


def _images(dcfg: Dict) -> pd.DataFrame:
    t = read_csv_defensively(dcfg["metadata_csv"], dtype={"md5hash": str, "qc": str})
    on_disk = {os.path.splitext(f)[0] for f in os.listdir(dcfg["image_dir"])}
    groups = {int(scale): str(name) for scale, name in dcfg["skin_type_groups"]}
    keep = t["md5hash"].isin(on_disk) & t["fitzpatrick_scale"].isin(list(groups)) & ~t["qc"].fillna("").isin(dcfg["excluded_qc"])
    t = t[keep]
    return pd.DataFrame({"key": t["md5hash"], "stratum": t["fitzpatrick_scale"].astype(int).map(groups), "label": t["three_partition_label"]})


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    t = _images(dcfg)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in t["key"]], "dataset": DATASET, "key": t["key"], "group_id": t["key"],
                         "split": "none", "partition": "eval", "image_path": [os.path.join(base, f"{k}.jpg") for k in t["key"]]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = _images(dcfg).set_index("key")
    out = ev.assign(stratum=ev["key"].map(t["stratum"]).to_numpy())
    label = ev["key"].map(t["label"]).to_numpy()
    return out[label == fcfg["positive_label"]].copy(), out[label == fcfg["negative_label"]].copy()
