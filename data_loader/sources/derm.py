"""
data_loader/sources/derm.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, mask_is_empty, read_csv_defensively, rel_to_root, tree_signature


DATASET = "isic2019"
_EMPTY: Dict[str, bool] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"groundtruth_csv": file_signature(dcfg["groundtruth_csv"]), "metadata_csv": file_signature(dcfg["metadata_csv"]),
            "image_dir": tree_signature(dcfg["image_dir"]), "lesion_mask_dir": tree_signature(dcfg["lesion_mask_dir"]),
            "attribute_mask_dir": tree_signature(dcfg["attribute_mask_dir"])}


def _table(dcfg: Dict) -> pd.DataFrame:
    gt = read_csv_defensively(dcfg["groundtruth_csv"], dtype={"image": str})
    md = read_csv_defensively(dcfg["metadata_csv"], usecols=["image", "lesion_id"], dtype=str)
    d = gt.merge(md, on="image", how="left", validate="one_to_one")
    d["key"] = d["image"].str.replace(dcfg["downsampled_suffix"], "", regex=False)
    if d["key"].duplicated().any():
        raise ValueError(f"[{DATASET}] {int(d['key'].duplicated().sum())} ids repeat once the suffix {dcfg['downsampled_suffix']!r} is removed")
    d["group_id"] = d["lesion_id"].fillna(d["image"])
    return d


def _attribute_path(dcfg: Dict, key: str, name: str) -> str:
    return os.path.join(dcfg["attribute_mask_dir"], f"{key}_attribute_{name}.png")


def _lesion_path(dcfg: Dict, key: str) -> str:
    return os.path.join(dcfg["lesion_mask_dir"], f"{key}_segmentation.png")


def _empty_flags(paths: List[str]) -> np.ndarray:
    from tqdm import tqdm
    todo = [p for p in paths if p not in _EMPTY]
    for p in tqdm(todo, desc=f"[{DATASET}] attribute masks", unit="mask", disable=not todo):
        _EMPTY[p] = mask_is_empty(p)
    return np.array([_EMPTY[p] for p in paths], dtype=bool)


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    d = _table(dcfg)
    on_disk = set(os.listdir(dcfg["image_dir"]))
    names = []
    for k in d["key"]:
        name = next((f"{k}{ext}" for ext in dcfg["image_extensions"] if f"{k}{ext}" in on_disk), None)
        if name is None:
            raise ValueError(f"[{DATASET}] {k} has no image in {dcfg['image_dir']} under {dcfg['image_extensions']}")
        names.append(name)
    evaluation = eval_half(d["group_id"], seed)
    base = rel_to_root(dcfg["image_dir"], root)
    return pd.DataFrame({
        "case_id": [make_case_id(DATASET, k) for k in d["key"]], "dataset": DATASET, "key": d["key"], "group_id": d["group_id"],
        "split": dcfg["release_split"], "partition": np.where(d["group_id"].isin(evaluation), "eval", "dev"),
        "image_path": [os.path.join(base, n) for n in names]})


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if fcfg["kind"] == "class":
        d = _table(dcfg).set_index("key")
        hit = (d[fcfg["positive_classes"]] == 1).any(axis=1)
        is_pos = ev["key"].map(hit)
        if is_pos.isna().any():
            raise ValueError(f"[{DATASET}] {int(is_pos.isna().sum())} evaluation images are missing from the ground truth")
        return ev[is_pos.astype(bool)].copy(), ev[~is_pos.astype(bool)].copy()
    if fcfg["kind"] != "attribute":
        raise ValueError(f"[{DATASET}] unknown finding kind {fcfg['kind']!r}")
    on_disk = set(os.listdir(dcfg["attribute_mask_dir"]))
    sub = ev[[f"{k}_attribute_{fcfg['source_label']}.png" in on_disk for k in ev["key"]]]
    empty = _empty_flags([_attribute_path(dcfg, k, fcfg["source_label"]) for k in sub["key"]])
    return sub[~empty].copy(), sub[empty].copy()
