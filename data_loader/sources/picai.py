"""
data_loader/sources/picai.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import eval_half, file_signature, make_case_id, read_csv_defensively, rel_to_root, tree_signature
from data_loader.slices import decile, extent, read_index, read_volume, slice_frame, slice_rows
from Inference.resume_utils import MissingInput


DATASET = "picai"
_STATS: Dict[str, Dict] = {}


def source_signature(dcfg: Dict) -> Dict:
    return {"index_csv": file_signature(dcfg["index_csv"]), "marksheet": file_signature(dcfg["marksheet"]),
            "lesion_dirs": [tree_signature(d) for d in dcfg["lesion_dirs"]], "gland_dir": tree_signature(dcfg["gland_dir"])}


def _lesion_path(dcfg: Dict, key: str) -> Optional[str]:
    for d in dcfg["lesion_dirs"]:
        path = os.path.join(d, f"{key}.nii.gz")
        if os.path.exists(path):
            return path
    return None


def _lesion_on_gland(lpath: str, gpath: str) -> np.ndarray:
    import SimpleITK as sitk
    les, gland = sitk.ReadImage(lpath), sitk.ReadImage(gpath)
    same = (les.GetSize() == gland.GetSize() and np.allclose(les.GetSpacing(), gland.GetSpacing(), atol=1e-3)
            and np.allclose(les.GetOrigin(), gland.GetOrigin(), atol=1e-2) and np.allclose(les.GetDirection(), gland.GetDirection(), atol=1e-3))
    if same:
        return read_volume(lpath)[0]
    les = sitk.Resample(les, gland, sitk.Transform(), sitk.sitkNearestNeighbor, 0, les.GetPixelID())
    return sitk.GetArrayFromImage(sitk.DICOMOrient(les, "LPS"))


def _cases(dcfg: Dict) -> pd.DataFrame:
    idx = read_index(dcfg)
    rows = []
    for f in idx.loc[idx["kind"] == "volume", "file"]:
        m = re.search(dcfg["file_regex"], os.path.basename(f))
        if m:
            rows.append({"file": f, "patient": m.group("patient"), "study": m.group("study")})
    cases = pd.DataFrame(rows)
    sheet = read_csv_defensively(dcfg["marksheet"], dtype=str)
    cols = [dcfg["patient_column"], dcfg["study_column"], dcfg["label_column"], dcfg["center_column"]]
    missing = [c for c in cols if c not in sheet.columns]
    if missing:
        raise ValueError(f"[{DATASET}] the marksheet lacks {missing}; set the column names under datasets.picai")
    sheet = sheet[cols].set_axis(["patient", "study", "cspca", "center"], axis=1)
    out = cases.merge(sheet, on=["patient", "study"], how="left", validate="one_to_one")
    if out["center"].isna().any():
        raise ValueError(f"[{DATASET}] {int(out['center'].isna().sum())} volumes have no marksheet row")
    return out


def partition_table(dcfg: Dict, root: str, seed: int) -> pd.DataFrame:
    c = _cases(dcfg)
    main = c["center"] == dcfg["main_center"]
    ev = eval_half(c.loc[main, "patient"], seed)
    key = c["patient"] + "_" + c["study"]
    return pd.DataFrame({"case_id": [make_case_id(DATASET, k) for k in key], "dataset": DATASET, "key": key, "group_id": c["patient"],
                         "split": c["center"], "partition": np.where(~main | c["patient"].isin(ev), "eval", "dev"),
                         "image_path": [rel_to_root(os.path.join(dcfg["dir"], f), root) for f in c["file"]]})


def _crop(gland: np.ndarray, factor: float) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(gland.any(axis=0))
    h, w = gland.shape[1:]
    side = int(min(h, w, np.ceil(factor * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))))
    cy, cx = (ys.min() + ys.max() + 1) / 2.0, (xs.min() + xs.max() + 1) / 2.0
    y0 = int(np.clip(round(cy - side / 2.0), 0, max(0, h - side)))
    x0 = int(np.clip(round(cx - side / 2.0), 0, max(0, w - side)))
    return y0, min(h, y0 + side), x0, min(w, x0 + side)


def _stats(dcfg: Dict, key: str) -> Optional[Dict]:
    from scipy import ndimage
    if key not in _STATS:
        gpath = os.path.join(dcfg["gland_dir"], f"{key}.nii.gz")
        if not os.path.exists(gpath):
            _STATS[key] = None
            return None
        gland, _ = read_volume(gpath)
        ext = extent(gland > 0)
        lpath = _lesion_path(dcfg, key)
        lesion_slices: List[int] = []
        if lpath:
            les = _lesion_on_gland(lpath, gpath)
            if les.shape != gland.shape:
                raise ValueError(f"[{DATASET}] {key}: the lesion and gland masks differ in shape")
            lab, n = ndimage.label(les > 0)
            for k in range(1, n + 1):
                areas = (lab == k).reshape(lab.shape[0], -1).sum(axis=1)
                lesion_slices.append(int(np.argmax(areas)))
        zs = range(ext[0], ext[1] + 1) if ext else range(0)
        _STATS[key] = {"deciles": {z: decile(z, *ext) for z in zs}, "crop": _crop(gland > 0, float(dcfg["crop_factor"])) if ext else None,
                       "lesion_slices": sorted(set(lesion_slices)), "has_lesion_mask": lpath is not None}
    return _STATS[key]


def finding_pools(fcfg: Dict, ev: pd.DataFrame, dcfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cases = _cases(dcfg)
    label = dict(zip(cases["patient"] + "_" + cases["study"], cases["cspca"].str.upper()))
    rows = ev[ev["split"].isin(fcfg.get("centers", [dcfg["main_center"]]))]
    pos: List[Dict] = []
    neg: List[Dict] = []
    skipped = 0
    for _, case in rows.iterrows():
        st = _stats(dcfg, case["key"])
        if st is None or st["crop"] is None:
            skipped += 1
            continue
        if label[case["key"]] == dcfg["positive_value"]:
            if not st["has_lesion_mask"]:
                skipped += 1
                continue
            pos += slice_rows(case, [z for z in st["lesion_slices"] if z in st["deciles"]], st["deciles"], case["image_path"], st["crop"])
        elif label[case["key"]] == dcfg["negative_value"]:
            neg += slice_rows(case, list(st["deciles"]), st["deciles"], case["image_path"], st["crop"])
    if not pos:
        raise MissingInput(f"{DATASET}: no csPCa lesion slice found; check lesion_dirs and gland_dir")
    return slice_frame(pos), slice_frame(neg)
