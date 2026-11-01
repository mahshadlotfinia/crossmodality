"""
data_loader/slices.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader.build_utils import read_csv_defensively


_VOLUMES: "OrderedDict[str, Tuple[np.ndarray, Tuple[float, float, float]]]" = OrderedDict()
_KEEP = 6
_GEOMETRY: "OrderedDict[str, Tuple[object, object]]" = OrderedDict()


def read_volume(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    import SimpleITK as sitk
    key = os.path.abspath(path)
    if key in _VOLUMES:
        _VOLUMES.move_to_end(key)
        return _VOLUMES[key]
    img = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    out = (sitk.GetArrayFromImage(img), tuple(float(v) for v in img.GetSpacing()))
    _VOLUMES[key] = out
    while len(_VOLUMES) > _KEEP:
        _VOLUMES.popitem(last=False)
    return out


def lps_box(path: str, k: int, box: Tuple[float, float, float, float]) -> Tuple[int, List[float]]:
    import SimpleITK as sitk
    key = os.path.abspath(path)
    if key not in _GEOMETRY:
        img = sitk.ReadImage(path)
        _GEOMETRY[key] = (img, sitk.DICOMOrient(img, "LPS"))
        while len(_GEOMETRY) > _KEEP:
            _GEOMETRY.popitem(last=False)
    img, lps = _GEOMETRY[key]
    x, y, w, h = (float(v) for v in box)
    corners = [lps.TransformPhysicalPointToContinuousIndex(img.TransformContinuousIndexToPhysicalPoint((cx - 0.5, cy - 0.5, float(k))))
               for cx, cy in ((x, y), (x + w, y + h))]
    xs, ys, zs = zip(*corners)
    return int(round(float(np.mean(zs)))), [min(xs) + 0.5, min(ys) + 0.5, max(xs) + 0.5, max(ys) + 0.5]


def slice_path(volume_rel: str, z: int, crop: Optional[Tuple[int, int, int, int]] = None) -> str:
    out = f"{volume_rel}#z={int(z)}"
    return out + (f"#crop={crop[0]}:{crop[1]},{crop[2]}:{crop[3]}" if crop is not None else "")


def parse_slice_path(image_path: str) -> Tuple[str, int, Optional[Tuple[int, int, int, int]]]:
    path, *specs = image_path.split("#")
    fields = dict(s.split("=", 1) for s in specs)
    crop = None
    if "crop" in fields:
        ys, xs = fields["crop"].split(",")
        crop = tuple(int(v) for v in ys.split(":")) + tuple(int(v) for v in xs.split(":"))
    return path, int(fields["z"]), crop


def is_slice_path(image_path: str) -> bool:
    return "#z=" in image_path


def load_slice(image_path: str, root: str) -> np.ndarray:
    path, z, crop = parse_slice_path(image_path)
    vol, _ = read_volume(os.path.join(root, path))
    s = vol[z]
    if crop is not None:
        s = s[crop[0]:crop[1], crop[2]:crop[3]]
    s = np.clip(s, 0, 255).astype(np.uint8)
    return np.repeat(s[..., None], 3, axis=2)


def extent(mask_volume: np.ndarray) -> Optional[Tuple[int, int]]:
    zs = np.flatnonzero(mask_volume.reshape(mask_volume.shape[0], -1).any(axis=1))
    return (int(zs[0]), int(zs[-1])) if zs.size else None


def decile(z: int, lo: int, hi: int) -> int:
    return int(min(9, max(0, np.floor(10.0 * (z - lo) / (hi - lo + 1)))))


def spaced_largest(areas: Dict[int, int], k: int, distance: int) -> List[int]:
    taken: List[int] = []
    for z in sorted(areas, key=lambda z: (-areas[z], z)):
        if areas[z] > 0 and all(abs(z - t) >= distance for t in taken):
            taken.append(z)
        if len(taken) == k:
            break
    return sorted(taken)


def read_index(dcfg: Dict) -> pd.DataFrame:
    return read_csv_defensively(dcfg["index_csv"], dtype=str, keep_default_na=False)


SLICE_COLUMNS = ["case_id", "dataset", "key", "group_id", "split", "partition", "image_path", "z", "decile"]


def slice_frame(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SLICE_COLUMNS)


def slice_rows(case: pd.Series, zs: List[int], deciles: Dict[int, int], volume_rel: str, crop=None) -> List[Dict]:
    return [{"case_id": f"{case['case_id']}__z{z:03d}", "dataset": case["dataset"], "key": f"{case['key']}#z={z}", "group_id": case["group_id"],
             "split": case["split"], "partition": case["partition"], "image_path": slice_path(volume_rel, z, crop), "z": int(z), "decile": deciles.get(z, -1)}
            for z in zs]
