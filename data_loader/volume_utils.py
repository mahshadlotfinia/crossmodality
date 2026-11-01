"""
data_loader/volume_utils.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np


def window_to_uint8(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2.0, center + width / 2.0
    return np.rint((np.clip(hu.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)


def percentile_to_uint8(vol: np.ndarray, mask: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    inside = vol[mask] if mask.any() else vol.ravel()
    lo, hi = np.percentile(inside.astype(np.float64), [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return np.rint((np.clip(vol.astype(np.float64), lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values.astype(np.float64), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def largest_component_3d(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fill_holes_per_slice(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return np.stack([ndimage.binary_fill_holes(s) for s in mask])


def head_mask(hu: np.ndarray, low: float = -15.0, high: float = 100.0) -> np.ndarray:
    return fill_holes_per_slice(largest_component_3d((hu >= low) & (hu <= high)))


def lung_mask(hu: np.ndarray, body_min: float = -500.0, lung_max: float = -400.0) -> np.ndarray:
    body = fill_holes_per_slice(largest_component_3d(hu > body_min))
    return body & (hu < lung_max)


def write_nifti_like(arr: np.ndarray, reference, path: str) -> None:
    import SimpleITK as sitk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference)
    tmp = f"{path[:-len('.nii.gz')]}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.nii.gz"
    try:
        sitk.WriteImage(img, tmp, useCompression=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_dicom_series(directory: str, series_uid: Optional[str] = None) -> Tuple[object, np.ndarray, List[str]]:
    import SimpleITK as sitk
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory)
    if not ids:
        raise ValueError(f"no DICOM series in {directory}")
    uid = series_uid if series_uid is not None else ids[0]
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uid))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    return img, sitk.GetArrayFromImage(img), files


def dicom_to_uint8(ds) -> np.ndarray:
    a = ds.pixel_array.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    out = (a - lo) / (hi - lo if hi > lo else 1.0) * 255.0
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        out = 255.0 - out
    return np.rint(out).astype(np.uint8)


def to_square(arr: np.ndarray, fill: int = 0) -> Tuple[np.ndarray, Dict]:
    h, w = arr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    out = np.full((side, side) + arr.shape[2:], fill, dtype=arr.dtype)
    out[top:top + h, left:left + w] = arr
    return out, {"top": top, "left": left, "side": side}


def resize_image(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS))


def prepare_2d(arr: np.ndarray, rule: str, size: int) -> Tuple[np.ndarray, Dict]:
    h, w = arr.shape[:2]
    if rule == "direct":
        return resize_image(arr, size), {"top": 0, "left": 0, "scale_x": size / w, "scale_y": size / h}
    if rule != "pad":
        raise ValueError(f"unknown 2D rule {rule!r}")
    sq, g = to_square(arr)
    s = size / g["side"]
    return resize_image(sq, size), {"top": g["top"] * s, "left": g["left"] * s, "scale_x": s, "scale_y": s}


def prepare_mask(mask: np.ndarray, rule: str, size: int) -> np.ndarray:
    from data_loader.build_utils import resize_mask
    m = mask.astype(bool)
    if rule == "pad":
        m, _ = to_square(m, fill=0)
    elif rule != "direct":
        raise ValueError(f"unknown 2D rule {rule!r}")
    return resize_mask(m, size, size)
