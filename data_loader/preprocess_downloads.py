"""
data_loader/preprocess_downloads.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.volume_utils import (dicom_to_uint8, head_mask, lung_mask, otsu_threshold, percentile_to_uint8, prepare_2d, prepare_mask,
                                      read_dicom_series, window_to_uint8, write_nifti_like)
from Inference.resume_utils import MissingInput, append_status, load_group_rows, run_units_resumable, status_path, write_csv_atomic, write_json_atomic


PROJECT = "Crossmodality"
PREPROCESS_VERSION = 3


class RawSource:

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise MissingInput(f"the download {path} is not on disk")
        self.path = path
        self.kind = "dir" if os.path.isdir(path) else "zip" if zipfile.is_zipfile(path) else "tar" if tarfile.is_tarfile(path) else None
        if self.kind is None:
            raise ValueError(f"{path} is neither a directory, a zip, nor a tar")
        self._zip, self._tar, self._members = None, None, None

    def names(self) -> List[str]:
        if self._members is None:
            if self.kind == "dir":
                out = []
                for root, dirs, files in os.walk(self.path):
                    dirs.sort()
                    out += [os.path.relpath(os.path.join(root, f), self.path) for f in sorted(files)]
                self._members = {n: n for n in out}
            elif self.kind == "zip":
                self._zip = zipfile.ZipFile(self.path)
                self._members = {i.filename: i for i in self._zip.infolist() if not i.is_dir()}
            else:
                self._tar = tarfile.open(self.path)
                self._members = {m.name: m for m in self._tar.getmembers() if m.isfile()}
        return sorted(self._members)

    def read(self, name: str) -> bytes:
        self.names()
        if self.kind == "dir":
            with open(os.path.join(self.path, name), "rb") as f:
                return f.read()
        if self.kind == "zip":
            return self._zip.read(self._members[name])
        return self._tar.extractfile(self._members[name]).read()

    def local_copy(self, name: str, scratch: str) -> str:
        if self.kind == "dir":
            return os.path.join(self.path, name)
        suffix = "".join(re.findall(r"(\.[A-Za-z0-9]+)", os.path.basename(name))[-2:]) or ".bin"
        path = os.path.join(scratch, f"{uuid.uuid4().hex}{suffix}")
        with open(path, "wb") as f:
            f.write(self.read(name))
        return path


def _image_from_bytes(data: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im.convert("RGB" if im.mode in ("RGB", "RGBA", "P", "CMYK", "YCbCr") else "L"))
    if arr.ndim == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        arr = arr[..., 0]
    return arr


def _save_png(arr: np.ndarray, path: str) -> None:
    from data_loader.render_inputs import save_png
    save_png(arr.astype(np.uint8), path)


def _copy_member(src: RawSource, name: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "wb") as f:
        f.write(src.read(name))
    os.replace(tmp, dest)


def _stem(name: str) -> str:
    base = os.path.basename(name)
    for ext in (".nii.gz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".mha", ".dcm"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _items_images(src: RawSource, d: Dict) -> List[Dict]:
    img = re.compile(d["image_pattern"])
    skip = re.compile(d["exclude_pattern"]) if d.get("exclude_pattern") else None
    names = [n for n in src.names() if img.search(n) and not (skip and skip.search(n))]
    if d.get("listed_member"):
        listed = pd.read_csv(io.BytesIO(src.read(d["listed_member"])), low_memory=False)[d["listed_column"]].dropna().astype(str)
        wanted = set(listed)
        names = [n for n in names if re.sub(d.get("strip_prefix", r"^$"), "", n) in wanted]
        if not names:
            raise ValueError(f"no image of {d['listed_member']} ({d['listed_column']}) matches the image pattern")
    return [{"id": n, "name": n} for n in names]


def _process_image(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    arr = _image_from_bytes(src.read(item["name"]))
    rel = os.path.join(d["images_out"], os.path.dirname(re.sub(d.get("strip_prefix", r"^$"), "", item["name"])), _stem(item["name"]) + ".png")
    img, _ = prepare_2d(arr, d["rule"], int(d["size"]))
    _save_png(img, os.path.join(out, rel))
    row = {"id": item["id"], "source": item["name"], "image": rel, "source_width": int(arr.shape[1]), "source_height": int(arr.shape[0])}
    if d.get("mask_regex"):
        folder = os.path.dirname(item["name"])
        parent = os.path.dirname(folder)
        stem = re.sub(d.get("mask_stem_from", r"^$"), d.get("mask_stem_to", ""), _stem(item["name"]))
        pattern = re.compile(d["mask_regex"].format(dirp=re.escape(folder + "/") if folder else "", parentp=re.escape(parent + "/") if parent else "",
                                                    stem=re.escape(stem)))
        masks = [n for n in src.names() if pattern.search(n)]
        row["n_mask_files"] = len(masks)
        if masks:
            union = np.zeros(arr.shape[:2], dtype=bool)
            for n in masks:
                m = _image_from_bytes(src.read(n))
                union |= (m if m.ndim == 2 else m[..., 0]) > 0
            mrel = os.path.join(d["masks_out"], os.path.relpath(rel, d["images_out"]))
            _save_png(prepare_mask(union, d["rule"], int(d["size"])).astype(np.uint8) * 255, os.path.join(out, mrel))
            row["mask"] = mrel
    return [row]


def _items_dicom(src: RawSource, d: Dict) -> List[Dict]:
    return [{"id": n, "name": n} for n in src.names() if n.lower().endswith(".dcm")]


def _mammo_id(name: str, ds, d: Dict) -> str:
    if not d.get("id_pattern"):
        return str(getattr(ds, "PatientID", ""))
    for part in name.replace("\\", "/").split("/"):
        if re.match(d["id_pattern"], part):
            return part
    raise ValueError(f"[P] no folder of {name} matches id_pattern {d['id_pattern']}")


def _process_mammo_dicom(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(src.read(item["name"])))
    pid = _mammo_id(item["name"], ds, d)
    lat = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", ""))
    view = str(getattr(ds, "ViewPosition", ""))
    raw = ds.pixel_array
    levels = np.unique(raw)
    is_roi = bool(d.get("roi_suffix")) and re.search(d["roi_suffix"], pid) is not None
    kind = ("mask" if len(levels) <= 2 else "cropped") if is_roi else "image"
    if kind == "cropped":
        return [{"id": item["id"], "source": item["name"], "patient_id": pid, "kind": "cropped", "file": ""}]
    sop = str(getattr(ds, "SOPInstanceUID", _stem(item["name"])))
    name = f"{pid}_{lat}_{view}_{sop[-12:]}" if d.get("name_with_sop") else pid
    folder = d["masks_out"] if kind == "mask" else d["images_out"]
    rel = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".png")
    if kind == "mask":
        arr = prepare_mask(raw > levels.min(), "pad", int(d["size"])).astype(np.uint8) * 255
    else:
        arr, _ = prepare_2d(dicom_to_uint8(ds), "pad", int(d["size"]))
    _save_png(arr, os.path.join(out, rel))
    return [{"id": item["id"], "source": item["name"], "patient_id": pid, "laterality": lat, "view": view, "sop_uid": sop, "kind": kind,
             "file": rel, "source_rows": int(raw.shape[0]), "source_columns": int(raw.shape[1])}]


def _items_by_pattern(src: RawSource, d: Dict) -> List[Dict]:
    pat = re.compile(d["volume_pattern"])
    return [{"id": n, "name": n} for n in src.names() if pat.search(n)]


def _process_volume(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    import SimpleITK as sitk
    rel = os.path.join(d["volumes_out"], re.sub(d.get("strip_prefix", r"^$"), "", item["name"]))
    rel = rel[: -len(".mha")] + ".nii.gz" if rel.endswith(".mha") else rel
    if d.get("label_pattern") and re.search(d["label_pattern"], item["name"]):
        _copy_member(src, item["name"], os.path.join(out, rel))
        return [{"id": item["id"], "source": item["name"], "file": rel, "kind": "label"}]
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        img = sitk.ReadImage(src.local_copy(item["name"], scratch))
    arr = sitk.GetArrayFromImage(img)
    row = {"id": item["id"], "source": item["name"], "file": rel, "kind": "volume", "shape": list(arr.shape)}
    if d["rule"] == "window":
        _write_window(arr, img, d, out, rel, row)
    else:
        mask = arr != 0 if d["body_mask"] == "nonzero" else arr > otsu_threshold(arr[arr > arr.min()] if (arr > arr.min()).any() else arr)
        write_nifti_like(percentile_to_uint8(arr, mask), img, os.path.join(out, rel))
    return [row]


def _write_window(arr: np.ndarray, img, d: Dict, out: str, rel: str, row: Dict) -> None:
    write_nifti_like(window_to_uint8(arr, float(d["window_center"]), float(d["window_width"])), img, os.path.join(out, rel))
    if d.get("anatomy") == "head":
        mrel = os.path.join(d["masks_out"], os.path.relpath(rel, d["volumes_out"]))
        write_nifti_like(head_mask(arr).astype(np.uint8), img, os.path.join(out, mrel))
        row["anatomy_mask"] = mrel
    elif d.get("anatomy") == "lung":
        mrel = os.path.join(d["masks_out"], os.path.relpath(rel, d["volumes_out"]))
        write_nifti_like(lung_mask(arr).astype(np.uint8), img, os.path.join(out, mrel))
        row["anatomy_mask"] = mrel


def _series_table(directory: str) -> pd.DataFrame:
    import pydicom
    rows = {}
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in sorted(files):
            try:
                ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
            except Exception:
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            r = rows.setdefault(uid, {"series_uid": uid, "folder": root, "n_files": 0, "thickness": float(getattr(ds, "SliceThickness", 0) or 0),
                                      "kernel": str(getattr(ds, "ConvolutionKernel", "")), "description": str(getattr(ds, "SeriesDescription", ""))})
            r["n_files"] += 1
    return pd.DataFrame(list(rows.values()))


def _items_cq500(src: RawSource, d: Dict) -> List[Dict]:
    if src.kind != "dir":
        raise ValueError("CQ500 is read from a directory of extracted scans")
    return [{"id": s, "name": s} for s in sorted(os.listdir(src.path)) if os.path.isdir(os.path.join(src.path, s)) and re.search(d["scan_pattern"], s)]


def _bhx_series(d: Dict) -> set:
    path = d["bhx_csv"]
    if not os.path.isfile(path):
        raise MissingInput(f"[cq500] {path} is not on disk; place the BHX files before preprocessing CQ500")
    t = read_csv_defensively(path)
    if d["bhx_series_column"] not in t.columns:
        raise ValueError(f"[cq500] {path} lacks {d['bhx_series_column']}; its columns are {list(t.columns)}")
    return set(t[d["bhx_series_column"]].dropna().astype(str))


def _process_cq500(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    import pydicom
    table = _series_table(os.path.join(src.path, item["name"]))
    if table.empty:
        raise ValueError(f"[cq500] {item['name']}: no DICOM series")
    bad = "|".join(f"(?:{p})" for p in d["excluded_patterns"])
    ok = table[(table["thickness"] >= float(d["min_thickness_mm"])) & ~(table["kernel"] + " " + table["description"]).str.contains(bad, case=False, regex=True)]
    boxed = ok[ok["series_uid"].isin(_bhx_series(d))]
    pick = (boxed if len(boxed) else ok).sort_values(["n_files", "series_uid"], ascending=[False, True])
    if pick.empty:
        return [{"id": item["id"], "source": item["name"], "file": "", "kind": "no_thick_series"}]
    chosen = pick.iloc[0]
    img, arr, files = read_dicom_series(chosen["folder"], chosen["series_uid"])
    rel = os.path.join(d["volumes_out"], f"{item['name']}.nii.gz")
    row = {"id": item["id"], "source": item["name"], "file": rel, "kind": "volume", "series_uid": chosen["series_uid"],
           "thickness": float(chosen["thickness"]), "n_slices": int(arr.shape[0]), "bhx_series": bool(len(boxed))}
    _write_window(arr, img, d, out, rel, row)
    sops = [str(pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID) for f in files]
    slices = pd.DataFrame({"scan": item["name"], "slice_index": range(len(sops)), "sop_uid": sops, "z": [img.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(len(sops))]})
    write_csv_atomic(slices, os.path.join(out, d["slices_out"], f"{item['name']}.csv"))
    return [row]


def _items_lidc(src: RawSource, d: Dict) -> List[Dict]:
    if src.kind != "dir":
        raise ValueError("LIDC-IDRI is read from the directory the NBIA Data Retriever writes")
    return [{"id": p, "name": p} for p in sorted(os.listdir(src.path)) if re.search(d["patient_pattern"], p)]


def _file_positions(files: List[str]) -> np.ndarray:
    import pydicom
    return np.array([float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in files])


def _file_slice(z_files: np.ndarray, z: float, name: str) -> int:
    k = int(np.argmin(np.abs(z_files - z)))
    if abs(z_files[k] - z) > 0.5:
        raise ValueError(f"[lidc] {name}: slice at z {z} has no file in the series read by SimpleITK")
    return k


def _process_lidc(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    for name, typ in (("bool", bool), ("int", int), ("float", float)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    try:
        import pylidc as pl
        from pylidc.utils import consensus
    except ImportError as e:
        raise RuntimeError("pylidc is missing: pip install pylidc, and point ~/.pylidcrc at the LIDC-IDRI directory") from e
    rows = []
    for scan in pl.query(pl.Scan).filter(pl.Scan.patient_id == item["name"]).all():
        folder = scan.get_path_to_dicom_files()
        img, arr, files = read_dicom_series(folder, scan.series_instance_uid)
        rel = os.path.join(d["volumes_out"], f"{item['name']}__{scan.series_instance_uid}.nii.gz")
        row = {"id": f"{item['id']}__{scan.series_instance_uid}", "source": item["name"], "file": rel, "kind": "volume", "n_slices": int(arr.shape[0])}
        _write_window(arr, img, d, out, rel, row)
        z_files = _file_positions(files)
        labels = np.zeros(arr.shape, dtype=np.uint8)
        nodules = []
        clusters = scan.cluster_annotations()
        marked = sum(1 for anns in clusters if max(float(a.diameter) for a in anns) >= float(d["min_diameter_mm"]))
        for k, anns in enumerate(clusters, start=1):
            diameter = float(np.mean([a.diameter for a in anns]))
            if len(anns) < int(d["min_readers"]) or diameter < float(d["min_diameter_mm"]):
                continue
            cmask, cbbox, _ = consensus(anns, clevel=float(d["consensus_level"]))
            zs = np.asarray(scan.slice_zvals)[cbbox[2].start:cbbox[2].stop]
            for j, z in enumerate(zs):
                labels[_file_slice(z_files, z, item["name"]), cbbox[0], cbbox[1]][cmask[:, :, j]] = len(nodules) + 1
            nodules.append({"scan": rel, "nodule": len(nodules) + 1, "n_readers": len(anns), "diameter_mm": diameter})
        if len(nodules) > 255:
            raise ValueError(f"[lidc] {item['name']}: more than 255 nodules")
        lrel = os.path.join(d["nodules_out"], os.path.basename(rel))
        write_nifti_like(labels, img, os.path.join(out, lrel))
        row.update(nodule_mask=lrel, n_nodules=len(nodules), n_marked_3mm=marked, nodules=json.dumps(nodules))
        rows.append(row)
    return rows


def _items_pmcvqa(src: RawSource, d: Dict) -> List[Dict]:
    t = read_csv_defensively(d["questions_csv"])
    wanted = set(t[d["image_column"]].astype(str))
    return [{"id": n, "name": n} for n in src.names() if os.path.basename(n) in wanted]


def _process_copy(src: RawSource, item: Dict, d: Dict, out: str) -> List[Dict]:
    rel = os.path.join(d["images_out"], os.path.basename(item["name"]))
    _copy_member(src, item["name"], os.path.join(out, rel))
    return [{"id": item["id"], "source": item["name"], "file": rel}]


ADAPTERS: Dict[str, Dict[str, Callable]] = {
    "images": {"items": _items_images, "process": _process_image},
    "mammo_dicom": {"items": _items_dicom, "process": _process_mammo_dicom},
    "volumes": {"items": _items_by_pattern, "process": _process_volume},
    "cq500": {"items": _items_cq500, "process": _process_cq500},
    "lidc": {"items": _items_lidc, "process": _process_lidc},
    "copy_subset": {"items": _items_pmcvqa, "process": _process_copy},
}


def _tree_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1024 ** 3


def main_preprocess_download(cfg_path: str, name: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    d = cfg["downloads"][name]
    out = d["out_dir"]
    if os.path.abspath(out).startswith(os.path.abspath(full["datasets_root"]) + os.sep):
        raise ValueError(f"[P] {name}: out_dir {out} lies under datasets_root, which this project only reads")
    status = status_path(cfg, "P")
    sources = [RawSource(p) for p in (d["raw"] if isinstance(d["raw"], list) else [d["raw"]])]
    adapter = ADAPTERS[d["adapter"]]
    items = [dict(it, part=i) for i, s in enumerate(sources) for it in adapter["items"](s, d)]
    if not items:
        raise ValueError(f"[P] {name}: the adapter found no item in {d['raw']}; check the patterns in config")
    shard = int(cfg["preprocess"]["shard_size"])
    units = [f"{i:05d}" for i in range(0, len(items), shard)]
    os.makedirs(out, exist_ok=True)

    def compute(unit: str) -> List[Dict]:
        from tqdm import tqdm
        rows = []
        for it in tqdm(items[int(unit):int(unit) + shard], desc=f"[P] {name} {unit}", unit="item", leave=False):
            rows += adapter["process"](sources[it["part"]], it, d, out)
        return rows
    params = {"stage": "P", "version": PREPROCESS_VERSION, "download": name,
              "rule": {k: v for k, v in d.items() if k not in ("raw", "out_dir", "expected_gb", "keep_files", "copy_files", "bhx_csv", "questions_csv")},
              "raw": [{"size": os.path.getsize(s.path)} if s.kind != "dir" else {"n_items": len(items)} for s in sources]}
    partial_dir = os.path.join(cfg["preprocess"]["partials_dir"], name)
    run_units_resumable(partial_dir, name, units, compute, progress_desc=f"[P] {name}", use_claims=True, status_file=status, build_params=params)
    index = pd.DataFrame(load_group_rows(partial_dir, name))
    write_csv_atomic(index, os.path.join(out, "preprocess_index.csv"))
    for kept in d.get("keep_files", []):
        matches = [n for n in sources[0].names() if n.endswith(kept)]
        if not matches:
            raise ValueError(f"[P] {name}: the file to keep {kept} is not in the download")
        _copy_member(sources[0], matches[0], os.path.join(out, os.path.basename(kept)))
    for path in d.get("copy_files", []):
        if not os.path.isfile(path):
            raise MissingInput(f"[P] {name}: {path} is not on disk")
        shutil.copyfile(path, os.path.join(out, os.path.basename(path)))
    size = _tree_size_gb(out)
    report = {"download": name, "items": len(items), "rows": len(index), "size_gb": round(size, 3), "expected_gb": d.get("expected_gb"),
              "kinds": index["kind"].value_counts().to_dict() if "kind" in index else {}}
    write_json_atomic(os.path.join(out, "preprocess_report.json"), report)
    append_status(status, f"P {name}: {len(index)} rows, {size:.2f} GB (estimate {d.get('expected_gb')} GB)")
