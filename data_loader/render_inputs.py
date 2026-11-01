"""
data_loader/render_inputs.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_question_sets import PROJECT, manifest_csv
from data_loader.build_utils import read_csv_defensively, write_csv_if_changed
from data_loader.sets import BENCHMARK, MAIN, benchmark_sets, external_sets, modality_sets, set_config, set_kind
from data_loader.slices import is_slice_path, load_slice
from Inference.resume_utils import (ClaimedElsewhere, MissingInput, append_status, digest_file, load_group_rows, read_build_params, run_units_resumable,
                                    status_path, unit_cache_path, write_build_params)


RENDER_VERSION = 1
_H5: Dict[str, object] = {}


def pixel_digest(arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def load_source(image_path: str, root: str) -> np.ndarray:
    import io
    from PIL import Image
    if is_slice_path(image_path):
        return load_slice(image_path, root)
    if ".parquet#" in image_path:
        from data_loader.benchmarks import parquet_image
        path, row = image_path.rsplit("#", 1)
        with Image.open(io.BytesIO(parquet_image(os.path.join(root, path), int(row)))) as im:
            return np.asarray(im.convert("RGB"))
    if "#" in image_path:
        import h5py
        path, idx = image_path.rsplit("#", 1)
        full = os.path.join(root, path)
        if full not in _H5:
            _H5[full] = h5py.File(full, "r")
        return np.asarray(_H5[full]["x"][int(idx)], dtype=np.uint8)
    with Image.open(os.path.join(root, image_path)) as im:
        return np.asarray(im.convert("RGB"))


def to_input(arr: np.ndarray, mode: str, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(arr)
    if mode == "pad":
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side))
        canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
        img = canvas
    elif mode not in ("direct", "direct_bicubic"):
        raise ValueError(f"unknown render mode {mode!r}")
    resample = Image.Resampling.BICUBIC if mode == "direct_bicubic" else Image.Resampling.LANCZOS
    if img.size != (size, size):
        img = img.resize((size, size), resample)
    return np.asarray(img.convert("RGB"))


def benchmark_input(arr: np.ndarray, max_side: int) -> np.ndarray:
    from PIL import Image
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / float(max(h, w))
    return np.asarray(Image.fromarray(arr).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS))


def downsize(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS))


def save_png(arr: np.ndarray, path: str) -> None:
    import uuid
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.png"
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _alloc(sizes: pd.Series, total: int) -> Dict[str, int]:
    raw = sizes / sizes.sum() * total
    base = np.floor(raw).astype(int)
    order = sorted(sizes.index, key=lambda f: (-(raw[f] - base[f]), f))
    for f in order[: total - int(base.sum())]:
        base[f] += 1
    return {f: int(min(base[f], sizes[f])) for f in sizes.index}


def robustness_subsets(manifest: pd.DataFrame, rcfg: Dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = manifest.sort_values("question_id", kind="mergesort").reset_index(drop=True)
    per_label = int(rcfg["n_per_label"])
    sizes = m[m["label"] == 1].groupby("finding").size()
    alloc = _alloc(sizes, min(per_label, int(sizes.sum())))
    phrasing = []
    for f in sorted(alloc):
        for label in (1, 0):
            pool = m.loc[(m["finding"] == f) & (m["label"] == label), "question_id"].to_numpy()
            phrasing += list(pool[rng.permutation(len(pool))[: alloc[f]]])
    phr = sorted(phrasing)
    det = sorted(np.array(phr)[rng.permutation(len(phr))[: int(rcfg["determinism_questions"])]].tolist())
    out = m[["question_id", "finding", "label"]].copy()
    out["in_phrasing"] = out["question_id"].isin(set(phr))
    out["in_determinism"] = out["question_id"].isin(set(det))
    return out[out[["in_phrasing", "in_determinism"]].any(axis=1)].reset_index(drop=True)


def render_table_csv(cfg: Dict, set_name: str) -> str:
    return os.path.join(cfg["outputs"]["renders_dir"], f"render_table_{set_name}.csv")


def robustness_csv(cfg: Dict, set_name: str) -> str:
    return os.path.join(cfg["outputs"]["renders_dir"], f"robustness_{set_name}.csv")


def renders_current(cfg: Dict, set_name: str) -> Optional[str]:
    if not os.path.exists(manifest_csv(cfg, set_name)):
        return "no main_build_manifests manifest; run main_build_manifests and main_render_inputs"
    table = render_table_csv(cfg, set_name)
    stored = read_build_params(table) if os.path.exists(table) else None
    if stored is None:
        return "no render table; run main_render_inputs"
    if stored.get("manifest") != digest_file(manifest_csv(cfg, set_name)):
        return "the render table was made from an older manifest; run main_render_inputs"
    return None


def _rel(cfg_full: Dict, path: str) -> str:
    return os.path.relpath(path, cfg_full["outputs_root"])


class _Renderer:

    def __init__(self, full: Dict, set_name: str, manifest: pd.DataFrame, robust: Optional[pd.DataFrame]):
        self.full, self.cfg = full, full[PROJECT]
        self.root = full["datasets_root"]
        self.set_name = set_name
        mcfg = set_config(self.cfg, set_name)
        self.size, self.small, self.channels = int(mcfg["input_size"]), int(mcfg["small_size"]), int(mcfg["channels"])
        self.manifest = manifest
        self.mode = {d: self.cfg["render"]["modes"][d] for d in manifest["dataset"].unique()}
        self.dir = os.path.join(self.cfg["outputs"]["renders_dir"], set_name)
        cases = manifest[["case_id", "image_path", "dataset"]]
        self.cases = cases.drop_duplicates("case_id").sort_values("case_id", kind="mergesort").reset_index(drop=True)
        small_ids = set(robust.loc[robust["in_phrasing"], "question_id"]) if robust is not None else set()
        self.small_cases = manifest[manifest["question_id"].isin(small_ids)][["case_id", "image_path", "dataset"]].drop_duplicates("case_id")
        shard = int(self.cfg["render"]["shard_size"])
        self.units = ([f"original__{i:04d}" for i in range(0, len(self.cases), shard)]
                      + [f"small__{i:04d}" for i in range(0, len(self.small_cases), shard)])
        self.shard = shard

    def _render_case(self, case_id: str, image_path: str, dataset: str) -> np.ndarray:
        return to_input(load_source(image_path, self.root), self.mode[dataset], self.size)

    def _row(self, condition: str, ident: str, case_id: str, question_id: str, arr: np.ndarray) -> Dict:
        path = os.path.join(self.dir, condition, f"{ident}.png")
        save_png(arr, path)
        return {"render_key": f"{condition}|{ident}", "condition": condition, "case_id": case_id, "question_id": question_id,
                "path": _rel(self.full, path), "width": int(arr.shape[1]), "height": int(arr.shape[0]), "digest": pixel_digest(arr)}

    def compute(self, unit: str) -> List[Dict]:
        from tqdm import tqdm
        kind, _, start = unit.partition("__")
        rows = []
        if kind == "original":
            part = self.cases.iloc[int(start):int(start) + self.shard]
            for r in tqdm(list(part.itertuples(index=False)), desc=f"[main_render_inputs] {self.set_name} {unit}", unit="image", leave=False):
                rows.append(self._row("original", r.case_id, r.case_id, "", self._render_case(r.case_id, r.image_path, r.dataset)))
        elif kind == "small":
            part = self.small_cases.sort_values("case_id", kind="mergesort").iloc[int(start):int(start) + self.shard]
            for r in tqdm(list(part.itertuples(index=False)), desc=f"[main_render_inputs] {self.set_name} {unit}", unit="image", leave=False):
                rows.append(self._row("original_small", r.case_id, r.case_id, "", downsize(self._render_case(r.case_id, r.image_path, r.dataset), self.small)))
        else:
            raise ValueError(f"unknown render unit {unit!r}")
        return rows


def _render_params(cfg: Dict, set_name: str) -> Dict:
    mcfg = set_config(cfg, set_name)
    modes = {d: cfg["render"]["modes"][d] for d in sorted({f["dataset"] for f in mcfg["findings"].values()})}
    params = {"stage": "main_render_inputs", "version": RENDER_VERSION, "set": set_name, "manifest": digest_file(manifest_csv(cfg, set_name)),
              "render": {"modes": modes, "shard_size": int(cfg["render"]["shard_size"])},
              "sizes": {"input": int(mcfg["input_size"]), "small": int(mcfg["small_size"]), "channels": int(mcfg["channels"])},
              "seed": int(cfg["question_sets"]["seed"])}
    if set_kind(cfg, set_name) == MAIN:
        params["robustness"] = cfg["robustness"]
    return params


class _BenchmarkRenderer:

    def __init__(self, full: Dict, set_name: str, manifest: pd.DataFrame):
        self.full, self.cfg, self.root, self.set_name = full, full[PROJECT], full["datasets_root"], set_name
        self.max_side = int(self.cfg["benchmarks"]["max_side"])
        self.dir = os.path.join(self.cfg["outputs"]["renders_dir"], set_name)
        self.cases = manifest[["case_id", "image_path"]].drop_duplicates("case_id").sort_values("case_id", kind="mergesort").reset_index(drop=True)
        self.shard = int(self.cfg["render"]["shard_size"])
        self.units = [f"original__{i:04d}" for i in range(0, len(self.cases), self.shard)]

    def compute(self, unit: str) -> List[Dict]:
        from tqdm import tqdm
        start = int(unit.partition("__")[2])
        rows = []
        for r in tqdm(list(self.cases.iloc[start:start + self.shard].itertuples(index=False)), desc=f"[main_render_inputs] {self.set_name} {unit}", unit="image", leave=False):
            arr = benchmark_input(load_source(r.image_path, self.root), self.max_side)
            path = os.path.join(self.dir, "original", f"{r.case_id}.png")
            save_png(arr, path)
            rows.append({"render_key": f"original|{r.case_id}", "condition": "original", "case_id": r.case_id, "question_id": "",
                         "path": _rel(self.full, path), "width": int(arr.shape[1]), "height": int(arr.shape[0]), "digest": pixel_digest(arr)})
        return rows


def _benchmark_params(cfg: Dict, set_name: str) -> Dict:
    return {"stage": "main_render_inputs benchmark", "version": RENDER_VERSION, "set": set_name, "manifest": digest_file(manifest_csv(cfg, set_name)),
            "max_side": int(cfg["benchmarks"]["max_side"]), "shard_size": int(cfg["render"]["shard_size"])}


def _drop_units_with_missing_files(full: Dict, partial_dir: str, set_name: str, units: List[str]) -> int:
    from Inference.resume_utils import read_jsonl
    dropped = 0
    for u in units:
        cache = unit_cache_path(partial_dir, set_name, u)
        if os.path.exists(cache) and any(not os.path.exists(os.path.join(full["outputs_root"], r["path"])) for r in read_jsonl(cache)):
            os.remove(cache)
            dropped += 1
    return dropped


def main_render_inputs(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    status = status_path(cfg, "main_render_inputs")
    skipped, held = [], []

    def finish(set_name, renderer, params, partial_dir, wait):
        rows = run_units_resumable(partial_dir, set_name, renderer.units, renderer.compute, progress_desc=f"[main_render_inputs] {set_name}",
                                   require_rows=True, use_claims=True, status_file=status, build_params=params, wait=wait)
        if rows.attrs.get("n_skipped"):
            skipped.append(f"{set_name} ({rows.attrs['n_skipped']} unit(s) could not run)")
            return
        table = pd.DataFrame(load_group_rows(partial_dir, set_name)).sort_values("render_key", kind="mergesort")
        if table["render_key"].duplicated().any():
            raise ValueError(f"[main_render_inputs] {set_name}: duplicated render keys")
        write_csv_if_changed(table, render_table_csv(cfg, set_name))
        if read_build_params(render_table_csv(cfg, set_name)) != params:
            write_build_params(render_table_csv(cfg, set_name), params)
        append_status(status, f"main_render_inputs {set_name}: {len(table)} renders")
    for set_name in modality_sets(cfg) + external_sets(cfg) + benchmark_sets(cfg):
        kind = set_kind(cfg, set_name)
        path = manifest_csv(cfg, set_name)
        if not (os.path.exists(path) and read_build_params(path) is not None):
            skipped.append(f"{set_name} (no manifest; run {'main_build_manifests' if kind == MAIN else 'main_build_external_sets'})")
            continue
        manifest = read_csv_defensively(path, dtype={"key": str, "group_id": str, "case_id": str}, keep_default_na=False)
        if kind == BENCHMARK:
            renderer, params = _BenchmarkRenderer(full, set_name, manifest), _benchmark_params(cfg, set_name)
        else:
            robust = None
            if kind == MAIN:
                robust = robustness_subsets(manifest, cfg["robustness"], int(cfg["question_sets"]["seed"]))
                write_csv_if_changed(robust, robustness_csv(cfg, set_name))
            renderer, params = _Renderer(full, set_name, manifest, robust), _render_params(cfg, set_name)
        partial_dir = os.path.join(cfg["outputs"]["renders_dir"], "partials", set_name)
        _drop_units_with_missing_files(full, partial_dir, set_name, renderer.units)
        try:
            finish(set_name, renderer, params, partial_dir, False)
        except ClaimedElsewhere:
            held.append((set_name, renderer, params, partial_dir))
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    for set_name, renderer, params, partial_dir in held:
        try:
            finish(set_name, renderer, params, partial_dir, True)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    if skipped:
        raise MissingInput(f"[main_render_inputs] {len(skipped)} set(s) not rendered: {'; '.join(skipped)}")
