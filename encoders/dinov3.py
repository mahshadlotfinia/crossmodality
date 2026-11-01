"""
encoders/dinov3.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_question_sets import PROJECT, SOURCES, absent_inputs, partition_csv, questions_csv, read_partition, run_build_unit, run_build_units
from data_loader.build_utils import cap_keys, capped_count, draw_capped, read_csv_defensively
from data_loader.render_inputs import load_source, to_input
from Inference.resume_utils import (MissingInput, append_status, claim_or_wait, claim_unit, digest_file, heartbeat_claim,
                                    release_claim, status_path, write_csv_atomic, write_npz_atomic)
from Inference.serving_utils import require_packages, resolve_secret


DEV_VERSION = 1
FEATURES_VERSION = 1
DEV_COLUMNS = ["case_id", "group_id", "image_path", "dataset", "finding", "label"]


def dinov3_dir(cfg: Dict) -> str:
    return cfg["outputs"]["dinov3_dir"]


def dev_csv(cfg: Dict, modality: str) -> str:
    return os.path.join(dinov3_dir(cfg), f"dev_{modality}.csv")


def features_npz(cfg: Dict, modality: str) -> str:
    return os.path.join(dinov3_dir(cfg), "features", f"{modality}.npz")


def _shard_npz(cfg: Dict, modality: str, index: int) -> str:
    return os.path.join(dinov3_dir(cfg), "features", modality, f"shard_{index:05d}.npz")


def _digest(values) -> str:
    h = hashlib.blake2b(digest_size=16)
    for v in values:
        h.update(str(v).encode("utf-8") + b"\0")
    return h.hexdigest()


def dev_sample(cfg: Dict, modality: str) -> Tuple[pd.DataFrame, List[str]]:
    dino_cfg = cfg["dinov3"]
    rows, missing = [], []
    for finding, fcfg in cfg["question_sets"]["modalities"][modality]["findings"].items():
        dataset = fcfg.get("dev_dataset", fcfg["dataset"])
        cap = int(fcfg.get("dev_group_cap", dino_cfg["dev_group_cap"]))
        try:
            part = read_partition(cfg, dataset)
        except MissingInput as e:
            missing.append(f"{finding} ({e})")
            continue
        dev = part[part["partition"] == "dev"]
        if dev.empty:
            missing.append(f"{finding} ({dataset} has no development partition)")
            continue
        src = SOURCES[dataset]
        dcfg = cfg["datasets"][dataset]
        pools = src.dev_finding_pools(fcfg, dev, dcfg) if hasattr(src, "dev_finding_pools") else src.finding_pools(fcfg, dev, dcfg)
        rng = np.random.default_rng(int(dino_cfg["seed"]))
        for label, pool in zip((1, 0), pools):
            pool = pool.sort_values("case_id", kind="mergesort").reset_index(drop=True)
            n = min(int(dino_cfg["dev_per_class"]), capped_count(cap_keys(pool), cap))
            if n:
                rows.append(draw_capped(pool, n, cap, rng)[["case_id", "group_id", "image_path"]].assign(dataset=dataset, finding=finding, label=label))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=DEV_COLUMNS)
    return table[DEV_COLUMNS], missing


def _dev_params(cfg: Dict, modality: str) -> Dict:
    mcfg = cfg["question_sets"]["modalities"][modality]
    datasets = sorted({d for f in mcfg["findings"].values() for d in (f["dataset"], f.get("dev_dataset", f["dataset"]))})
    dino_cfg = cfg["dinov3"]
    return {"stage": "main_dinov3_cache dev", "version": DEV_VERSION, "findings": mcfg["findings"],
            "sampling": {"dev_per_class": int(dino_cfg["dev_per_class"]), "dev_group_cap": int(dino_cfg["dev_group_cap"]), "seed": int(dino_cfg["seed"])},
            "partitions": {d: digest_file(partition_csv(cfg, d)) for d in datasets},
            "sources": {d: SOURCES[d].source_signature(cfg["datasets"][d]) for d in datasets}}


def image_list(cfg: Dict, modality: str) -> pd.DataFrame:
    dev = read_csv_defensively(dev_csv(cfg, modality), dtype={"case_id": str, "group_id": str}, keep_default_na=False)
    q = read_csv_defensively(questions_csv(cfg, modality), usecols=["case_id", "image_path", "dataset"], dtype=str, keep_default_na=False)
    both = pd.concat([dev[["case_id", "image_path", "dataset"]], q], ignore_index=True)
    return both.drop_duplicates("case_id").sort_values("case_id", kind="mergesort").reset_index(drop=True)


def _hf_load(cfg: Dict, loader: Callable, hf_id: str, **kwargs):
    from huggingface_hub.errors import GatedRepoError
    try:
        return loader(hf_id, local_files_only=True, **kwargs)
    except OSError:
        pass
    try:
        return loader(hf_id, token=resolve_secret(cfg, "hf_token"), **kwargs)
    except OSError as e:
        cause = e
        while cause is not None and not isinstance(cause, GatedRepoError):
            cause = cause.__cause__
        if cause is None:
            raise
        raise MissingInput(f"[main_dinov3_cache] {hf_id} is gated and the token of this machine has no access to it; accept its terms on Hugging Face with "
                           f"the account of hf_token (or of Crossmodality.hf_token_env) and run it again") from e


class Dinov3Encoder:

    def __init__(self, cfg: Dict):
        require_packages(["torch", "transformers"], "main_dinov3_cache DINOv3-L")
        import torch
        from transformers import AutoImageProcessor, AutoModel
        dino_cfg = cfg["dinov3"]
        self.hf_id, self.width, self.size = dino_cfg["hf_id"], int(dino_cfg["width"]), int(dino_cfg["image_size"])
        self.device = dino_cfg["device"] if torch.cuda.is_available() else "cpu"
        self.processor = _hf_load(cfg, AutoImageProcessor.from_pretrained, self.hf_id)
        model, info = _hf_load(cfg, AutoModel.from_pretrained, self.hf_id, output_loading_info=True)
        if info.get("missing_keys"):
            raise RuntimeError(f"[main_dinov3_cache] {self.hf_id} loaded with {len(info['missing_keys'])} missing weights (first: {info['missing_keys'][0]}); refusing random weights")
        self.model = model.to(self.device, dtype=torch.bfloat16).eval()

    def prepare(self, images: List[np.ndarray]):
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        if tuple(pixels.shape[1:]) != (3, self.size, self.size):
            raise ValueError(f"[main_dinov3_cache] the processor gave {tuple(pixels.shape)}, not (n, 3, {self.size}, {self.size})")
        return pixels

    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self.model(pixel_values=self.prepare(images).to(self.device, dtype=torch.bfloat16)).last_hidden_state[:, 0, :]
        feats = out.float().cpu().numpy()
        if feats.shape[1] != self.width:
            raise ValueError(f"[main_dinov3_cache] {self.hf_id} gives width {feats.shape[1]}, not the configured {self.width}")
        if not np.isfinite(feats).all():
            raise ValueError("[main_dinov3_cache] DINOv3-L gave non-finite features in bf16")
        return feats


def load_encoder(cfg: Dict) -> Dinov3Encoder:
    return Dinov3Encoder(cfg)


def _shard_current(path: str, ids: List[str], params: Dict, width: int) -> Optional[str]:
    if not os.path.exists(path):
        return "absent"
    try:
        with np.load(path, allow_pickle=False) as z:
            if [str(x) for x in z["case_ids"]] != ids:
                return "other case ids"
            if z["embeddings"].shape != (len(ids), width) or not np.isfinite(z["embeddings"]).all():
                return "wrong width or non-finite features"
            stored = {k[4:]: str(z[k]) for k in z.files if k.startswith("_bp_")}
    except (OSError, ValueError, KeyError) as e:
        return f"unreadable ({e})"
    want = {k: str(v) for k, v in params.items()}
    moved = [k for k in want if stored.get(k) not in (None, want[k])]
    return f"built under other parameters: {moved}" if moved else None


def _features_params(cfg: Dict, modality: str, images: pd.DataFrame) -> Dict:
    dino_cfg = cfg["dinov3"]
    mcfg = cfg["question_sets"]["modalities"][modality]
    return {"version": FEATURES_VERSION, "hf_id": dino_cfg["hf_id"], "precision": "bf16", "image_size": int(dino_cfg["image_size"]), "input_size": int(mcfg["input_size"]),
            "modes": _digest(sorted(cfg["render"]["modes"].items())), "images": _digest(zip(images["case_id"], images["image_path"]))}


def _encode_shard(cfg: Dict, root: str, modality: str, part: pd.DataFrame, encoder, beat: Callable[[], None]) -> np.ndarray:
    from tqdm import tqdm
    dino_cfg = cfg["dinov3"]
    size = int(cfg["question_sets"]["modalities"][modality]["input_size"])
    modes = cfg["render"]["modes"]

    def _prepare(r):
        return to_input(load_source(r.image_path, root), modes[r.dataset], size)

    rows = list(part.itertuples(index=False))
    batch = int(dino_cfg["batch_size"])
    out = np.zeros((len(rows), int(dino_cfg["width"])), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=int(dino_cfg["num_workers"])) as pool:
        for start in tqdm(range(0, len(rows), batch), desc=f"[main_dinov3_cache] {modality} batches", unit="batch", leave=False):
            images = list(pool.map(_prepare, rows[start:start + batch]))
            out[start:start + len(images)] = encoder.encode(images)
            beat()
    return out


def _purge(cfg: Dict) -> None:
    dino_cfg = cfg["dinov3"]
    if not dino_cfg["purge_hf_cache_after_use"]:
        return
    folder = os.path.join(dino_cfg["hf_cache_dir"], "models--" + dino_cfg["hf_id"].replace("/", "--"))
    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def main_dinov3_cache(cfg_path: str) -> None:
    require_packages(["torch", "transformers"], "main_dinov3_cache")
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    dino_cfg = cfg["dinov3"]
    status = status_path(cfg, "main_dinov3_cache")
    skipped, encoder, todo, attempts, ready = [], None, [], [], []
    for modality, mcfg in cfg["question_sets"]["modalities"].items():
        absent = sorted({p for f in mcfg["findings"].values() for p in absent_inputs(cfg["datasets"][f["dataset"]])})
        if absent or not os.path.exists(questions_csv(cfg, modality)):
            skipped.append(f"{modality} ({'lacks ' + ', '.join(absent[:2]) if absent else 'no question set; run main_build_question_sets'})")
            continue

        def build(modality=modality):
            table, missing = dev_sample(cfg, modality)
            write_csv_atomic(table, dev_csv(cfg, modality))
        attempts.append((f"main_dinov3_cache_dev__{modality}", lambda wait, modality=modality, build=build:
                         run_build_unit(cfg, f"main_dinov3_cache_dev__{modality}", dev_csv(cfg, modality), _dev_params(cfg, modality), lambda: True, build, wait=wait)))
        ready.append(modality)
    run_build_units(attempts, "[main_dinov3_cache] development samples")
    for modality in ready:
        images = image_list(cfg, modality)
        params = _features_params(cfg, modality, images)
        if _shard_current(features_npz(cfg, modality), images["case_id"].tolist(), params, int(dino_cfg["width"])) is None:
            continue
        todo.append((modality, images, params))
    def encode(modality, parts, claims, params, index, reason):
        nonlocal encoder
        part = parts[index]
        path, ids = _shard_npz(cfg, modality, index), part["case_id"].tolist()
        try:
            if encoder is None:
                encoder = load_encoder(cfg)
            feats = _encode_shard(cfg, root, modality, part, encoder, lambda: heartbeat_claim(claims, f"shard_{index:05d}"))
            write_npz_atomic(path, embeddings=feats, case_ids=np.array(ids), **{f"_bp_{k}": np.array(str(v)) for k, v in params.items()})
        finally:
            release_claim(claims, f"shard_{index:05d}")
        append_status(status, f"main_dinov3_cache {modality} shard {index}: {len(ids)} images")

    work = []
    for modality, images, params in todo:
        shard = int(dino_cfg["shard_size"])
        parts = [images.iloc[i:i + shard] for i in range(0, len(images), shard)]
        claims = os.path.join(dinov3_dir(cfg), "features", modality)
        os.makedirs(claims, exist_ok=True)
        held = []
        for index, part in enumerate(parts):
            reason = _shard_current(_shard_npz(cfg, modality, index), part["case_id"].tolist(), params, int(dino_cfg["width"]))
            if reason is None:
                continue
            if claim_unit(claims, f"shard_{index:05d}", 1800.0):
                encode(modality, parts, claims, params, index, reason)
            else:
                held.append(index)
        work.append((modality, images, params, parts, claims, held))
    for modality, images, params, parts, claims, held in work:
        for index in held:
            path, ids = _shard_npz(cfg, modality, index), parts[index]["case_id"].tolist()
            if claim_or_wait(claims, f"shard_{index:05d}", lambda path=path, ids=ids, params=params: _shard_current(path, ids, params, int(dino_cfg["width"])) is None):
                encode(modality, parts, claims, params, index, _shard_current(path, ids, params, int(dino_cfg["width"])))
        blocks = []
        for index in range(len(parts)):
            with np.load(_shard_npz(cfg, modality, index)) as z:
                blocks.append(z["embeddings"])
        feats = np.concatenate(blocks)
        write_npz_atomic(features_npz(cfg, modality), embeddings=feats, case_ids=np.array(images["case_id"].tolist()),
                         **{f"_bp_{k}": np.array(str(v)) for k, v in params.items()})
        append_status(status, f"main_dinov3_cache {modality}: features merged")
    if encoder is not None:
        del encoder
        _purge(cfg)
    if skipped:
        raise MissingInput(f"[main_dinov3_cache] {len(skipped)} modality(ies) without features: {'; '.join(skipped)}")
