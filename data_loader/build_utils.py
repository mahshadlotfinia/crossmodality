"""
data_loader/build_utils.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Inference.resume_utils import digest_file


PARTITION_COLUMNS = ["case_id", "dataset", "key", "group_id", "split", "partition", "image_path"]


def read_csv_defensively(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


def make_case_id(dataset: str, key) -> str:
    return f"{dataset}__{str(key).replace(os.sep, '_').replace(' ', '_')}"


def rel_to_root(path: str, root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        raise ValueError(f"{path} is not under the datasets root {root}")
    return rel


def file_signature(path: str, digest_max_bytes: int = 64 * 1024 ** 2) -> Dict:
    if not os.path.isfile(path):
        return {"size": None, "digest": None}
    size = int(os.path.getsize(path))
    return {"size": size} if size > digest_max_bytes else {"size": size, "digest": digest_file(path)}


_WRITE_RECORDERS: List[List[str]] = []


@contextmanager
def recording_writes():
    paths: List[str] = []
    _WRITE_RECORDERS.append(paths)
    try:
        yield paths
    finally:
        _WRITE_RECORDERS.remove(paths)


def write_csv_if_changed(df: pd.DataFrame, path: str) -> bool:
    from Inference.resume_utils import write_csv_atomic
    for paths in _WRITE_RECORDERS:
        paths.append(path)
    text = df.to_csv(index=False)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return False
    write_csv_atomic(df, path)
    return True


def tree_signature(path: str) -> Dict:
    if not os.path.isdir(path):
        return {"n_files": None, "names": None}
    names = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        rel = os.path.relpath(root, path)
        names.extend(os.path.normpath(os.path.join(rel, f)) for f in sorted(files))
    return {"n_files": len(names), "names": hashlib.blake2b("\n".join(names).encode("utf-8"), digest_size=16).hexdigest()}


def eval_half(groups, seed: int) -> set:
    uniq = np.array(sorted({str(g) for g in groups}))
    np.random.default_rng(seed).shuffle(uniq)
    return set(uniq[: len(uniq) // 2].tolist())


def check_partition_table(df: pd.DataFrame, name: str, partitions: Tuple[str, ...]) -> None:
    missing = [c for c in PARTITION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] partition table lacks {missing}")
    if df["case_id"].duplicated().any():
        raise ValueError(f"[{name}] {int(df['case_id'].duplicated().sum())} duplicated case ids, first {df.loc[df['case_id'].duplicated(), 'case_id'].iloc[0]}")
    blank = df["group_id"].isna() | (df["group_id"].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(f"[{name}] {int(blank.sum())} rows without a group id; the group is the resampling unit and may not be missing")
    spans = df.groupby("group_id")["partition"].nunique()
    if (spans > 1).any():
        raise ValueError(f"[{name}] {int((spans > 1).sum())} groups in more than one partition, first {spans[spans > 1].index[0]}")
    got = set(df["partition"])
    if got != set(partitions):
        raise ValueError(f"[{name}] partitions {sorted(got)}, expected {sorted(partitions)}")


def cap_keys(frame: pd.DataFrame) -> np.ndarray:
    return frame["cap_id" if "cap_id" in frame.columns else "group_id"].to_numpy()


def capped_count(groups: np.ndarray, cap: int) -> int:
    if len(groups) == 0:
        return 0
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    return int(np.minimum(counts, cap).sum())


def draw_capped(frame: pd.DataFrame, n: int, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = cap_keys(frame)
    taken, used = [], {}
    for i in rng.permutation(len(frame)):
        g = groups[i]
        if used.get(g, 0) >= cap:
            continue
        used[g] = used.get(g, 0) + 1
        taken.append(int(i))
        if len(taken) == n:
            break
    if len(taken) < n:
        raise ValueError(f"only {len(taken)} rows are available within {cap} per group, and {n} were asked for")
    return frame.iloc[sorted(taken)]


def _n_questions(pos: pd.DataFrame, neg: pd.DataFrame, n_max: Optional[int], cap: int, pos_cap: Optional[int] = None) -> int:
    n = min(capped_count(cap_keys(pos), cap if pos_cap is None else pos_cap), capped_count(cap_keys(neg), cap))
    return n if n_max is None else min(int(n_max), n)


def sample_finding(pos: pd.DataFrame, neg: pd.DataFrame, n_max: Optional[int], cap: int, seed: int,
                   pos_cap: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos = pos.sort_values("case_id", kind="mergesort").reset_index(drop=True)
    neg = neg.sort_values("case_id", kind="mergesort").reset_index(drop=True)
    n = _n_questions(pos, neg, n_max, cap, pos_cap)
    if n <= 0:
        raise ValueError(f"{len(pos)} eligible positives and {len(neg)} eligible negatives leave no question to ask")
    rng = np.random.default_rng(seed)
    drawn_pos = draw_capped(pos, n, cap if pos_cap is None else pos_cap, rng)
    parts = []
    for split, k in sorted(drawn_pos["split"].value_counts().items()):
        parts.append(draw_capped(neg[neg["split"] == split], int(k), cap, rng))
    return drawn_pos.reset_index(drop=True), pd.concat(parts, ignore_index=True)


def sample_matched(pos: pd.DataFrame, neg: pd.DataFrame, n_max: Optional[int], cap: int, seed: int, match_decile: bool,
                   exclude_positive_groups: bool, pos_cap: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    pos = pos.sort_values("case_id", kind="mergesort").reset_index(drop=True)
    neg = neg.sort_values("case_id", kind="mergesort").reset_index(drop=True)
    n = _n_questions(pos, neg, n_max, cap, pos_cap)

    def draw(k):
        rng = np.random.default_rng(seed)
        drawn = draw_capped(pos, k, cap if pos_cap is None else pos_cap, rng)
        return rng, drawn, neg[~neg["group_id"].isin(set(drawn["group_id"]))] if exclude_positive_groups else neg

    rng, drawn_pos, pool = draw(max(n, 0))
    while n > 0 and capped_count(cap_keys(pool), cap) < n:
        n -= 1
        rng, drawn_pos, pool = draw(n)
    if n <= 0:
        raise ValueError(f"{len(pos)} eligible positives and {len(neg)} eligible negatives leave no question to ask")
    if not match_decile:
        return drawn_pos.reset_index(drop=True), draw_capped(pool.reset_index(drop=True), n, cap, rng).reset_index(drop=True), 0
    pool = pool.reset_index(drop=True)
    groups, deciles = cap_keys(pool), pool["decile"].to_numpy().astype(int)
    order = {d: list(np.flatnonzero(deciles == d)[rng.permutation(int((deciles == d).sum()))]) for d in range(10)}
    used: Dict[str, int] = {}
    taken: Dict[int, bool] = {}
    for d, k in sorted(drawn_pos["decile"].astype(int).value_counts().items()):
        for near in sorted(range(10), key=lambda t: (abs(t - d), t)):
            while k and order[near]:
                i = order[near].pop(0)
                if used.get(groups[i], 0) >= cap:
                    continue
                used[groups[i]] = used.get(groups[i], 0) + 1
                taken[i] = near != d
                k -= 1
            if not k:
                break
        if k:
            raise ValueError(f"too few negatives within {cap} per group to match decile {d}")
    rows = sorted(taken)
    drawn_neg = pool.iloc[rows].reset_index(drop=True).assign(neighbor_decile=[taken[i] for i in rows])
    return drawn_pos.reset_index(drop=True).assign(neighbor_decile=False), drawn_neg, int(sum(taken.values()))


def sample_stratified(pos: pd.DataFrame, neg: pd.DataFrame, n_max: Optional[int], cap: int, seed: int,
                      pos_cap: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_parts, neg_parts = [], []
    for stratum in sorted(set(pos["stratum"]) | set(neg["stratum"])):
        p, q = pos[pos["stratum"] == stratum], neg[neg["stratum"] == stratum]
        if _n_questions(p, q, n_max, cap, pos_cap) <= 0:
            raise ValueError(f"stratum {stratum}: {len(p)} eligible positives and {len(q)} eligible negatives leave no question to ask")
        a, b = sample_finding(p, q, n_max, cap, seed, pos_cap)
        pos_parts.append(a)
        neg_parts.append(b)
    return pd.concat(pos_parts, ignore_index=True), pd.concat(neg_parts, ignore_index=True)


def mask_is_empty(path: str) -> bool:
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("L").getbbox() is None


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(img.resize((int(width), int(height)), Image.Resampling.BOX)) > 0
