"""
Inference/stats_utils.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_BOOT = 1000
BOOT_SEED = 0


def _is_missing(g) -> bool:
    if g is None:
        return True
    if isinstance(g, (float, np.floating)) and not np.isfinite(g):
        return True
    return str(g).strip().lower() in ("", "nan", "none")


def cluster_layout(groups: Sequence) -> Dict[str, np.ndarray]:
    groups = list(groups)
    missing = sum(1 for g in groups if _is_missing(g))
    if missing:
        raise ValueError(f"{missing} of {len(groups)} rows have no group id; the cluster bootstrap needs one for every row.")
    keys = np.asarray([str(g) for g in groups], dtype=object)
    uniq, cluster_of_row = np.unique(keys, return_inverse=True)
    cluster_of_row = np.asarray(cluster_of_row, dtype=np.int64).ravel()
    order = np.argsort(cluster_of_row, kind="stable")
    sizes = np.bincount(cluster_of_row, minlength=len(uniq)).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    return {"flat": order.astype(np.int64), "sizes": sizes, "starts": starts, "cluster_of_row": cluster_of_row}


def cluster_draws(n_clusters: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_clusters, size=(n_boot, n_clusters))


def stratified_group_folds(y: Sequence, groups: Sequence, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    perm = np.arange(uniq.size)
    np.random.RandomState(int(seed)).shuffle(perm)
    rank = np.empty_like(perm)
    rank[perm] = np.arange(perm.size)
    return list(StratifiedGroupKFold(n_splits=int(n_splits)).split(np.zeros(len(inv)), np.asarray(y), rank[np.ravel(inv)]))


def midrank_rows(mat: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(np.asarray(mat, dtype=float), method="average", axis=-1)


def auroc_batch(labels: np.ndarray, scores: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    yt = np.atleast_2d(np.asarray(labels, dtype=float))
    ys = np.atleast_2d(np.asarray(scores, dtype=float))
    n = yt.shape[1]
    if valid is not None:
        valid = np.atleast_2d(np.asarray(valid, dtype=bool))
        ys = np.where(valid, ys, -np.inf)
        yt = np.where(valid, yt, 0.0)
        n_valid = valid.sum(axis=1)
    else:
        n_valid = np.full(yt.shape[0], n)
    n_invalid = n - n_valid
    ranks = midrank_rows(ys)
    n_pos = yt.sum(axis=1)
    n_neg = n_valid - n_pos
    sum_pos = (ranks * yt).sum(axis=1) - n_pos * n_invalid
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc = np.asarray(auc, dtype=float)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    if keep.sum() == 0:
        return float("nan")
    return float(auroc_batch(y[keep], s[keep])[0])


def percentile_summary(boot: np.ndarray) -> Tuple[float, float, float]:
    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    sd = float(np.std(b, ddof=1)) if b.size > 1 else float("nan")
    return (sd, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def shift_reflect_p(boot: np.ndarray, point: float) -> float:
    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0 or not np.isfinite(point):
        return float("nan")
    p = float(np.mean(np.abs(b - point) >= abs(point)))
    return max(p, 1.0 / b.size)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = p.copy()
    valid = np.where(np.isfinite(p))[0]
    m = len(valid)
    if m == 0:
        return out
    order = np.argsort(p[valid], kind="stable")
    adj = p[valid][order] * m / (np.arange(m, dtype=float) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    back = np.empty(m, dtype=float)
    back[order] = adj
    out[valid] = back
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    from scipy.stats import spearmanr
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    n = int(keep.sum())
    if n < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return (float("nan"), float("nan"), n)
    rho, p = spearmanr(a[keep], b[keep])
    return (float(rho), float(p), n)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import kendalltau
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return float("nan")
    return float(kendalltau(a[keep], b[keep], variant="b").statistic)


def resample_hash(n_boot: int, seed: int, groups: Sequence) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{int(n_boot)}|{int(seed)}|".encode("ascii"))
    for g in groups:
        h.update(str(g).encode("utf-8", "replace") + b"\x1f")
    return h.hexdigest()


def check_interval(point: float, lo: float, hi: float, what: str) -> None:
    vals = (point, lo, hi)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return
    if not (lo - 1e-12 <= point <= hi + 1e-12):
        raise ValueError(f"{what}: the interval [{lo}, {hi}] does not contain its point estimate {point}.")
