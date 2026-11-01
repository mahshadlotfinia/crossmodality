"""
analysis/joint_bootstrap.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from Inference.stats_utils import BOOT_SEED, N_BOOT, auroc, cluster_draws, cluster_layout, percentile_summary, resample_hash


def replicate_counts(groups: Sequence, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Tuple[np.ndarray, str]:
    layout = cluster_layout(groups)
    k = len(layout["sizes"])
    draws = cluster_draws(k, n_boot, seed)
    per_cluster = np.zeros((n_boot, k), dtype=np.float64)
    for b in range(n_boot):
        per_cluster[b] = np.bincount(draws[b], minlength=k)
    return per_cluster[:, layout["cluster_of_row"]], resample_hash(n_boot, seed, groups)


def weighted_rate(counts: np.ndarray, values: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(values, dtype=float))
    n = int(mask.sum())
    if n == 0:
        return float("nan"), np.full(counts.shape[0], np.nan), 0
    v = np.where(mask, np.asarray(values, dtype=float), 0.0)
    num, den = counts @ v, counts @ mask.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(den > 0, num / den, np.nan)
    return float(v.sum() / n), boot, n


def weighted_auroc(counts: np.ndarray, labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray, int]:
    keep = np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(scores, dtype=float)) & np.isfinite(np.asarray(labels, dtype=float))
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return float("nan"), np.full(counts.shape[0], np.nan), 0
    y, s = np.asarray(labels, dtype=float)[idx], np.asarray(scores, dtype=float)[idx]
    point = auroc(y, s)
    order = np.argsort(s, kind="mergesort")
    s_sorted, pos = s[order], (y[order] == 1).astype(float)
    w = counts[:, idx][:, order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    tied = np.add.reduceat(w, starts, axis=1)
    cum = np.cumsum(tied, axis=1)
    mid = cum - tied + (tied + 1.0) / 2.0
    group_of = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, s_sorted.size]))
    n_pos = w @ pos
    n_neg = w.sum(axis=1) - n_pos
    rank_sum = (w * pos * mid[:, group_of]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    boot[(n_pos == 0) | (n_neg == 0)] = np.nan
    return point, boot, int(idx.size)


def summarize(boot: np.ndarray) -> Dict[str, float]:
    b = np.asarray(boot, dtype=float)
    finite = b[np.isfinite(b)]
    sd, lo, hi = percentile_summary(finite)
    return {"mean": float(finite.mean()) if finite.size else float("nan"), "sd": sd, "lo": lo, "hi": hi, "n_finite": int(finite.size)}
