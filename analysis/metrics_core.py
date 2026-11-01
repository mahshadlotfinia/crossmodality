"""
analysis/metrics_core.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.joint_bootstrap import weighted_auroc, weighted_rate


NON_ANSWER = ("abstain", "truncated", "empty", "unparsed")
P_FALLBACK = ("none_in_top", "no_answer_position")
Series = Tuple[str, str, float, np.ndarray, int, int]


def _complete(arrays: List[Optional[np.ndarray]], mask: np.ndarray) -> bool:
    return all(a is not None and np.isfinite(a[mask]).all() for a in arrays)


def finding_series(arr: Dict, q: pd.DataFrame, counts: np.ndarray, finding: str) -> List[Series]:
    m = (q["finding"] == finding).to_numpy()
    label = q["label"].to_numpy(dtype=float)
    a, p = arr["ans"], arr["p"]
    a0, p0 = a.get("original"), p.get("original")
    out: List[Series] = []
    if not _complete([a0, p0], m):
        return out
    out.append(("auroc", finding, *weighted_auroc(counts, label, p0, m)))
    pos, neg = m & (label == 1), m & (label == 0)
    sens, spec = weighted_rate(counts, (a0 == 1).astype(float), pos), weighted_rate(counts, (a0 == 0).astype(float), neg)
    out.append(("balanced_accuracy", finding, 0.5 * (sens[0] + spec[0]), 0.5 * (sens[1] + spec[1]), sens[2] + spec[2]))
    out.append(("non_answer_rate", finding, *weighted_rate(counts, (a0 == -1).astype(float), m)))
    src = arr["src"].get("original")
    if src is not None:
        out.append(("p_yes_fallback_rate", finding, *weighted_rate(counts, np.isin(src, P_FALLBACK).astype(float), m)))
    return [s + (1,) for s in out]


def macro_series(per_finding: List[Series]) -> List[Series]:
    by_metric: Dict[str, List[Series]] = {}
    for s in per_finding:
        by_metric.setdefault(s[0], []).append(s)
    out = []
    for metric, rows in by_metric.items():
        rows = [r for r in rows if np.isfinite(r[2])]
        if not rows:
            continue
        boot = np.mean(np.vstack([r[3] for r in rows]), axis=0)
        out.append((metric, "macro", float(np.mean([r[2] for r in rows])), boot, int(sum(r[4] for r in rows)), len(rows)))
    return out


def answer_counts(arr: Dict, q: pd.DataFrame) -> List[Dict]:
    rows = []
    for cond, cls in arr["cls"].items():
        src = arr["src"].get(cond)
        for f in sorted(q["finding"].unique()):
            m = (q["finding"] == f).to_numpy() & np.array([c is not None for c in cls])
            row = {"condition": cond, "finding": f, "n": int(m.sum())}
            row.update({f"class_{c}": int((cls[m] == c).sum()) for c in ("yes", "no") + NON_ANSWER})
            if src is not None:
                row.update({f"source_{s}": int((src[m] == s).sum()) for s in ("first_token", "answer_tag") + P_FALLBACK})
            rows.append(row)
    return rows
