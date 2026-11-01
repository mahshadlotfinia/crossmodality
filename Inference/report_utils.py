"""
Inference/report_utils.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.stats_utils import bh_fdr, check_interval


UNITS = ("percent", "percentage_points")
BOUND = {"percent": 1.0, "percentage_points": 2.0}


def _finite(x) -> bool:
    return x is not None and np.isfinite(float(x))


def fmt_pct(x, bound: float = 1.0) -> float:
    if not _finite(x):
        return float("nan")
    x = float(x)
    if abs(x) > bound + 1e-9:
        raise ValueError(f"fmt_pct got {x}, outside [-{bound:g}, {bound:g}]: a caller already scaled it to percent.")
    return float(Decimal(repr(round(x * 100.0, 8))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def keep_p(p) -> float:
    return float(p) if _finite(p) else float("nan")


def metric_row(point, sd, lo, hi, n: int, unit: str, what: str, draws: bool = False) -> Dict:
    if unit not in UNITS:
        raise ValueError(f"{what}: unit {unit!r} is not one of {UNITS}")
    if not draws:
        check_interval(point, lo, hi, what)
    elif _finite(lo) and _finite(hi) and lo > hi:
        raise ValueError(f"{what}: the percentiles over draws are in the wrong order ({lo} above {hi}).")
    raw = {k: (float(v) if _finite(v) else float("nan")) for k, v in (("mean_raw", point), ("std_raw", sd), ("ci_low_raw", lo), ("ci_high_raw", hi))}
    b = BOUND[unit]
    return {"mean_pct": fmt_pct(point, b), "std_pct": fmt_pct(sd, b), "ci_low_pct": fmt_pct(lo, b), "ci_high_pct": fmt_pct(hi, b),
            **raw, "n": int(n), "unit": unit}


def stat_row(value, p, n: int) -> Dict:
    return {"value": float(value) if _finite(value) else float("nan"), "p_raw": keep_p(p), "n": int(n)}


def add_fdr(df: pd.DataFrame, family_cols: Optional[List[str]] = None, p_col: str = "p_raw", out_col: str = "p_fdr") -> pd.DataFrame:
    df = df.copy()
    if p_col not in df.columns:
        return df
    df[out_col] = np.nan
    if family_cols:
        for _, idx in df.groupby(family_cols, dropna=False).groups.items():
            df.loc[idx, out_col] = bh_fdr(df.loc[idx, p_col].to_numpy(dtype=float))
    else:
        df[out_col] = bh_fdr(df[p_col].to_numpy(dtype=float))
    df["significant_fdr05"] = df[out_col].apply(lambda q: bool(q < 0.05) if pd.notna(q) else False)
    return df
