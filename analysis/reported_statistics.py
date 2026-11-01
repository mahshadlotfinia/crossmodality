"""
analysis/reported_statistics.py
Created on October 30, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analysis.compute_metrics import metrics_csv
from analysis.contamination import marked_findings, marked_pairs, training_table
from analysis.contrasts import finding_values, macro_difference
from analysis.joint_bootstrap import summarize
from analysis.selection import panel_scores, run_draws
from config.serde import read_config
from data_loader.build_question_sets import manifest_csv
from data_loader.build_utils import write_csv_if_changed
from data_loader.sets import external_sets, modality_sets
from Inference.model_facts import facts_csv

PROJECT = "Crossmodality"
NEWER_SMALL, OLDER_LARGE = "Qwen3.5-4B", "Qwen3-VL-32B"
SMALL_PARAMETERS = 4e9


def _out(cfg: Dict, name: str) -> str:
    return os.path.join(os.path.dirname(cfg["outputs"]["contrasts_dir"]), "reported_statistics", f"{name}.csv")


def _read(cfg: Dict, key: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(cfg["outputs"][key], f"{name}.csv"), low_memory=False)


def macro_auroc(cfg: Dict, set_name: str) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv(cfg, set_name, "metrics"), low_memory=False)
    return m[(m["metric"] == "auroc") & (m["scope"] == "macro") & (m["model"].isin(cfg["models"]))]


def paired_differences(cfg: Dict) -> pd.DataFrame:
    rows = []
    reference = cfg["dinov3"]["model_key"]
    for s in modality_sets(cfg):
        hashes: Dict[str, str] = {}
        leader = macro_auroc(cfg, s).sort_values("mean_raw", ascending=False).iloc[0]["model"]
        for kind, a, b in (("leader_minus_reference", reference, leader), ("newer_small_minus_older_large", OLDER_LARGE, NEWER_SMALL)):
            point, boot, n, nf = macro_difference(finding_values(cfg, s, a, "auroc", hashes), finding_values(cfg, s, b, "auroc", hashes))
            sm = summarize(boot)
            rows.append({"set": s, "kind": kind, "model": b, "comparator": a, "point": point, "mean": sm["mean"], "sd": sm["sd"],
                         "ci_low": sm["lo"], "ci_high": sm["hi"], "n": n, "n_findings": nf, "n_replicates": sm["n_finite"]})
    return pd.DataFrame(rows)


def series_steps(cfg: Dict) -> pd.DataFrame:
    c = _read(cfg, "contrasts_dir", "contrasts")
    st = c[(c["family"] == "A") & (c["metric"] == "auroc") & c["contrast"].str.endswith(" step")].copy()
    st[["larger", "smaller"]] = st["models"].str.split(" - ", expand=True)
    p = pd.read_csv(facts_csv(cfg), low_memory=False).set_index("model")["parameters"].astype(float)
    st["smaller_parameters"] = st["smaller"].map(p)
    st["doublings"] = np.log2(st["larger"].map(p) / st["smaller"].map(p))
    st["change_per_doubling"] = st["mean_raw"] / st["doublings"]
    return st[["set", "contrast", "models", "smaller", "larger", "smaller_parameters", "doublings", "mean_raw", "change_per_doubling",
               "significant_fdr05"]]


def _first_best(score: pd.Series) -> str:
    top = score.max()
    return next(m for m in score.index if score[m] == top)


def _questions(cfg: Dict, set_name: str) -> pd.DataFrame:
    q = pd.read_csv(manifest_csv(cfg, set_name), low_memory=False).sort_values("question_id", kind="mergesort").reset_index(drop=True)
    return q.assign(cluster=q["dataset"].astype(str) + "|" + q["group_id"].astype(str))


def other_modality_choice(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mains = modality_sets(cfg)
    models = list(cfg["models"])
    point = pd.DataFrame({s: macro_auroc(cfg, s).set_index("model")["point_raw"] for s in mains}).reindex(index=models)
    if point.isna().any().any():
        raise ValueError("a model lacks the macro AUROC of a main set")
    pick: Dict[str, Dict[str, str]] = {"mean": {}, "median": {}, "mean_rank": {}}
    ranks = []
    for s in mains:
        other = point.drop(columns=s)
        mean = other.mean(axis=1)
        pick["mean"][s], pick["median"][s] = _first_best(mean), _first_best(other.median(axis=1))
        pick["mean_rank"][s] = _first_best(-other.rank(ascending=False).mean(axis=1))
        printed = macro_auroc(cfg, s).set_index("model")["mean_pct"]
        top = printed[printed == printed.max()]
        if len(top) != 1:
            raise ValueError(f"the leading model is tied: {s}")
        ranks.append({"set": s, "leader": top.index[0], "leader_rank_other_mean": int((mean > mean[top.index[0]]).sum()) + 1, "choice_mean": pick["mean"][s],
                      "margin_over_second": float(mean.max() - mean.drop(pick["mean"][s]).max())})
    if pick["mean_rank"] != pick["mean"]:
        raise ValueError("the mean rank over the other modalities chooses another model than the mean")
    marked = marked_pairs(training_table(cfg))
    rows = []
    for s in mains:
        q = _questions(cfg, s)
        used, P, missing = panel_scores(cfg, s, q)
        if missing:
            raise ValueError(f"{s}: {len(missing)} models lack P(Yes)")
        keep_unmarked = ~q["finding"].isin(marked_findings(cfg, marked, s, used)).to_numpy()
        for scope, keep in (("all", np.ones(len(q), bool)), ("unmarked", keep_unmarked)):
            if not keep.any():
                continue
            hows = ["mean", "median"] if scope == "all" else ["mean"]
            d = run_draws(cfg, q[keep].reset_index(drop=True), used, P[:, keep], {f"other_{h}": pick[h][s] for h in hows})
            first = d[d["n"] == d["n"].iloc[0]]
            for h in hows:
                v = first[f"regret_other_{h}"].to_numpy(dtype=float)
                rows.append({"set": s, "scope": scope, "aggregation": h, "model": pick[h][s], "n_draws": int(v.size), "mean_raw": float(v.mean()),
                             "sd_raw": float(v.std(ddof=1)), "p2_5_raw": float(np.percentile(v, 2.5)), "p97_5_raw": float(np.percentile(v, 97.5)),
                             "share_at_most_0_01": float(np.mean(v <= 0.01))})
    return pd.DataFrame(rows), pd.DataFrame(ranks)


def _row(section: str, quantity: str, scope: str, value) -> Dict:
    return {"section": section, "quantity": quantity, "scope": scope, "value": value}


def _significant(t: pd.DataFrame) -> pd.Series:
    return t["significant_fdr05"].astype(str).str.lower().isin(["true", "1"])


def _signs(section: str, t: pd.DataFrame, scope: str) -> List[Dict]:
    sig = _significant(t)
    return [_row(section, "differences", scope, len(t)), _row(section, "significant increases", scope, int((sig & (t["mean_raw"] > 0)).sum())),
            _row(section, "significant decreases", scope, int((sig & (t["mean_raw"] < 0)).sum())),
            _row(section, "lowest", scope, float(t["mean_raw"].min())), _row(section, "highest", scope, float(t["mean_raw"].max()))]


def summary_rows(cfg: Dict, steps: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    mains = modality_sets(cfg)
    leaders = {}
    for s in mains:
        m = macro_auroc(cfg, s).sort_values("mean_raw", ascending=False).reset_index(drop=True)
        leaders[s] = m.loc[0, "model"]
        rows += [_row("image use", "leading model", s, leaders[s]), _row("image use", "leading AUROC", s, float(m.loc[0, "mean_raw"])),
                 _row("image use", "median AUROC of the models", s, float(m["mean_raw"].median())),
                 _row("image use", "models above chance", s, int((m["ci_low_raw"] > 0.5).sum()))]
        ranks = m.set_index("model")["mean_pct"].rank(ascending=False, method="min")
        for model, rank in ranks.items():
            rows.append(_row("rank", model, s, int(rank)))
        ba = pd.read_csv(metrics_csv(cfg, s, "metrics"), low_memory=False)
        ba = ba[(ba["metric"] == "balanced_accuracy") & (ba["scope"] == "macro") & ba["model"].isin(cfg["models"])]
        for model, rank in ba.set_index("model")["mean_pct"].rank(ascending=False, method="min").items():
            rows.append(_row("balanced accuracy rank", model, s, int(rank)))
    rows.append(_row("image use", "distinct leading models", "all", len(set(leaders.values()))))

    rows += _signs("size series, adjacent steps", steps, "all")
    pooled = steps.groupby(steps["smaller_parameters"] < SMALL_PARAMETERS)["change_per_doubling"].mean()
    rows += [_row("size series, change per doubling", "mean of the steps", "from a model below 4B", float(pooled.get(True))),
             _row("size series, change per doubling", "mean of the steps", "from a model of 4B or more", float(pooled.get(False)))]

    c = _read(cfg, "contrasts_dir", "contrasts")
    b = c[(c["family"] == "B") & (c["metric"] == "auroc")]
    rows += _signs("generation steps", b[b["contrast"] == "generation"], "all")
    for s in mains:
        rows += _signs("medical pairs", b[(b["contrast"] == "medical") & (b["set"] == s)], s)
    medical = b[b["contrast"] == "medical"]
    for model, g in medical.groupby(medical["models"].str.split(" - ").str[0]):
        rows += _signs("medical pairs, main sets", g, model)
    ext = _read(cfg, "externals_dir", "external_contrasts")
    ext = ext[ext["family_set"].isin(external_sets(cfg)) & (ext["metric"] == "auroc")]
    rows += _signs("training steps relative to size series, external sets", ext[ext["family"] == "C"], "all")
    med = ext[(ext["family"] == "B") & (ext["contrast"] == "medical")]
    for model, g in med.groupby(med["models"].str.split(" - ").str[0]):
        rows += _signs("medical pairs, external sets", g, model)

    target = float(cfg["selection"]["regret_target"])
    reg = _read(cfg, "selection_dir", "selection_regret")
    for s in mains:
        sel = reg[(reg["set"] == s) & (reg["rule"] == "selection")].sort_values("n_per_finding")
        enough = sel.loc[sel["mean_raw"] <= target, "n_per_finding"]
        rows.append(_row("selection", "smallest label budget per finding with a mean regret within the criterion", s,
                         int(enough.min()) if len(enough) else None))
        draws = _read(cfg, "selection_dir", f"draws_{s}")
        for n, d in draws.groupby("n"):
            share = d["top_model"].value_counts(normalize=True)
            rows += [_row("selection", f"share of draws within the criterion at budget {n}", s, float((d["regret"] <= target).mean())),
                     _row("selection", f"most often selected model at budget {n}", s, share.index[0]),
                     _row("selection", f"share of draws selecting it at budget {n}", s, float(share.iloc[0])),
                     _row("selection", f"models selected at budget {n}", s, int(d["top_model"].nunique()))]

    bc = _read(cfg, "benchmarks_dir", "benchmark_correlations")
    rows += [_row("benchmarks", "lowest Spearman rho", "all", float(bc["value"].min())), _row("benchmarks", "highest Spearman rho", "all", float(bc["value"].max())),
             _row("benchmarks", "significant correlations", "all", int(_significant(bc).sum())), _row("benchmarks", "correlations", "all", len(bc))]
    accuracy = _read(cfg, "benchmarks_dir", "benchmark_accuracy")
    for r in bc.itertuples(index=False):
        a = accuracy[(accuracy["benchmark"] == r.benchmark) & (accuracy["scope"] == r.scope) & (accuracy["measure"] == "accuracy_image")].set_index("model")["mean_pct"]
        top = a[a == a.max()]
        if len(top) != 1:
            raise ValueError(f"benchmark leader is tied: {r.benchmark}/{r.scope}")
        model = top.index[0]
        m = macro_auroc(cfg, r.modality).set_index("model")["mean_pct"]
        scope = f"{r.benchmark}/{r.scope}/{r.modality}"
        rows += [_row("benchmark leaders", "leading model", scope, model),
                 _row("benchmark leaders", "rank in within-finding AUROC", scope, int((m > m[model]).sum()) + 1),
                 _row("benchmark leaders", "benchmark rank of the image-use leader", scope, int((a > a[leaders[r.modality]]).sum()) + 1)]

    ra = _read(cfg, "externals_dir", "external_rank_agreement")
    rows += [_row("external sets", "significant rank agreements", "all", int(_significant(ra).sum())), _row("external sets", "rank agreements", "all", len(ra))]
    rob = _read(cfg, "robustness_dir", "robustness_metrics")
    det = rob[rob["axis"] == "determinism"].pivot_table(index="model", columns="set", values="point_raw")
    rows.append(_row("robustness", "models with at least 0.86 identical outputs in every modality", "all", int((det.min(axis=1) >= 0.86).sum())))
    res = rob[rob["axis"] == "resolution"]
    rows += [_row("robustness", "significant resolution changes", "all", int(_significant(res).sum())), _row("robustness", "resolution changes", "all", len(res))]
    second = rob[(rob["axis"] == "phrasing") & (rob["measure"].isin(["generation, second phrasing", "medical, second phrasing"]))]
    con = _read(cfg, "contrasts_dir", "contrasts")
    main = con[(con["family"] == "B") & (con["metric"] == "auroc")][["models", "set", "mean_raw", "significant_fdr05"]]
    both = second.rename(columns={"model": "models"}).merge(main, on=["models", "set"], suffixes=("_second", "_main"))
    sig = both["significant_fdr05_main"].astype(str).str.lower().isin(["true", "1"])
    same = sig & (np.sign(both["mean_raw_main"]) == np.sign(both["mean_raw_second"]))
    rows += [_row("robustness", "training-step changes significant under the main phrasing", "all", int(sig.sum())),
             _row("robustness", "of these, changes with the same sign under the second phrasing", "all", int(same.sum()))]
    for s in mains:
        a = pd.read_csv(metrics_csv(cfg, s, "answers"), low_memory=False)
        a = a[(a["model"] == "LLaVA-Med-7B") & (a["condition"] == "original")]
        rows.append(_row("medical pairs", "share of Yes answers of LLaVA-Med-7B", s, float(a["class_yes"].sum() / a["n"].sum())))
    return rows


def main_reported_statistics(cfg_path: str) -> None:
    cfg = read_config(cfg_path)[PROJECT]
    os.makedirs(os.path.dirname(_out(cfg, "x")), exist_ok=True)
    steps = series_steps(cfg)
    write_csv_if_changed(paired_differences(cfg), _out(cfg, "paired_differences"))
    write_csv_if_changed(steps, _out(cfg, "series_steps"))
    write_csv_if_changed(pd.DataFrame(summary_rows(cfg, steps)), _out(cfg, "reported_statistics"))
    choice, ranks = other_modality_choice(cfg)
    write_csv_if_changed(choice, _out(cfg, "other_modality_choice"))
    write_csv_if_changed(ranks, _out(cfg, "other_modality_ranks"))
