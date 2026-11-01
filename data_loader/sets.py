"""
data_loader/sets.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List


MAIN, EXTERNAL, BENCHMARK = "main", "external", "benchmark"
INHERITED = ("noun", "input_size", "small_size", "channels")
BENCHMARK_PREFIX = "bench_"


def modality_sets(cfg: Dict) -> List[str]:
    return list(cfg["question_sets"]["modalities"])


def external_sets(cfg: Dict) -> List[str]:
    return list(cfg["question_sets"]["externals"])


def benchmark_sets(cfg: Dict) -> List[str]:
    return [f"{BENCHMARK_PREFIX}{name}" for name in cfg["benchmarks"]["sources"]]


def set_kind(cfg: Dict, name: str) -> str:
    if name in cfg["question_sets"]["modalities"]:
        return MAIN
    if name in cfg["question_sets"]["externals"]:
        return EXTERNAL
    if name.startswith(BENCHMARK_PREFIX) and name[len(BENCHMARK_PREFIX):] in cfg["benchmarks"]["sources"]:
        return BENCHMARK
    raise KeyError(f"unknown question set {name!r}")


def set_config(cfg: Dict, name: str) -> Dict:
    q = cfg["question_sets"]
    if name in q["modalities"]:
        return q["modalities"][name]
    if name in q["externals"]:
        ext = q["externals"][name]
        parent = q["modalities"][ext["modality"]]
        return {**{k: parent[k] for k in INHERITED}, **ext}
    raise KeyError(f"{name!r} is not a main or external question set")


def asked_findings(cfg: Dict, name: str) -> List[str]:
    return [f for f, fc in set_config(cfg, name)["findings"].items() if fc.get("n_expected") != 0]


def modality_of(cfg: Dict, name: str) -> str:
    q = cfg["question_sets"]
    return name if name in q["modalities"] else q["externals"][name]["modality"]


def sampling(cfg: Dict, name: str) -> Dict[str, int]:
    q = cfg["question_sets"]
    if name in q["modalities"]:
        return {"seed": int(q["seed"]), "n_per_class": int(q["n_per_class"]), "group_cap": int(q["group_cap"])}
    return {"seed": int(q["seed"]), "n_per_class": int(q["external_n_per_class"]), "group_cap": int(q["external_group_cap"])}
