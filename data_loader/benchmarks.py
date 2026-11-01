"""
data_loader/benchmarks.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_question_sets import PROJECT, absent_inputs, manifest_csv, run_build_unit
from data_loader.build_utils import file_signature, read_csv_defensively, rel_to_root
from data_loader.sets import BENCHMARK_PREFIX, benchmark_sets
from Inference.resume_utils import MissingInput, append_status, status_path, write_csv_atomic


BENCH_VERSION = 1
LETTERS = "ABCDEFGH"
_PARQUET: Dict[str, object] = {}
MANIFEST_COLUMNS = ["question_id", "benchmark", "kind", "question", "options", "label", "answer", "source_modality", "study_modality", "case_id",
                    "group_id", "dataset", "image_path"]


def parquet_image(path: str, row: int) -> bytes:
    if path not in _PARQUET:
        import pyarrow.parquet as pq
        _PARQUET[path] = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    cell = _PARQUET[path][int(row)]
    return cell["bytes"] if isinstance(cell, dict) else bytes(cell)


def _image_id(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def _yes_no(answer: str, bcfg: Dict) -> Optional[int]:
    a = str(answer).strip().lower()
    return 1 if a in bcfg["yes_answers"] else 0 if a in bcfg["no_answers"] else None


def _row(name: str, qid: str, kind: str, question: str, options: List[List[str]], label: int, answer: str, image_key: str, image_path: str,
         source_modality: str = "", study_modality: str = "") -> Dict:
    return {"question_id": f"{BENCHMARK_PREFIX}{name}__{qid}", "benchmark": name, "kind": kind, "question": str(question).strip(),
            "options": json.dumps(options) if options else "", "label": int(label), "answer": answer, "source_modality": source_modality,
            "study_modality": study_modality, "case_id": f"{name}__{image_key}", "group_id": image_key, "dataset": name, "image_path": image_path}


def _parquet_yesno(name: str, scfg: Dict, bcfg: Dict, root: str) -> List[Dict]:
    import pyarrow.parquet as pq
    rows = []
    for f_index, path in enumerate(scfg["files"]):
        t = pq.read_table(path, columns=[scfg["question_column"], scfg["answer_column"]]).to_pandas()
        for i, (q, a) in enumerate(zip(t[scfg["question_column"]], t[scfg["answer_column"]])):
            label = _yes_no(a, bcfg)
            if label is None:
                continue
            ident = _image_id(parquet_image(path, i))
            rows.append(_row(name, f"{f_index}_{i}", "yesno", q, [], label, "yes" if label else "no", ident, f"{rel_to_root(path, root)}#{i}"))
    return rows


def _slake(name: str, scfg: Dict, bcfg: Dict, root: str) -> List[Dict]:
    with open(scfg["questions_json"]) as f:
        records = json.load(f)
    rows = []
    for r in records:
        label = _yes_no(r["answer"], bcfg)
        if r.get("q_lang") != scfg["language"] or label is None:
            continue
        rows.append(_row(name, str(r["qid"]), "yesno", r["question"], [], label, "yes" if label else "no", str(r["img_name"]).replace("/", "_"),
                         rel_to_root(os.path.join(scfg["image_dir"], r["img_name"]), root), source_modality=str(r.get("modality", ""))))
    return rows


def _strip_letter(text: str, letter: str) -> str:
    t = str(text).strip()
    for sep in (":", ".", ")"):
        if t.upper().startswith(letter + sep):
            return t[len(letter) + 1:].strip()
    return t


def _pmcvqa(name: str, scfg: Dict, bcfg: Dict, root: str) -> List[Dict]:
    t = read_csv_defensively(scfg["questions_csv"], dtype=str)
    need = [scfg["image_column"], scfg["question_column"], scfg["answer_column"]] + [c for _, c in scfg["choice_columns"]]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise ValueError(f"[{name}] test_clean lacks {missing}; its columns are {list(t.columns)}")
    rows = []
    for i, r in t.iterrows():
        options = [[letter, _strip_letter(r[col], letter)] for letter, col in scfg["choice_columns"] if isinstance(r[col], str) and r[col].strip()]
        letters = [o[0] for o in options]
        answer = str(r[scfg["answer_column"]]).strip().upper()
        if answer not in letters:
            raise ValueError(f"[{name}] row {i}: the answer {answer!r} is not one of the options {letters}")
        rows.append(_row(name, str(i), "options", r[scfg["question_column"]], options, letters.index(answer), answer, str(r[scfg["image_column"]]),
                         rel_to_root(os.path.join(scfg["image_dir"], r[scfg["image_column"]]), root)))
    return rows


def _modality_name(r: Dict) -> str:
    return str(r.get("modality_type", r.get("modality", "")))


def _omnimedvqa(name: str, scfg: Dict, bcfg: Dict, root: str) -> List[Dict]:
    records = []
    for f in sorted(os.listdir(scfg["qa_dir"])):
        if f.endswith(".json"):
            with open(os.path.join(scfg["qa_dir"], f)) as fh:
                records += json.load(fh)
    names = sorted({_modality_name(r) for r in records})
    match = {}
    for study, prefix in scfg["modalities"]:
        hits = [n for n in names if n.lower().startswith(str(prefix).lower())]
        if len(hits) != 1:
            raise ValueError(f"[{name}] the prefix {prefix!r} of {study} matches {hits}; the modality names in the files are {names}")
        match[hits[0]] = str(study)
    rows = []
    rng_seed = int(scfg["seed"])
    for source, study in sorted(match.items()):
        pool = sorted((r for r in records if _modality_name(r) == source), key=lambda r: str(r["question_id"]))
        take = np.random.default_rng(rng_seed).permutation(len(pool))[: int(scfg["per_modality"])]
        for k in sorted(take):
            r = pool[k]
            options = [[letter, str(r[f"option_{letter}"]).strip()] for letter in LETTERS if str(r.get(f"option_{letter}") or "").strip()]
            texts = [o[1] for o in options]
            answer = str(r["gt_answer"]).strip()
            if texts.count(answer) != 1:
                raise ValueError(f"[{name}] question {r['question_id']}: the answer {answer!r} is not one of the options {texts}")
            idx = texts.index(answer)
            rows.append(_row(name, str(r["question_id"]), "options", r["question"], options, idx, options[idx][0], str(r["image_path"]).replace("/", "_"),
                             rel_to_root(os.path.join(scfg["root"], r["image_path"]), root), source_modality=source, study_modality=study))
    return rows


READERS: Dict[str, Callable[[str, Dict, Dict, str], List[Dict]]] = {
    "parquet_yesno": _parquet_yesno, "slake": _slake, "pmcvqa": _pmcvqa, "omnimedvqa": _omnimedvqa}


def _signature(scfg: Dict) -> Dict:
    paths = [p for v in scfg.values() for p in (v if isinstance(v, list) else [v]) if isinstance(p, str) and os.path.isabs(p) and os.path.isfile(p)]
    return {os.path.basename(p): file_signature(p) for p in paths}


def _valid(path: str) -> bool:
    try:
        t = read_csv_defensively(path, usecols=["question_id"], dtype=str)
    except (OSError, ValueError):
        return False
    return len(t) > 0 and not t["question_id"].duplicated().any()


def main_build_benchmarks(cfg_path: str) -> None:
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    bcfg = cfg["benchmarks"]
    status = status_path(cfg, "main_build_benchmarks")
    skipped = []
    for set_name in benchmark_sets(cfg):
        name = set_name[len(BENCHMARK_PREFIX):]
        scfg = bcfg["sources"][name]
        absent = absent_inputs(scfg)
        if absent:
            skipped.append(f"{set_name} (lacks {', '.join(absent[:2])})")
            continue
        params = {"stage": "main_build_benchmarks benchmark", "version": BENCH_VERSION, "benchmark": name,
                  "settings": {k: v for k, v in scfg.items() if not (isinstance(v, str) and os.path.isabs(v))}, "files": _signature(scfg),
                  "answers": {"yes": bcfg["yes_answers"], "no": bcfg["no_answers"]}}

        def build(set_name=set_name, name=name, scfg=scfg):
            rows = READERS[scfg["reader"]](name, scfg, bcfg, root)
            if not rows:
                raise ValueError(f"[main_build_benchmarks] {set_name}: no question passed the rules")
            t = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
            if t["question_id"].duplicated().any():
                raise ValueError(f"[main_build_benchmarks] {set_name}: duplicated question ids")
            write_csv_atomic(t, manifest_csv(cfg, set_name))
            append_status(status, f"main_build_benchmarks {set_name}: {len(t)} questions")
        try:
            run_build_unit(cfg, f"build_benchmarks__{set_name}", manifest_csv(cfg, set_name), params, lambda set_name=set_name: _valid(manifest_csv(cfg, set_name)), build)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    if skipped:
        raise MissingInput(f"[main_build_benchmarks] {len(skipped)} benchmark(s) not built: {'; '.join(skipped)}")
