"""
Inference/reparse_records.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List

from config.serde import read_config
from Inference.parser import option_parser_version, parser_version, reparse, reparse_option
from Inference.resume_utils import (MissingInput, append_status, claim_unit, read_jsonl, release_claim, status_path,
                                    write_jsonl_atomic)


PROJECT = "Crossmodality"


def _record_files(runs_dir: str) -> List[str]:
    out = []
    for set_name in sorted(os.listdir(runs_dir)):
        set_dir = os.path.join(runs_dir, set_name)
        if not os.path.isdir(set_dir):
            continue
        for model in sorted(os.listdir(set_dir)):
            model_dir = os.path.join(set_dir, model)
            if os.path.isdir(model_dir):
                out += [os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir)) if f.endswith(".jsonl")]
    return out


def main_reparse_records(cfg_path: str) -> None:
    from tqdm import tqdm
    cfg = read_config(cfg_path)[PROJECT]
    runs_dir = cfg["runs"]["dir"]
    if not os.path.isdir(runs_dir):
        raise MissingInput(f"no record store at {runs_dir}; run main_run first")
    version = parser_version(cfg["parser"])
    option_version = option_parser_version(cfg["parser"], cfg["option_parser"])

    def expected(rec):
        return option_version if rec.get("options") else version
    status = status_path(cfg, "reparse_records")
    files = _record_files(runs_dir)
    busy, unknown, n_units, n_changed, n_removed = [], set(), 0, 0, 0
    for path in tqdm(files, desc="[main_compute_metrics] reparse", unit="unit"):
        model = os.path.basename(os.path.dirname(path))
        if model not in cfg["models"]:
            unknown.add(model)
            continue
        if all(r.get("parser_version") == expected(r) for r in read_jsonl(path, "reparse_records")):
            continue
        udir, tag = os.path.dirname(path), os.path.basename(path)[:-len(".jsonl")]
        if not claim_unit(udir, tag, float(cfg["runs"]["claim_stale_after_s"])):
            busy.append(path)
            continue
        try:
            records, kept = read_jsonl(path, "reparse_records"), []
            for rec in records:
                if rec.get("parser_version") != expected(rec):
                    reasoning = bool(cfg["models"][model]["reasoning"])
                    new = reparse_option(rec, reasoning, cfg["parser"], cfg["option_parser"]) if rec.get("options") else reparse(rec, reasoning, cfg["parser"])
                    if new is None:
                        n_removed += 1
                        continue
                    rec.update(new)
                    n_changed += 1
                kept.append(rec)
            write_jsonl_atomic(path, kept)
        finally:
            release_claim(udir, tag)
        n_units += 1
        append_status(status, f"{os.path.relpath(path, runs_dir)}: reparsed")
    if busy:
        raise MissingInput(f"[main_compute_metrics] {len(busy)} unit(s) are with a live job (first: {busy[0]}); run it again when it is done")
