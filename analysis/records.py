"""
analysis/records.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader.build_question_sets import questions_csv
from Inference.inference_runner import MAIN_CONDITIONS, SetData, Unit, load_records, unit_items, unit_path
from Inference.model_wrappers import ServedModel
from Inference.parser import parser_version
from Inference.resume_utils import digest_file, read_build_params


def file_fingerprint(paths: List[str]) -> Dict[str, Optional[List]]:
    out = {}
    for p in paths:
        try:
            st = os.stat(p)
            out[os.path.basename(p)] = [int(st.st_size), round(float(st.st_mtime), 3)]
        except OSError:
            out[os.path.basename(p)] = None
    return out


def unit_files(cfg: Dict, set_name: str, model_key: str, conditions=MAIN_CONDITIONS) -> List[str]:
    return [unit_path(cfg, Unit(set_name, model_key, c)) for c in conditions]


def served_arrays(cfg: Dict, sd: SetData, model_key: str, questions: pd.DataFrame, conditions=MAIN_CONDITIONS) -> Dict:
    model = ServedModel(model_key, cfg)
    version = parser_version(cfg["parser"])
    pos = {q: i for i, q in enumerate(questions["question_id"])}
    n = len(questions)
    out = {"ans": {}, "p": {}, "cls": {}, "src": {}, "open": {}, "stale_parser": {}}
    for cond in conditions:
        u = Unit(sd.set_name, model_key, cond)
        items = unit_items(sd, u, model, cfg)
        newest, _, _ = load_records(unit_path(cfg, u))
        current, n_open, n_parser = {}, 0, 0
        for it in items:
            rec = newest.get(it["item_id"])
            if rec is None or rec.get("input_key") != it["input_key"] or rec.get("request_digest") != it["request_digest"]:
                n_open += 1
                continue
            if rec.get("parser_version") != version:
                n_parser += 1
            current[it["item_id"]] = rec
        out["open"][cond], out["stale_parser"][cond] = n_open, n_parser
        ans, p = np.full(n, np.nan), np.full(n, np.nan)
        cls, src = np.full(n, None, dtype=object), np.full(n, None, dtype=object)
        for iid, rec in current.items():
            i = pos.get(iid)
            if i is None:
                continue
            ans[i] = float(rec["parsed_answer"])
            p[i] = float(rec["p_yes"]) if rec.get("p_yes") is not None else np.nan
            cls[i], src[i] = rec.get("answer_class"), rec.get("p_yes_source")
        out["ans"][cond], out["p"][cond], out["cls"][cond], out["src"][cond] = ans, p, cls, src
    return out


def reference_arrays(cfg: Dict, set_name: str, questions: pd.DataFrame) -> Optional[Dict]:
    from encoders.heads import reference_path
    path = reference_path(cfg, set_name)
    stamp = read_build_params(path) if os.path.exists(path) else None
    if stamp is None or stamp.get("heads", {}).get("questions") != digest_file(questions_csv(cfg, set_name)):
        return None
    newest, _, _ = load_records(path)
    case_of = dict(zip(questions["question_id"], questions["case_id"]))
    n = len(questions)
    ans, p = np.full(n, np.nan), np.full(n, np.nan)
    for i, qid in enumerate(questions["question_id"]):
        rec = newest.get(qid)
        if rec is not None and rec.get("input_key") == f"dinov3|{case_of[qid]}":
            ans[i], p[i] = float(rec["parsed_answer"]), float(rec["p_yes"])
    return {"ans": {"original": ans}, "p": {"original": p}, "cls": {}, "src": {}, "open": {"original": int(np.isnan(p).sum())}, "stale_parser": {}}


def unit_answers(cfg: Dict, sd: SetData, model_key: str, questions: pd.DataFrame, condition: str, variant: str = "default") -> Optional[Dict]:
    from Inference.parser import option_parser_version
    u = Unit(sd.set_name, model_key, condition, variant)
    path = unit_path(cfg, u)
    if not os.path.exists(path):
        return None
    items = unit_items(sd, u, ServedModel(model_key, cfg), cfg)
    newest, _, _ = load_records(path)
    versions = (parser_version(cfg["parser"]), option_parser_version(cfg["parser"], cfg["option_parser"]))
    pos = {q: i for i, q in enumerate(questions["question_id"])}
    n = len(questions)
    ans, p = np.full(n, np.nan), np.full(n, np.nan)
    raws: Dict[str, List] = {}
    n_open, n_parser = 0, 0
    for it in items:
        rec = newest.get(it["item_id"])
        if rec is None or rec.get("input_key") != it["input_key"] or rec.get("request_digest") != it["request_digest"]:
            n_open += 1
            continue
        if rec.get("parser_version") != versions[1 if it.get("options") else 0]:
            n_parser += 1
        raws.setdefault(it["question_id"], []).append(rec.get("raw_answer"))
        i = pos.get(it["question_id"])
        if i is not None and int(it.get("repeat") or 0) == 0:
            ans[i] = float(rec["parsed_answer"])
            p[i] = float(rec["p_yes"]) if rec.get("p_yes") is not None else np.nan
    return {"ans": ans, "p": p, "raw": raws, "open": n_open, "stale_parser": n_parser, "n_items": len(items)}
