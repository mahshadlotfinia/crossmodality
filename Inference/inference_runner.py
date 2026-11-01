"""
Inference/inference_runner.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from config.serde import read_config
from data_loader.build_question_sets import PROJECT, manifest_csv
from data_loader.build_utils import read_csv_defensively
from data_loader.render_inputs import render_table_csv, renders_current, robustness_csv
from data_loader.sets import BENCHMARK, MAIN, benchmark_sets, external_sets, modality_sets, set_kind
from Inference.model_wrappers import NoLogprobs, ServedModel, benchmark_prompt, model_prompt, question_stem
from Inference.resume_utils import (MissingInput, append_jsonl, append_status, claim_unit, ensure_dir, heartbeat_claim, read_jsonl,
                                    release_claim, status_path)
from Inference.serving_utils import require_packages


MAIN_CONDITIONS = ("original",)
EXTERNAL_CONDITIONS = ("original",)
BENCHMARK_CONDITIONS = ("original", "no_image")
ITEM_FIELDS = ("item_id", "question_id", "finding", "label", "image_label", "image_case_id", "repeat", "input_key", "request_digest", "options")


@dataclass(frozen=True)
class Unit:
    set_name: str
    model: str
    condition: str
    variant: str = "default"

    @property
    def name(self) -> str:
        return self.condition if self.variant == "default" else f"{self.condition}__{self.variant}"

    def __str__(self) -> str:
        return f"{self.set_name}/{self.model}/{self.name}"


def unit_dir(cfg: Dict, u: Unit) -> str:
    return os.path.join(cfg["runs"]["dir"], u.set_name, u.model)


def unit_path(cfg: Dict, u: Unit) -> str:
    return os.path.join(unit_dir(cfg, u), f"{u.name}.jsonl")


def load_records(path: str) -> Tuple[Dict[str, Dict], int, int]:
    newest: Dict[str, Dict] = {}
    copies: Dict[Tuple, set] = {}
    counts: Dict[str, int] = {}
    for rec in read_jsonl(path, "inference_runner"):
        iid = str(rec["item_id"])
        newest[iid] = rec
        counts[iid] = counts.get(iid, 0) + 1
        p = rec.get("p_yes")
        copies.setdefault((iid, rec.get("input_key"), rec.get("request_digest")), set()).add(
            (int(rec.get("parsed_answer", -1)), None if p is None else round(float(p), 6)))
    multi = sum(1 for n in counts.values() if n > 1)
    disagree = len({k[0] for k, v in copies.items() if len(v) > 1})
    return newest, multi, disagree


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def conditions_of(cfg: Dict, set_name: str) -> Tuple[str, ...]:
    kind = set_kind(cfg, set_name)
    return MAIN_CONDITIONS if kind == MAIN else BENCHMARK_CONDITIONS if kind == BENCHMARK else EXTERNAL_CONDITIONS


class SetData:

    def __init__(self, full: Dict, set_name: str):
        cfg = full[PROJECT]
        reason = renders_current(cfg, set_name)
        if reason:
            raise MissingInput(f"{set_name}: {reason}")
        self.set_name, self.kind = set_name, set_kind(cfg, set_name)
        self.manifest = read_csv_defensively(manifest_csv(cfg, set_name), dtype={"key": str, "group_id": str, "case_id": str}, keep_default_na=False)
        table = read_csv_defensively(render_table_csv(cfg, set_name), keep_default_na=False)
        self.renders = {k: (os.path.join(full["outputs_root"], p), d) for k, p, d in zip(table["render_key"], table["path"], table["digest"])}
        self.subsets = {}
        if self.kind == MAIN:
            robust = read_csv_defensively(robustness_csv(cfg, set_name), keep_default_na=False)
            self.subsets = {c: set(robust.loc[_flag(robust[c]), "question_id"]) for c in ("in_phrasing", "in_determinism")}

    def render(self, key: str) -> Tuple[str, str]:
        if key not in self.renders:
            raise MissingInput(f"{self.set_name}: no render {key} in the render table; run main_render_inputs")
        return self.renders[key]


def _rows(sd: SetData, u: Unit) -> pd.DataFrame:
    m = sd.manifest.sort_values("question_id", kind="mergesort")
    if u.variant in ("phrasing", "determinism") or u.condition == "original_small":
        return m[m["question_id"].isin(sd.subsets["in_determinism" if u.variant == "determinism" else "in_phrasing"])]
    if u.condition == "original":
        return m
    raise ValueError(f"unknown condition {u.condition!r}")


def _item(model: ServedModel, prompt: str, image: Optional[Tuple[str, str]], render_key: Optional[str], options: Optional[List] = None, **fields) -> Dict:
    item = dict(fields, prompt=prompt, request_digest=model.request_digest(prompt), image_path=image[0] if image else None,
                input_key=f"{render_key}|{image[1]}" if image else "none", options=options)
    return item


def _benchmark_items(sd: SetData, u: Unit, model: ServedModel, cfg: Dict) -> List[Dict]:
    if u.condition not in BENCHMARK_CONDITIONS:
        raise ValueError(f"unknown benchmark condition {u.condition!r}")
    items = []
    for r in sd.manifest.sort_values("question_id", kind="mergesort").itertuples(index=False):
        options = json.loads(r.options) if r.options else None
        prompt = benchmark_prompt(r.question, options, model.reasoning, cfg["prompts"])
        key = f"original|{r.case_id}" if u.condition == "original" else None
        image = sd.render(key) if key else None
        items.append(_item(model, prompt, image, key, options, item_id=r.question_id, question_id=r.question_id, finding=r.benchmark, label=int(r.label),
                           image_label=int(r.label) if image else None, image_case_id=r.case_id if image else "", repeat=0))
    return items


def unit_items(sd: SetData, u: Unit, model: ServedModel, cfg: Dict) -> List[Dict]:
    if sd.kind == BENCHMARK:
        return _benchmark_items(sd, u, model, cfg)
    prompts = cfg["prompts"]
    items: List[Dict] = []
    repeats = int(cfg["runs"]["determinism_repeats"]) if u.variant == "determinism" else 1
    for r in _rows(sd, u).itertuples(index=False):
        prompt = model_prompt(question_stem(r.template, r.display, r.noun, prompts), r.modality, u.variant, model.reasoning, prompts)
        label = int(r.label)
        case, key, image_label = r.case_id, f"{u.condition}|{r.case_id}", label
        image = sd.render(key)
        for k in range(repeats):
            items.append(_item(model, prompt, image, key, item_id=r.question_id if repeats == 1 else f"{r.question_id}__rep{k}",
                               question_id=r.question_id, finding=r.finding, label=label, image_label=image_label, image_case_id=case, repeat=k))
    return items


def _done(rec: Optional[Dict], item: Dict, budget: List[int], model: ServedModel) -> bool:
    if rec is None or rec.get("input_key") != item["input_key"] or rec.get("request_digest") != item["request_digest"]:
        return False
    if rec.get("budget") == budget:
        return True
    return not model.reasoning and model.parse(rec.get("raw_answer"), item.get("options")) >= 0


def _todo(path: str, items: List[Dict], model: ServedModel, u: Unit, announce: bool) -> List[Dict]:
    newest, multi, disagree = load_records(path)
    return [it for it in items if not _done(newest.get(it["item_id"]), it, model.budget_for(it.get("options")), model)]


def run_unit(cfg: Dict, u: Unit, model: ServedModel, items: List[Dict], status_file: str) -> Dict[str, int]:
    from tqdm import tqdm
    path = unit_path(cfg, u)
    todo = _todo(path, items, model, u, announce=True)
    counts = {"items": len(items), "done_before": len(items) - len(todo), "run": 0, "failed": 0}
    if not todo:
        return counts
    ensure_dir(unit_dir(cfg, u))
    if not claim_unit(unit_dir(cfg, u), u.name, float(cfg["runs"]["claim_stale_after_s"])):
        counts["claimed_elsewhere"] = 1
        return counts
    try:
        todo = _todo(path, items, model, u, announce=False)
        wave = max(1, int(cfg["serving"]["concurrency"]) * int(cfg["runs"]["wave_factor"]))
        limit = int(cfg["serving"]["max_consecutive_failures"])
        consecutive = 0
        bar = tqdm(total=len(todo), desc=f"[run] {u}", unit="item")
        for start in range(0, len(todo), wave):
            chunk = todo[start:start + wave]
            try:
                results = model.answer_batch(chunk)
            except NoLogprobs:
                raise
            except Exception as e:
                if not model.served_now():
                    raise MissingInput(f"[run] {u}: {model.model_key} is no longer served ({type(e).__name__}); load it and run it again") from e
                raise
            for it, res in zip(chunk, results):
                if res is None:
                    counts["failed"] += 1
                    consecutive += 1
                    if consecutive >= limit:
                        if not model.served_now():
                            raise MissingInput(f"[run] {u}: {model.model_key} stopped answering and is no longer served; load it and run it again")
                        raise RuntimeError(f"[run] {u}: {limit} consecutive failed calls while the server lists {model.model_key}. Stopping.")
                    continue
                consecutive = 0
                rec = {k: it[k] for k in ITEM_FIELDS}
                rec.update(res)
                rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                append_jsonl(path, rec)
                counts["run"] += 1
            bar.update(len(chunk))
            heartbeat_claim(unit_dir(cfg, u), u.name)
            append_status(status_file, f"{u}: {counts['run']} written, {counts['failed']} failed, {bar.n}/{len(todo)}")
        bar.close()
    finally:
        release_claim(unit_dir(cfg, u), u.name)
    return counts


def plan_main(cfg: Dict, model_key: str) -> List[Unit]:
    return [Unit(s, model_key, c) for s in modality_sets(cfg) for c in MAIN_CONDITIONS]


def plan_external(cfg: Dict, model_key: str) -> List[Unit]:
    return ([Unit(s, model_key, c) for s in external_sets(cfg) for c in EXTERNAL_CONDITIONS]
            + [Unit(b, model_key, c) for b in benchmark_sets(cfg) for c in BENCHMARK_CONDITIONS])


def plan_robustness(cfg: Dict, model_key: str) -> List[Unit]:
    plan = []
    for s in modality_sets(cfg):
        plan += [Unit(s, model_key, "original", "phrasing"), Unit(s, model_key, "original", "determinism")]
        if model_key in cfg["runs"]["robustness_models"]:
            plan += [Unit(s, model_key, "original_small")]
    return plan


def _run_plan(cfg_path: str, model_key: str, stage: str, planner: Callable[[Dict, str], List[Unit]]) -> None:
    require_packages(["openai", "httpx"], stage)
    full = read_config(cfg_path)
    cfg = full[PROJECT]
    if model_key not in cfg["models"]:
        raise KeyError(f"unknown model {model_key}; the panel is {list(cfg['models'])}")
    model = ServedModel(model_key, cfg)
    status = status_path(cfg, f"{stage}_{model_key}")
    plan = planner(cfg, model_key)
    data: Dict[str, object] = {}
    work, skipped = [], {}
    for u in plan:
        if u.set_name not in data:
            try:
                data[u.set_name] = SetData(full, u.set_name)
            except MissingInput as e:
                data[u.set_name] = e
        sd = data[u.set_name]
        if isinstance(sd, MissingInput):
            skipped[u.set_name] = str(sd)
            continue
        try:
            work.append((u, unit_items(sd, u, model, cfg)))
        except MissingInput as e:
            skipped[str(u)] = str(e)
    open_units = [u for u, items in work if _todo(unit_path(cfg, u), items, model, u, announce=False)]
    if open_units:
        model.require_served()
        model.require_image_input()
    append_status(status, f"{stage} {model_key} start: {len(open_units)} unit(s) with open items")
    total = {"run": 0, "failed": 0}
    for u, items in work:
        counts = run_unit(cfg, u, model, items, status)
        total["run"] += counts["run"]
        total["failed"] += counts["failed"]
    append_status(status, f"{stage} {model_key} done: {total['run']} record(s), {total['failed']} failed call(s)")
    if skipped:
        raise MissingInput(f"[{stage}] {model_key}: {len(skipped)} without input: " + "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))


def main_run(cfg_path: str, model_key: str) -> None:
    _run_plan(cfg_path, model_key, "main_run", plan_main)


def main_run_external(cfg_path: str, model_key: str) -> None:
    _run_plan(cfg_path, model_key, "main_run_external", plan_external)


def main_run_robustness(cfg_path: str, model_key: str) -> None:
    _run_plan(cfg_path, model_key, "main_run_robustness", plan_robustness)
