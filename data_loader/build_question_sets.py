"""
data_loader/build_question_sets.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import functools
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (PARTITION_COLUMNS, cap_keys, capped_count, check_partition_table, read_csv_defensively,
                                     recording_writes, sample_finding, sample_matched, sample_stratified)
from data_loader.sets import external_sets, modality_sets, modality_of, sampling, set_config
from data_loader.sources import (aptos, bhsd, brats, breast, busbra, busi, camelyon17, cbis_ddsm, chestxdet, cmmd, cq500, cxr, ddi, derm,
                                 derm7pt, fitzpatrick17k, fundus, idrid, lidc, mammo, messidor2, nct_crc, padchest, padufes, pcam, picai, taix)
from Inference.resume_utils import ClaimedElsewhere, MissingInput, append_status, check_build_params, claim_beat, claim_or_wait, claim_unit, digest_file, params_path, read_build_params, release_claim, status_path, write_build_params, write_csv_atomic


PROJECT = "Crossmodality"
SOURCES = {m.DATASET: m for m in (cxr, mammo, fundus, derm, pcam, nct_crc, bhsd, lidc, brats, picai, busbra, busi, taix, padchest, chestxdet,
                                   cbis_ddsm, cmmd, cq500, breast, aptos, messidor2, idrid, derm7pt, ddi, fitzpatrick17k, padufes, camelyon17)}
BUILD_VERSION = {"main_build_partitions": 1, "main_build_question_sets": 1, "main_build_manifests": 1, "main_build_external_sets": 1}


def _out(cfg: Dict, key: str, name: str) -> str:
    return os.path.join(cfg["outputs"][key], name)


def partition_csv(cfg: Dict, dataset: str) -> str:
    return _out(cfg, "partitions_dir", f"{dataset}.csv")


def questions_csv(cfg: Dict, modality: str) -> str:
    return _out(cfg, "question_sets_dir", f"questions_{modality}.csv")


def pools_csv(cfg: Dict, modality: str) -> str:
    return _out(cfg, "question_sets_dir", f"pools_{modality}.csv")


def manifest_csv(cfg: Dict, modality: str) -> str:
    return _out(cfg, "question_sets_dir", f"manifest_{modality}.csv")


def _strings(v) -> List[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [s for x in v for s in _strings(x)]
    return []


def _is_path(v) -> bool:
    return any(os.path.isabs(s) for s in _strings(v))


def _settings(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if not _is_path(v)}


def absent_inputs(dcfg: Dict) -> List[str]:
    return [s for v in dcfg.values() for s in _strings(v) if os.path.isabs(s) and not os.path.exists(s)]


def run_build_unit(cfg: Dict, tag: str, artifact: str, params: Dict, is_valid: Callable[[], bool], build: Callable[[], None],
                   wait: bool = True) -> bool:
    claims = cfg["outputs"]["claims_dir"]

    def done():
        return (os.path.exists(artifact) and read_build_params(artifact) is not None and check_build_params(artifact, params, owner=tag)
                and is_valid())
    if wait:
        owned = claim_or_wait(claims, tag, done)
    elif done():
        owned = False
    elif not claim_unit(claims, tag):
        raise ClaimedElsewhere(f"{tag} is with another job")
    elif done():
        release_claim(claims, tag)
        owned = False
    else:
        owned = True
    if not owned:
        return False
    try:
        with claim_beat(claims, tag):
            build()
        write_build_params(artifact, params)
    finally:
        release_claim(claims, tag)
    return True


def stage_files(cfg: Dict, *kinds: str) -> List[str]:
    import glob
    o = cfg["outputs"]
    patterns = {"caches": [os.path.join(o["metrics_dir"], "boot", "*", "*.npz")],
                "tables": [os.path.join(o[k], "*.csv") for k in ("metrics_dir", "contrasts_dir", "selection_dir", "benchmarks_dir", "externals_dir",
                                                                  "robustness_dir", "contamination_dir")],
                "manifests": [os.path.join(o["question_sets_dir"], "manifest_*.csv"), os.path.join(o["renders_dir"], "render_table_*"),
                              os.path.join(o["renders_dir"], "robustness_*.csv")],
                "records": [os.path.join(cfg["runs"]["dir"], "*", "*", "*.jsonl")],
                "facts": [os.path.join(o["models_dir"], "model_facts.csv")],
                "training": [cfg["contamination"]["training_data_csv"]],
                "heads": [os.path.join(o["dinov3_dir"], "heads", "*.npz")]}
    return sorted({f for k in kinds for pat in patterns[k] for f in glob.glob(pat)})


CODE_DIRS = ("analysis", "config", "data_loader", "encoders", "Inference")


def _code_digest() -> str:
    import hashlib
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.blake2b(digest_size=16)
    for d in CODE_DIRS:
        for root, dirs, files in os.walk(os.path.join(repo, d)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                if f.endswith(".py"):
                    with open(os.path.join(root, f), "rb") as fh:
                        h.update(os.path.relpath(os.path.join(root, f), repo).encode("utf-8") + b"\0" + fh.read())
    return h.hexdigest()


def _fingerprints(files: List[str]) -> Dict:
    out = {}
    for f in files:
        try:
            st = os.stat(f)
            out[f] = [int(st.st_size), int(st.st_mtime_ns)]
        except OSError:
            out[f] = None
    return out


def _line_stamp(cfg: Dict, version: int, files: List[str]) -> Dict:
    import hashlib
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return {"version": int(version), "code": _code_digest(), "config": hashlib.blake2b(blob, digest_size=16).hexdigest(), "files": _fingerprints(files)}


def single_shot_line(tag: str, inputs: Callable[[Dict], List[str]], version: int = 1, skip: bool = True) -> Callable:
    def wrap(fn):
        @functools.wraps(fn)
        def line(cfg_path: str, *args, **kwargs):
            cfg = read_config(cfg_path)[PROJECT]
            claims, record = cfg["outputs"]["claims_dir"], os.path.join(cfg["outputs"]["status_dir"], tag)

            def current():
                if not skip or not os.path.exists(params_path(record)):
                    return False
                old = read_build_params(record)
                outputs = old.pop("outputs", None)
                return outputs is not None and outputs == _fingerprints(list(outputs)) and old == _line_stamp(cfg, version, inputs(cfg))
            if not claim_or_wait(claims, tag, current):
                return None
            try:
                stamp = _line_stamp(cfg, version, inputs(cfg))
                with recording_writes() as written, claim_beat(claims, tag):
                    result = fn(cfg_path, *args, **kwargs)
                stamp["outputs"] = _fingerprints(sorted(set(written)))
                write_build_params(record, stamp)
                return result
            finally:
                release_claim(claims, tag)
        return line
    return wrap


def run_build_units(attempts: List[Tuple[str, Callable[[bool], object]]], desc: str) -> None:
    from tqdm import tqdm
    held = []
    for tag, attempt in tqdm(attempts, desc=desc, unit="unit"):
        try:
            attempt(False)
        except ClaimedElsewhere:
            held.append((tag, attempt))
    for tag, attempt in held:
        attempt(True)


def _partition_params(cfg: Dict, dataset: str) -> Dict:
    dcfg = cfg["datasets"][dataset]
    return {"stage": "main_build_partitions", "version": BUILD_VERSION["main_build_partitions"], "dataset": dataset, "seed": int(cfg["question_sets"]["seed"]),
            "settings": _settings(dcfg), "source": SOURCES[dataset].source_signature(dcfg)}


def _partition_valid(path: str, partitions: Tuple[str, ...]) -> bool:
    try:
        t = read_csv_defensively(path, usecols=["partition"])
    except (OSError, ValueError):
        return False
    return len(t) > 0 and set(t["partition"]) == set(partitions)


def main_build_partitions(cfg_path: str) -> None:
    from tqdm import tqdm
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    status = status_path(cfg, "main_build_partitions")
    absent = {d: absent_inputs(dcfg) for d, dcfg in cfg["datasets"].items()}
    ready = [d for d, miss in absent.items() if not miss]
    append_status(status, f"main_build_partitions start: {len(ready)} of {len(absent)} dataset(s) have every input")
    for dataset in tqdm(ready, desc="[main_build_partitions] datasets", unit="dataset"):
        dcfg = cfg["datasets"][dataset]
        path = partition_csv(cfg, dataset)
        parts = tuple(dcfg["partitions"])

        def build(dataset=dataset, dcfg=dcfg, path=path, parts=parts):
            t = SOURCES[dataset].partition_table(dcfg, root, int(cfg["question_sets"]["seed"]))
            t = t.assign(group_id=t["group_id"].astype(str), key=t["key"].astype(str))[PARTITION_COLUMNS]
            check_partition_table(t, dataset, parts)
            write_csv_atomic(t, path)
            counts = t["partition"].value_counts().to_dict()
            append_status(status, f"main_build_partitions {dataset}: {len(t)} rows {counts}")
        run_build_unit(cfg, f"main_build_partitions__{dataset}", path, _partition_params(cfg, dataset), lambda path=path, parts=parts: _partition_valid(path, parts), build)
    missing = {d: m for d, m in absent.items() if m}
    if missing:
        names = "; ".join(f"{d}: {', '.join(m[:3])}" for d, m in missing.items())
        raise MissingInput(f"[main_build_partitions] {len(missing)} dataset(s) lack inputs and were not partitioned ({names}). Place them under datasets_root and re-run main_build_partitions.")
    append_status(status, "main_build_partitions done")


def read_partition(cfg: Dict, dataset: str) -> pd.DataFrame:
    path = partition_csv(cfg, dataset)
    if not os.path.exists(path) or read_build_params(path) is None:
        raise MissingInput(f"the partition table of {dataset} does not exist; run main_build_partitions (main_build_partitions)")
    t = read_csv_defensively(path, dtype={"key": str, "group_id": str, "split": str})
    check_partition_table(t, dataset, tuple(cfg["datasets"][dataset]["partitions"]))
    return t


def _question_params(cfg: Dict, set_name: str) -> Dict:
    scfg = set_config(cfg, set_name)
    datasets = sorted({f["dataset"] for f in scfg["findings"].values()})
    stage = "main_build_question_sets" if set_name in modality_sets(cfg) else "main_build_external_sets"
    return {"stage": stage, "version": BUILD_VERSION[stage], "modality": set_name, "modality_settings": scfg, "sampling": sampling(cfg, set_name),
            "partitions": {d: digest_file(partition_csv(cfg, d)) for d in datasets},
            "sources": {d: SOURCES[d].source_signature(cfg["datasets"][d]) for d in datasets}}


def _expected_rows(scfg: Dict) -> Optional[int]:
    counts = [f["n_expected"] for f in scfg["findings"].values()]
    return None if any(c is None for c in counts) else 2 * sum(int(c) for c in counts)


def _questions_valid(cfg: Dict, set_name: str) -> bool:
    paths = (questions_csv(cfg, set_name), pools_csv(cfg, set_name))
    if not all(os.path.exists(p) for p in paths):
        return False
    q = read_csv_defensively(paths[0], usecols=["question_id"], keep_default_na=False)
    expected = _expected_rows(set_config(cfg, set_name))
    return not ((expected is not None and len(q) != expected) or q["question_id"].duplicated().any())


def _check_image_sizes(rows: pd.DataFrame, dcfg: Dict, root: str, dataset: str) -> None:
    from PIL import Image
    files = [p for p in rows["image_path"] if "#" not in p]
    if not files or "image_size" not in dcfg:
        return
    want = tuple(int(v) for v in dcfg["image_size"])
    for p in files:
        with Image.open(os.path.join(root, p)) as im:
            if im.size != want:
                raise ValueError(f"[{dataset}] {p} is {im.size} on disk, not {want}; the transform of its boxes and masks assumes {want}")


def _draw(fcfg: Dict, pos: pd.DataFrame, neg: pd.DataFrame, n_max: Optional[int], cap: int, seed: int,
          pos_cap: int) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if fcfg.get("stratify"):
        return (*sample_stratified(pos, neg, n_max, cap, seed, pos_cap), 0)
    if fcfg.get("match_decile") or fcfg.get("exclude_positive_groups"):
        return sample_matched(pos, neg, n_max, cap, seed, bool(fcfg.get("match_decile")), bool(fcfg.get("exclude_positive_groups")), pos_cap)
    return (*sample_finding(pos, neg, n_max, cap, seed, pos_cap), 0)


def _build_questions(cfg: Dict, root: str, set_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scfg, smp = set_config(cfg, set_name), sampling(cfg, set_name)
    modality = modality_of(cfg, set_name)
    size = int(scfg["input_size"])
    questions, pools = [], []
    for finding, fcfg in scfg["findings"].items():
        dataset = fcfg["dataset"]
        src, dcfg = SOURCES[dataset], cfg["datasets"][dataset]
        absent = absent_inputs(dcfg)
        if absent:
            raise MissingInput(f"{dataset} lacks {', '.join(absent[:3])}")
        part = read_partition(cfg, dataset)
        ev = part[part["partition"] == "eval"]
        pos, neg = src.finding_pools(fcfg, ev, dcfg)
        cap = int(fcfg.get("group_cap", smp["group_cap"]))
        pos_cap = int(fcfg.get("positive_group_cap", cap))
        n_max = fcfg["n_per_class"] if "n_per_class" in fcfg else smp["n_per_class"]
        n_pos = capped_count(cap_keys(pos), pos_cap)
        if n_pos < int(fcfg.get("min_eligible_positives", 0)):
            continue
        drawn_pos, drawn_neg, filled = _draw(fcfg, pos, neg, n_max, cap, smp["seed"], pos_cap)
        if fcfg["n_expected"] is not None and (len(drawn_pos) != int(fcfg["n_expected"]) or len(drawn_neg) != int(fcfg["n_expected"])):
            raise ValueError(f"[{set_name}] {finding}: {len(drawn_pos)} per class, while the configuration expects {fcfg['n_expected']}")
        chosen = pd.concat([drawn_pos.assign(label=1), drawn_neg.assign(label=0)], ignore_index=True)
        _check_image_sizes(chosen, dcfg, root, dataset)
        for r in chosen.itertuples(index=False):
            qid = f"{set_name}__{finding}__{r.case_id}"
            questions.append({"question_id": qid, "modality": modality, "finding": finding, "display": fcfg["display"],
                              "noun": fcfg.get("noun", scfg["noun"]), "template": fcfg.get("template", "presence"), "dataset": dataset,
                              "case_id": r.case_id, "key": r.key, "group_id": r.group_id, "split": r.split, "partition": r.partition,
                              "label": int(r.label), "image_path": r.image_path, "input_size": size,
                              **({"z": int(r.z), "decile": int(r.decile)} if hasattr(r, "decile") else {}),
                              **({"stratum": str(r.stratum)} if hasattr(r, "stratum") else {}),
                              **({"neighbor_decile": bool(r.neighbor_decile)} if hasattr(r, "neighbor_decile") else {})})
        extra = [c for c in ("z", "decile", "stratum", "cap_id") if c in pos.columns]
        cols = ["case_id", "key", "group_id", "split", "image_path"] + extra
        pools.append(pd.concat([pos[cols].assign(label=1), neg[cols].assign(label=0)]).assign(finding=finding))
    if not questions:
        raise ValueError(f"[{set_name}] no finding has enough eligible positives")
    return pd.DataFrame(questions), pd.concat(pools, ignore_index=True)


def _absent_for_set(cfg: Dict, set_name: str) -> List[str]:
    return sorted({p for f in set_config(cfg, set_name)["findings"].values() for p in absent_inputs(cfg["datasets"][f["dataset"]])})


def _question_unit(cfg: Dict, root: str, set_name: str, tag: str, status: str) -> None:
    absent = _absent_for_set(cfg, set_name)
    if absent:
        raise MissingInput(f"lacks {', '.join(absent[:3])}")

    def build():
        q, pools = _build_questions(cfg, root, set_name)
        if q["question_id"].duplicated().any():
            raise ValueError(f"[{tag}] {set_name}: duplicated question ids")
        write_csv_atomic(pools, pools_csv(cfg, set_name))
        write_csv_atomic(q, questions_csv(cfg, set_name))
        per = q.groupby("finding")["label"].agg(["size", "sum"]).to_dict("index")
        append_status(status, f"{tag} {set_name}: {len(q)} questions {per}")
    run_build_unit(cfg, f"{tag}__{set_name}", questions_csv(cfg, set_name), _question_params(cfg, set_name), lambda: _questions_valid(cfg, set_name), build)


def main_build_question_sets(cfg_path: str) -> None:
    from tqdm import tqdm
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    status = status_path(cfg, "main_build_question_sets")
    sets = modality_sets(cfg)
    append_status(status, f"main_build_question_sets start: {len(sets)} modalities")
    skipped = []
    for set_name in tqdm(sets, desc="[main_build_question_sets] modalities", unit="modality"):
        try:
            _question_unit(cfg, root, set_name, "main_build_question_sets", status)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    if skipped:
        raise MissingInput(f"[main_build_question_sets] {len(skipped)} modality(ies) not built: {'; '.join(skipped)}")
    append_status(status, "main_build_question_sets done")


def _manifest_params(cfg: Dict, set_name: str) -> Dict:
    stage = "main_build_manifests" if set_name in modality_sets(cfg) else "main_build_external_sets"
    return {"stage": stage, "version": BUILD_VERSION[stage], "modality": set_name, "questions": digest_file(questions_csv(cfg, set_name))}


def _manifest_valid(cfg: Dict, set_name: str) -> bool:
    path = manifest_csv(cfg, set_name)
    if not os.path.exists(path):
        return False
    expected = _expected_rows(set_config(cfg, set_name))
    return expected is None or len(read_csv_defensively(path, usecols=["question_id"])) == expected


def _manifest_unit(cfg: Dict, root: str, set_name: str, tag: str, status: str) -> None:
    if not (os.path.exists(questions_csv(cfg, set_name)) and read_build_params(questions_csv(cfg, set_name)) is not None):
        raise MissingInput("no question set; build it first")
    absent = _absent_for_set(cfg, set_name)
    if absent:
        raise MissingInput(f"lacks {', '.join(absent[:3])}")

    def build():
        q = read_csv_defensively(questions_csv(cfg, set_name), dtype={"key": str, "group_id": str}, keep_default_na=False)
        write_csv_atomic(q, manifest_csv(cfg, set_name))
        append_status(status, f"{tag} {set_name}: {len(q)} questions")
    run_build_unit(cfg, f"{tag}__{set_name}", manifest_csv(cfg, set_name), _manifest_params(cfg, set_name),
                   lambda: _manifest_valid(cfg, set_name), build)


def main_build_manifests(cfg_path: str) -> None:
    from tqdm import tqdm
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    status = status_path(cfg, "main_build_manifests")
    sets = modality_sets(cfg)
    append_status(status, f"main_build_manifests start: {len(sets)} modalities")
    skipped = []
    for set_name in tqdm(sets, desc="[main_build_manifests] modalities", unit="modality"):
        try:
            _manifest_unit(cfg, root, set_name, "main_build_manifests", status)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    if skipped:
        raise MissingInput(f"[main_build_manifests] {len(skipped)} modality(ies) not built: {'; '.join(skipped)}")
    append_status(status, "main_build_manifests done")


def main_build_external_sets(cfg_path: str) -> None:
    from tqdm import tqdm
    full = read_config(cfg_path)
    cfg, root = full[PROJECT], full["datasets_root"]
    status = status_path(cfg, "main_build_external_sets")
    sets = external_sets(cfg)
    append_status(status, f"main_build_external_sets start: {len(sets)} external sets")
    skipped = []
    for set_name in tqdm(sets, desc="[main_build_external_sets] external sets", unit="set"):
        try:
            _question_unit(cfg, root, set_name, "external_questions", status)
            _manifest_unit(cfg, root, set_name, "external_manifests", status)
        except MissingInput as e:
            skipped.append(f"{set_name} ({e})")
    if skipped:
        raise MissingInput(f"[main_build_external_sets] {len(skipped)} of {len(sets)} external set(s) not built: {'; '.join(skipped)}")
    append_status(status, "main_build_external_sets done")
