"""
Inference/model_facts.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, List

from config.serde import read_config
from data_loader.build_utils import write_csv_if_changed
from Inference.resume_utils import MissingInput, run_units_resumable, status_path
from Inference.serving_utils import require_packages, resolve_secret


PROJECT = "Crossmodality"
FLOAT_DTYPES = ("BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2")
FACTS_VERSION = 1


def facts_csv(cfg: Dict) -> str:
    return os.path.join(cfg["outputs"]["models_dir"], "model_facts.csv")


def _facts(model_key: str, spec: Dict, token) -> List[Dict]:
    from huggingface_hub import get_safetensors_metadata, model_info
    from huggingface_hub.errors import GatedRepoError
    dated_by = spec.get("date_from", spec["hf_id"])
    try:
        info = model_info(spec["hf_id"], token=token)
        meta = get_safetensors_metadata(spec["hf_id"], revision=info.sha, token=token)
        dated = info if dated_by == spec["hf_id"] else model_info(dated_by, token=token)
    except GatedRepoError as e:
        raise MissingInput(f"{spec['hf_id']} is gated; accept its terms on Hugging Face and set hf_token (or {PROJECT}.hf_token_env)") from e
    counts = {str(k): int(v) for k, v in meta.parameter_count.items()}
    n = sum(v for k, v in counts.items() if k in FLOAT_DTYPES)
    return [{"model": model_key, "hf_id": spec["hf_id"], "revision": info.sha, "parameters": n,
             "parameters_by_dtype": json.dumps(counts, sort_keys=True), "n_files": len(meta.files_metadata),
             "dated_by": dated_by, "created_at": dated.created_at.date().isoformat()}]


def main_model_facts(cfg_path: str) -> None:
    require_packages(["huggingface_hub"], "model_facts")
    cfg = read_config(cfg_path)[PROJECT]
    token = resolve_secret(cfg, "hf_token")
    models = list(cfg["models"])
    params = {"stage": "model facts", "version": FACTS_VERSION,
              "repos": {k: [cfg["models"][k]["hf_id"], cfg["models"][k].get("date_from")] for k in models}}
    rows = run_units_resumable(os.path.join(cfg["outputs"]["models_dir"], "partials"), "model_facts", models,
                               lambda m: _facts(m, cfg["models"][m], token), progress_desc="[main_model_facts] model facts", require_rows=True,
                               use_claims=True, status_file=status_path(cfg, "main_model_facts_model_facts"), build_params=params)
    if rows.attrs.get("n_skipped"):
        raise MissingInput(f"[main_model_facts] model facts: {rows.attrs['n_skipped']} of {len(models)} model(s) could not be read; model_facts.csv is not written")
    table = rows.set_index("model").loc[models].reset_index()
    write_csv_if_changed(table, facts_csv(cfg))
