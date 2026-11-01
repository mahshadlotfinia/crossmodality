"""
config/serde.py
https://github.com/mahshadlotfinia
"""

import re

import yaml


_INDENTED_MACHINE_ROOT = re.compile(rb"^[ \t]+(?=(?:ocean_root|home_root|user_home|datasets_root|repo_root)[ \t]*:)", re.M)


def _resolve_once(s: str, flat_vars: dict) -> str:
    return re.sub(r'\$\{([^}]+)\}', lambda m: flat_vars.get(m.group(1).strip(), m.group(0)), s)


def _resolve_all(obj, flat_vars: dict):
    if isinstance(obj, str):
        prev = None
        while prev != obj:
            prev = obj
            obj = _resolve_once(obj, flat_vars)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_all(v, flat_vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_all(item, flat_vars) for item in obj]
    return obj


def load_yaml(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return yaml.safe_load(_INDENTED_MACHINE_ROOT.sub(b"", f.read()))


def read_config(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    flat_vars = {k: v for k, v in cfg.items() if isinstance(v, str)}
    for _ in range(10):
        resolved = {k: _resolve_once(v, flat_vars) for k, v in flat_vars.items()}
        if resolved == flat_vars:
            break
        flat_vars = resolved
    return _resolve_all(cfg, flat_vars)
