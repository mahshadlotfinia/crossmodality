"""
Inference/resume_utils.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager, nullcontext
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class MissingInput(Exception):
    pass


class ClaimedElsewhere(MissingInput):
    pass


class IncompatibleArtifact(Exception):
    pass


def _safe(s) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return parent


def _atomic_write(path: str, write: Callable[[str], None], suffix: str = "") -> None:
    ensure_parent(path)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{suffix}"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _fsync_file(path: str) -> None:
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_atomic(path: str, obj) -> None:
    def _write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=True)
            f.flush()
            os.fsync(f.fileno())
    _atomic_write(path, _write)


def write_csv_atomic(df: pd.DataFrame, path: str, index: bool = False) -> None:
    def _write(tmp):
        df.to_csv(tmp, index=index)
        _fsync_file(tmp)
    _atomic_write(path, _write)


def write_npz_atomic(path: str, compressed: bool = False, **arrays) -> None:
    save = np.savez_compressed if compressed else np.savez
    _atomic_write(path, lambda tmp: save(tmp, **arrays), suffix=".npz")


def write_jsonl_atomic(path: str, records: List[Dict]) -> None:
    def _write(tmp):
        with open(tmp, "w", encoding="ascii") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())
    _atomic_write(path, _write)


def digest_file(path: str) -> Optional[str]:
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.blake2b(f.read(), digest_size=16).hexdigest()
    except OSError:
        return None


def params_path(artifact_path: str) -> str:
    return artifact_path + ".params.json"


def write_build_params(artifact_path: str, params: Dict) -> None:
    write_json_atomic(params_path(artifact_path), params)


def read_build_params(artifact_path: str) -> Optional[Dict]:
    p = params_path(artifact_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _moved_keys(stored: Dict, expected: Dict, prefix: str = "") -> List:
    moved = []
    for k, want in expected.items():
        if k not in stored:
            continue
        was = stored[k]
        name = f"{prefix}{k}"
        if isinstance(was, dict) and isinstance(want, dict):
            moved.extend(_moved_keys(was, want, prefix=f"{name}."))
        elif was != want:
            moved.append((name, was, want))
    return moved


def check_build_params(artifact_path: str, expected: Dict, owner: str, raise_on_mismatch: bool = False,
                       legacy: Optional[Dict] = None) -> bool:
    stored = read_build_params(artifact_path)
    if stored is None:
        return True
    if legacy:
        stored = dict(stored)
        for k, was in legacy.items():
            stored.setdefault(k, was)
    moved = _moved_keys(stored, expected)
    if not moved:
        return True
    lines = "\n  ".join(f"{k}: {was!r} -> {now!r}" for k, was, now in moved)
    msg = f"[{owner}] REBUILD: {os.path.basename(artifact_path)} was built under different parameters.\n  {lines}"
    if raise_on_mismatch:
        raise IncompatibleArtifact(msg)
    return False


def append_jsonl(path: str, record: Dict) -> None:
    ensure_parent(path)
    line = json.dumps(record, ensure_ascii=True, default=_json_default) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("ascii"))
    finally:
        os.close(fd)


_JSONL_WARNED = set()


def read_jsonl(path: str, owner: str = "resume_utils") -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    bad = 0
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
    if bad and path not in _JSONL_WARNED:
        _JSONL_WARNED.add(path)
    return rows


def unit_cache_path(partial_dir: str, group: str, unit: str) -> str:
    return os.path.join(partial_dir, f"{_safe(group)}__{_safe(unit)}.jsonl")


def unit_done(partial_dir: str, group: str, unit: str, require_rows: bool = False) -> bool:
    path = unit_cache_path(partial_dir, group, unit)
    if not os.path.exists(path):
        return False
    return (not require_rows) or len(read_jsonl(path)) > 0


def write_unit_rows(partial_dir: str, group: str, unit: str, rows: List[Dict]) -> None:
    ensure_dir(partial_dir)

    def _write(tmp):
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=True, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())
    _atomic_write(unit_cache_path(partial_dir, group, unit), _write)


def load_group_rows(partial_dir: str, group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(partial_dir):
        return rows
    prefix = f"{_safe(group)}__"
    for name in sorted(os.listdir(partial_dir)):
        if name.startswith(prefix) and name.endswith(".jsonl"):
            rows.extend(read_jsonl(os.path.join(partial_dir, name)))
    return rows


def clear_partial_dir(partial_dir: str) -> None:
    shutil.rmtree(partial_dir, ignore_errors=True)


def run_units_resumable(partial_dir: str, group: str, units: Sequence[str], compute_unit: Callable[[str], List[Dict]],
                        progress_desc: Optional[str] = None, require_rows: bool = False, use_claims: bool = False,
                        status_file: Optional[str] = None, build_params: Optional[Dict] = None,
                        stale_after_s: Optional[float] = None, wait: bool = True) -> pd.DataFrame:
    stale = _CLAIM_STALE_AFTER_S if stale_after_s is None else float(stale_after_s)
    ensure_dir(partial_dir)
    if build_params is not None:
        stamp = os.path.join(partial_dir, f"{_safe(group)}__build_params.json")
        if not check_build_params(stamp, build_params, owner=group):
            if use_claims and not claim_unit(partial_dir, f"{group}__reset", stale):
                raise MissingInput(f"[{group}] another job is clearing partials built under other parameters.")
            try:
                if not check_build_params(stamp, build_params, owner=group):
                    clear_partial_dir(partial_dir)
                    ensure_dir(partial_dir)
            finally:
                if use_claims:
                    release_claim(partial_dir, f"{group}__reset")
        write_build_params(stamp, build_params)
    todo = [u for u in units if not unit_done(partial_dir, group, u, require_rows)]
    n_skipped = 0
    mine = set()

    def _compute(unit):
        nonlocal n_skipped
        tag = f"{group}__{unit}"
        try:
            try:
                with claim_beat(partial_dir, tag) if use_claims else nullcontext():
                    rows = compute_unit(unit)
            except MissingInput:
                n_skipped += 1
                mine.add(unit)
                return
            write_unit_rows(partial_dir, group, unit, rows)
            mine.add(unit)
        finally:
            if use_claims:
                release_claim(partial_dir, tag)
        if status_file:
            append_status(status_file, f"{group}: {unit} wrote {len(rows)} row(s)")

    def _run(candidates, desc):
        from tqdm import tqdm
        for unit in tqdm(candidates, desc=desc, unit="unit"):
            if unit_done(partial_dir, group, unit, require_rows):
                continue
            if use_claims and not claim_unit(partial_dir, f"{group}__{unit}", stale):
                continue
            _compute(unit)

    _run(todo, progress_desc or f"[{group}]")
    if use_claims and not wait:
        held = [u for u in units if u not in mine and not unit_done(partial_dir, group, u, require_rows)]
        if held:
            raise ClaimedElsewhere(f"[{group}] {len(held)} of {len(units)} unit(s) are with another job (first: {held[0]})")
    if use_claims:
        for unit in [u for u in units if u not in mine and not unit_done(partial_dir, group, u, require_rows)]:
            if claim_or_wait(partial_dir, f"{group}__{unit}", lambda unit=unit: unit_done(partial_dir, group, unit, require_rows)):
                _compute(unit)
    elsewhere = [u for u in units if u not in mine and not unit_done(partial_dir, group, u, require_rows)]
    if elsewhere:
        raise MissingInput(f"[{group}] {len(elsewhere)} of {len(units)} unit(s) are unfinished (first: {elsewhere[0]}).")
    out = pd.DataFrame(load_group_rows(partial_dir, group))
    out.attrs["n_skipped"] = n_skipped
    out.attrs["n_units"] = len(units)
    return out


def append_status(status_file: str, message: str) -> None:
    ensure_parent(status_file)
    with open(status_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
        f.flush()
        os.fsync(f.fileno())


def status_path(cfg: Dict, stage: str) -> str:
    return os.path.join(cfg["outputs"]["status_dir"], f"{stage}_status.txt")


_CLAIM_STALE_AFTER_S = 21600.0
_TAKEOVER_LOCK_S = 600.0
_HEARTBEAT_EVERY_S = 30.0
_LIVE_CLAIMS: set = set()
_CLAIM_TOKENS: Dict[str, str] = {}
_LAST_HEARTBEAT: Dict[str, float] = {}


def _claim_path(claims_dir: str, tag: str) -> str:
    return os.path.join(claims_dir, f"claim__{_safe(tag)}.json")


def _slurm_job_state(job_id: str) -> str:
    if not job_id:
        return "unknown"
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "gone" if "invalid job id" in (out.stderr or "").lower() else "unknown"
    return "gone" if out.stdout.strip() == "" else "alive"


def _owner_state(claim: Dict) -> str:
    state = _slurm_job_state(str(claim.get("slurm_job_id", "")))
    if state != "unknown" or claim.get("host") != socket.gethostname():
        return state
    try:
        pid = int(claim.get("pid", 0))
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except OSError:
        return "unknown"
    return "alive"


def _read_claim(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _remove_own_claim(path: str) -> None:
    token = _CLAIM_TOKENS.pop(path, None)
    _LIVE_CLAIMS.discard(path)
    _LAST_HEARTBEAT.pop(path, None)
    if not os.path.exists(path):
        return
    if token is not None and _read_claim(path).get("token") not in (None, token):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _create_claim(path: str, token: str) -> bool:
    tmp = f"{path}.{token}.tmp"
    with open(tmp, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": token}, f, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f, open(tmp) as src:
            f.write(src.read())
        return True
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _take_over(path: str, expired: Dict, token: str) -> bool:
    lock = f"{path}.takeover"
    if not _create_claim(lock, token):
        try:
            if time.time() - os.path.getmtime(lock) > _TAKEOVER_LOCK_S:
                os.remove(lock)
        except OSError:
            pass
        return False
    try:
        if _read_claim(path) != expired:
            return False
        try:
            os.remove(path)
        except OSError:
            return False
        return _create_claim(path, token)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def claim_unit(claims_dir: str, tag: str, stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    ensure_dir(claims_dir)
    path = _claim_path(claims_dir, tag)
    token = uuid.uuid4().hex
    if not _create_claim(path, token):
        claim = _read_claim(path)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        owner = _owner_state(claim)
        if owner == "alive" or (owner == "unknown" and age < stale_after_s):
            return False
        if not claim or not _take_over(path, claim, token):
            return False
    _LIVE_CLAIMS.add(path)
    _CLAIM_TOKENS[path] = token
    return True


def claim_or_wait(claims_dir: str, tag: str, is_done: Callable[[], bool], stale_after_s: float = 1800.0,
                  poll_s: float = 10.0) -> bool:
    last_note, pause = 0.0, 1.0
    while True:
        if is_done():
            return False
        if claim_unit(claims_dir, tag, stale_after_s):
            if is_done():
                release_claim(claims_dir, tag)
                return False
            return True
        if time.time() - last_note > 60.0:
            last_note = time.time()
        time.sleep(pause)
        pause = min(2.0 * pause, poll_s)


@contextmanager
def claim_beat(claims_dir: str, tag: str, every_s: float = 60.0):
    import threading
    path = _claim_path(claims_dir, tag)
    stop = threading.Event()

    def beat():
        while not stop.wait(every_s):
            token = _CLAIM_TOKENS.get(path)
            if token is not None and _read_claim(path).get("token") == token:
                try:
                    os.utime(path, None)
                except OSError:
                    pass
    thread = threading.Thread(target=beat, name=f"claim beat {tag}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def heartbeat_claim(claims_dir: str, tag: str) -> None:
    path = _claim_path(claims_dir, tag)
    now = time.time()
    if now - _LAST_HEARTBEAT.get(path, 0.0) < _HEARTBEAT_EVERY_S:
        return
    _LAST_HEARTBEAT[path] = now
    if os.path.exists(path):
        os.utime(path, None)


def release_claim(claims_dir: str, tag: str) -> None:
    _remove_own_claim(_claim_path(claims_dir, tag))
