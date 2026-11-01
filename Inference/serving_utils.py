"""
Inference/serving_utils.py
Created on October 1, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import random
import sys
import time
from typing import Callable, Dict, Optional


_PLACEHOLDER_MARKERS = ("YOUR_", "_HERE", "__FROM_")


def is_placeholder(value: Optional[str]) -> bool:
    return not value or any(m in str(value) for m in _PLACEHOLDER_MARKERS)


def resolve_secret(cfg: Dict, key: str) -> Optional[str]:
    value = cfg.get(key)
    if not is_placeholder(value):
        return value
    return os.environ.get(str(cfg.get(f"{key}_env", "")), None) or None


def require_packages(names, stage: str) -> None:
    import importlib
    failed = []
    for n in names:
        try:
            importlib.import_module(n)
        except ImportError as e:
            failed.append(f"{n} ({type(e).__name__}: {e})")
    if failed:
        raise RuntimeError(f"[{stage}] these imports failed in {sys.executable}: {'; '.join(failed)}.")


class ProviderUnavailable(RuntimeError):

    deterministic = False


def is_rate_limit(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) == 429:
            return True
    except Exception:
        pass
    m = str(e).lower()
    return " 429" in m or "rate limit" in m


def retry_after_s(e: Exception) -> Optional[float]:
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                return float(ra)
    except Exception:
        pass
    return None


def is_timeout(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.APITimeoutError):
            return True
    except Exception:
        pass
    try:
        import httpx
        if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return True
    except Exception:
        pass
    return isinstance(e, TimeoutError) or "timed out" in str(e).lower()


def is_transient_server_error(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) in (502, 503, 504, 429):
            return True
    except Exception:
        pass
    try:
        import httpx
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                          httpx.PoolTimeout, httpx.RemoteProtocolError)):
            return True
    except Exception:
        pass
    if isinstance(e, (ConnectionError, ConnectionRefusedError, TimeoutError)):
        return True
    m = str(e).lower()
    return any(s in m for s in ("connection refused", "connection error", "connection reset", "connect",
                                "timed out", "timeout", "temporarily unavailable", "reset by peer",
                                "bad gateway", "service unavailable", "gateway timeout",
                                " 502", " 503", " 504", " 429"))


def call_timeout_s(serving: Dict, max_tokens: int) -> float:
    return min(float(serving["timeout_request_max_s"]),
               float(serving["timeout_s"]) + int(max_tokens or 0) * float(serving["timeout_per_token_s"]))


def retry_call(fn: Callable[[Optional[float]], object], serving: Dict, call_timeout: float, name: str):
    max_retries = max(1, int(serving["max_retries"]))
    base = float(serving["retry_backoff_s"])
    cap = float(serving["retry_backoff_max_s"])
    connect_budget = float(serving["connect_retry_max_s"])
    t0 = time.time()
    last, attempt, rl_attempt = None, 0, 0
    slept, long_tries, timed_out = 0.0, 0, False
    while True:
        t_attempt = time.time()
        try:
            try:
                return fn(call_timeout)
            except TypeError:
                return fn(None)
        except Exception as e:
            last = e
            attempt_s = time.time() - t_attempt
            if is_rate_limit(e):
                rl_attempt += 1
                if rl_attempt > int(serving["rate_limit_retries"]):
                    break
                wait = retry_after_s(e) or min(30.0, base * (2 ** min(rl_attempt, 5)))
                time.sleep(wait * (0.7 + 0.6 * random.random()))
                continue
            if is_transient_server_error(e):
                if is_timeout(e) and call_timeout and attempt_s >= 0.9 * float(call_timeout):
                    timed_out = True
                    break
                if call_timeout and attempt_s >= 0.5 * float(call_timeout):
                    long_tries += 1
                    if long_tries >= max_retries:
                        break
                if slept >= connect_budget:
                    break
                sleep = min(cap, base * (2 ** min(attempt, 8))) * (0.7 + 0.6 * random.random())
                sleep = min(sleep, connect_budget - slept)
                time.sleep(sleep)
                slept += sleep
                attempt += 1
                continue
            raise
    exc = ProviderUnavailable(f"[{name}] abandoned after {time.time() - t0:.0f}s "
                              f"({'spent its full per-call budget' if timed_out else f'{slept:.0f}s of backoff'}): {last}")
    exc.deterministic = timed_out
    raise exc from last
