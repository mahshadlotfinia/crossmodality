"""
Inference/model_wrappers.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from Inference.parser import (answer_position, classify, classify_option, option_parser_version, p_yes, parse_option_output, parse_output,
                              parser_version)
from Inference.resume_utils import MissingInput
from Inference.serving_utils import ProviderUnavailable, call_timeout_s, require_packages, resolve_secret, retry_call


class NoLogprobs(RuntimeError):
    pass


def _pairs(rows) -> Dict[str, str]:
    return {str(k): str(v) for k, v in rows}


def question_stem(template: str, display: str, noun: str, prompts: Dict) -> str:
    stems = _pairs(prompts["stems"])
    if template not in stems:
        raise KeyError(f"no question stem for template {template!r}; prompts.stems has {sorted(stems)}")
    return stems[template].format(display=display, noun=noun)


def model_prompt(stem: str, modality: str, variant: str, reasoning: bool, prompts: Dict) -> str:
    if variant == "phrasing":
        question = prompts["specialist_question"].format(specialist=_pairs(prompts["specialists"])[modality], stem=stem)
        instruction = prompts["specialist_instruction"]
    else:
        question, instruction = stem, prompts["answer_instruction"]
    if reasoning:
        return prompts["reasoning_template"].format(question=question)
    return f"{question} {instruction}"


def benchmark_prompt(question: str, options: Optional[List[List[str]]], reasoning: bool, prompts: Dict) -> str:
    if not options:
        return prompts["reasoning_template"].format(question=question) if reasoning else f"{question} {prompts['answer_instruction']}"
    body = question + "\n" + "\n".join(prompts["option_line"].format(letter=letter, text=text) for letter, text in options)
    return prompts["reasoning_template_options"].format(question=body) if reasoning else f"{body}\n{prompts['options_instruction']}"


def png_data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


IMAGE_CHECK_QUESTION = "Is there a white square in this image?"


def write_test_image(path: str) -> None:
    import numpy as np
    from PIL import Image
    arr = np.full((512, 512, 3), 70, np.uint8)
    arr[192:320, 192:320] = 255
    Image.fromarray(arr).save(path)


def image_token_problem(with_image: Optional[int], without_image: Optional[int], minimum: int) -> Optional[str]:
    if with_image is None or without_image is None:
        return "the server reports no prompt tokens, so whether the image reaches the model cannot be checked"
    added = int(with_image) - int(without_image)
    if added < int(minimum):
        return (f"the 512 x 512 test image adds {added} prompt tokens, fewer than serving.min_image_tokens = {minimum}, so the server shrinks "
                f"or drops the image. The server administrator has to check the image settings of this model")
    return None


def make_client(cfg: Dict, spec: Optional[Dict] = None):
    require_packages(["openai", "httpx"], "serving client")
    import httpx
    from openai import OpenAI
    spec = spec or {}
    base_url = spec.get("llm_api_base_url") or resolve_secret(cfg, "llm_api_base_url")
    if not base_url:
        raise RuntimeError("no serving base URL: set llm_api_base_url in config.yaml or the variable named by llm_api_base_url_env")
    serving = cfg["serving"]
    conc = max(1, int(serving["concurrency"]))
    limits = httpx.Limits(max_connections=conc + 8, max_keepalive_connections=conc + 8)
    http = httpx.Client(verify=bool(serving["verify_ssl"]), timeout=float(serving["timeout_s"]), limits=limits)
    return OpenAI(base_url=base_url, api_key=spec.get("llm_api_key") or resolve_secret(cfg, "llm_api_key") or "EMPTY", http_client=http, max_retries=0), base_url


_SILENT_UNTIL: Dict[str, float] = {}
_SILENT_FOR_S = 300.0


def served_ids(client, base_url: str, serving: Dict, budget_s: float = 60.0) -> List[str]:
    if time.time() < _SILENT_UNTIL.get(base_url, 0.0):
        raise MissingInput(f"the serving system at {base_url} did not answer from this machine in the last {_SILENT_FOR_S / 60:.0f} min; "
                           f"run it again once it answers")
    short = dict(serving, connect_retry_max_s=min(float(budget_s), float(serving["connect_retry_max_s"])))

    def _list(timeout):
        return client.models.list(timeout=timeout) if timeout else client.models.list()
    try:
        listed = retry_call(_list, short, float(serving["timeout_s"]), "served models")
    except ProviderUnavailable as e:
        _SILENT_UNTIL[base_url] = time.time() + _SILENT_FOR_S
        raise MissingInput(f"the serving system at {base_url} did not answer from this machine ({e}); run it again once it answers") from e
    return sorted(m.id for m in listed.data)


def _token_rows(resp) -> List[Tuple[str, List[Tuple[str, float]]]]:
    try:
        content = resp.choices[0].logprobs.content
    except AttributeError:
        return []
    return [(str(c.token), [(str(t.token), float(t.logprob)) for t in (c.top_logprobs or [])]) for c in (content or [])]


class ServedModel:

    def __init__(self, model_key: str, cfg: Dict):
        self.model_key, self.cfg = model_key, cfg
        self.spec = cfg["models"][model_key]
        self.reasoning = bool(self.spec["reasoning"])
        self.serving, self.gen, self.pcfg, self.ocfg = cfg["serving"], cfg["generation"], cfg["parser"], cfg["option_parser"]
        extra = {}
        if "enable_thinking" in self.spec:
            extra["chat_template_kwargs"] = {"enable_thinking": bool(self.spec["enable_thinking"])}
        if "mm_processor_kwargs" in self.spec:
            extra["mm_processor_kwargs"] = dict(self.spec["mm_processor_kwargs"])
        if "reasoning_effort" in self.spec:
            extra["reasoning_effort"] = str(self.spec["reasoning_effort"])
        self.extra_body = extra or None
        self._client, self._base_url = None, None
        self._long_calls = threading.BoundedSemaphore(max(1, int(self.serving["long_call_concurrency"])))

    @property
    def budget(self) -> List[int]:
        if self.reasoning:
            return [int(self.gen["reasoning_max_new_tokens"]), 0]
        return [int(self.gen["max_new_tokens"]), int(self.gen["regenerate_max_new_tokens"])]

    def budget_for(self, options: Optional[List[List[str]]]) -> List[int]:
        if options and not self.reasoning:
            return [int(self.gen["max_new_tokens"]), int(self.gen["option_regenerate_max_new_tokens"])]
        return self.budget

    def request_digest(self, prompt: str) -> str:
        blob = json.dumps({"prompt": prompt, "served_name": self.spec["served_name"], "extra_body": self.extra_body,
                           "temperature": float(self.gen["temperature"]), "top_k": int(self.serving["logprobs_top_k"])},
                          sort_keys=True, ensure_ascii=True)
        return hashlib.blake2b(blob.encode("ascii"), digest_size=8).hexdigest()

    def _ensure_client(self):
        if self._client is None:
            self._client, self._base_url = make_client(self.cfg, self.spec)

    def require_served(self) -> None:
        self._ensure_client()
        try:
            ids = served_ids(self._client, self._base_url, self.serving)
        except MissingInput as e:
            if "llm_api_base_url" not in self.spec:
                raise
            raise MissingInput(f"{self.model_key} is served from its own address {self._base_url} (its config entry), and no server answered there: "
                               f"{e}. Start a vLLM server for {self.spec['served_name']} on this machine first") from e
        if self.spec["served_name"] not in ids:
            raise MissingInput(f"{self.model_key} ({self.spec['served_name']}) is not served now; load it on the server and run it again "
                               f"(served: {', '.join(ids) or 'none'})")

    def served_now(self) -> bool:
        try:
            self.require_served()
            return True
        except MissingInput:
            return False

    def require_image_input(self) -> None:
        prompt = model_prompt(IMAGE_CHECK_QUESTION, "", "default", self.reasoning, self.cfg["prompts"])
        with tempfile.TemporaryDirectory(prefix="image_check_") as tmp:
            path = os.path.join(tmp, "test.png")
            write_test_image(path)
            try:
                with_image = self._call(png_data_url(path), prompt, 1)["prompt_tokens"]
                without_image = self._call(None, prompt, 1)["prompt_tokens"]
            except (NoLogprobs, MissingInput):
                raise
            except ProviderUnavailable as e:
                raise MissingInput(f"{self.model_key}: the image-token check did not complete ({e}); run this line again") from e
            except Exception as e:
                if not self.served_now():
                    raise MissingInput(f"{self.model_key} is no longer served ({type(e).__name__}); load it and run it again") from e
                raise
        problem = image_token_problem(with_image, without_image, int(self.serving["min_image_tokens"]))
        if problem:
            raise MissingInput(f"{self.model_key} ({self.spec['served_name']}): {problem}. Nothing is written")

    def _call(self, image_url: Optional[str], prompt: str, max_tokens: int) -> Dict:
        self._ensure_client()
        content = prompt if image_url is None else [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": prompt}]
        kwargs = dict(model=self.spec["served_name"], messages=[{"role": "user", "content": content}], max_tokens=int(max_tokens),
                      temperature=float(self.gen["temperature"]), logprobs=True, top_logprobs=int(self.serving["logprobs_top_k"]))
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        def _create(timeout):
            if timeout is not None:
                return self._client.chat.completions.create(timeout=timeout, **kwargs)
            return self._client.chat.completions.create(**kwargs)

        t0 = time.time()
        resp = retry_call(_create, self.serving, call_timeout_s(self.serving, max_tokens), self.model_key)
        choice = resp.choices[0]
        raw = choice.message.content or ""
        rows = _token_rows(resp)
        if raw and not rows:
            raise NoLogprobs(f"[{self.model_key}] the server answered without log-probabilities")
        usage = getattr(resp, "usage", None)
        return {"raw": raw, "rows": rows, "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None),
                "elapsed_s": round(time.time() - t0, 3), "response_id": getattr(resp, "id", None), "served_model": getattr(resp, "model", None)}

    def parse(self, raw: Optional[str], options: Optional[List[List[str]]]) -> int:
        return parse_option_output(raw, options, self.reasoning, self.pcfg, self.ocfg) if options else parse_output(raw, self.reasoning, self.pcfg)

    def answer(self, image_path: Optional[str], prompt: str, options: Optional[List[List[str]]] = None) -> Dict:
        url = None if image_path is None else png_data_url(image_path)
        first, second = self.budget_for(options)
        out = self._call(url, prompt, first)
        parsed = self.parse(out["raw"], options)
        first_raw = None
        if parsed == -1 and second > first:
            first_raw = out["raw"]
            if second > int(self.gen["regenerate_max_new_tokens"]):
                with self._long_calls:
                    out = self._call(url, prompt, second)
            else:
                out = self._call(url, prompt, second)
            parsed = self.parse(out["raw"], options)
        tokens = [t for t, _ in out["rows"]]
        position = answer_position(tokens, self.reasoning, self.pcfg)
        candidates = out["rows"][position][1] if position is not None else []
        if options:
            prob, source, version = None, "options", option_parser_version(self.pcfg, self.ocfg)
            cls = classify_option(out["raw"], parsed, self.reasoning, out["finish_reason"], self.pcfg)
        else:
            (prob, source), version = p_yes(candidates, position, self.reasoning, self.pcfg), parser_version(self.pcfg)
            cls = classify(out["raw"], parsed, self.reasoning, out["finish_reason"], self.pcfg)
        rec = {"raw_answer": out["raw"], "first_raw_answer": first_raw, "regenerated": first_raw is not None, "parsed_answer": parsed,
               "answer_class": cls, "p_yes": prob, "p_yes_source": source, "answer_position": position,
               "answer_token": tokens[position] if position is not None else None, "candidates": [[t, lp] for t, lp in candidates],
               "parser_version": version, "budget": [first, second], "max_tokens": second if first_raw is not None else first}
        rec.update({k: out[k] for k in ("finish_reason", "prompt_tokens", "completion_tokens", "elapsed_s", "response_id", "served_model")})
        if self.reasoning:
            rec["tokens"] = tokens
        return rec

    def answer_batch(self, items: List[Dict]) -> List[Optional[Dict]]:
        out: List[Optional[Dict]] = [None] * len(items)

        def _do(i):
            try:
                return i, self.answer(items[i]["image_path"], items[i]["prompt"], items[i].get("options"))
            except ProviderUnavailable:
                return i, None

        with ThreadPoolExecutor(max_workers=max(1, int(self.serving["concurrency"]))) as ex:
            for i, res in ex.map(_do, range(len(items))):
                out[i] = res
        return out
