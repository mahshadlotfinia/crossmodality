"""
Inference/parser.py
Created on October 2, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import hashlib
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple


PARSER_VERSION = 1
_STRIP = ".,!?:;*\"'`()[]"


def parser_version(pcfg: Dict) -> str:
    blob = json.dumps(pcfg, sort_keys=True, ensure_ascii=True).encode("ascii")
    return f"{PARSER_VERSION}-{hashlib.blake2b(blob, digest_size=6).hexdigest()}"


def clean_text(text: Optional[str]) -> str:
    text = "" if text is None else str(text)
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()


def _tokens(pcfg: Dict, key: str) -> List[str]:
    toks = pcfg[key]
    if not all(isinstance(t, str) for t in toks):
        raise TypeError(f"parser.{key} contains a non-string token {toks}; quote yes, no, true, and false in config.yaml")
    return toks


def _tag_start(text: str, pcfg: Dict) -> int:
    close = text.find(pcfg["think_tag_close"])
    after = close + len(pcfg["think_tag_close"]) if close >= 0 else 0
    i = text.find(pcfg["answer_tag_open"], after)
    return text.find(pcfg["answer_tag_open"]) if i < 0 and close < 0 else i


def answer_block(text: Optional[str], pcfg: Dict) -> Optional[str]:
    text = clean_text(text)
    i = _tag_start(text, pcfg)
    if i < 0:
        return None
    rest = text[i + len(pcfg["answer_tag_open"]):]
    j = rest.find(pcfg["answer_tag_close"])
    return (rest if j < 0 else rest[:j]).strip()


def is_abstention(text: Optional[str], pcfg: Dict) -> bool:
    low = clean_text(text).lower()
    return any(p in low for p in pcfg["abstention_phrases"])


def parse_answer(text: Optional[str], pcfg: Dict) -> int:
    text = clean_text(text)
    if not text:
        return -1
    word = text.split()[0].lower().strip(_STRIP)
    if word in _tokens(pcfg, "affirmative_tokens"):
        return 1
    if word in _tokens(pcfg, "negative_tokens"):
        return 0
    if is_abstention(text, pcfg):
        return -1
    head = text.lower()[: int(pcfg["head_chars"])]
    has_yes, has_no = re.search(r"\byes\b", head) is not None, re.search(r"\bno\b", head) is not None
    if has_yes != has_no:
        return 1 if has_yes else 0
    return -1


def parse_output(raw: Optional[str], reasoning: bool, pcfg: Dict) -> int:
    if not reasoning:
        return parse_answer(raw, pcfg)
    block = answer_block(raw, pcfg)
    return -1 if block is None else parse_answer(block, pcfg)


def classify(raw: Optional[str], parsed: int, reasoning: bool, finish_reason: Optional[str], pcfg: Dict) -> str:
    if parsed == 1:
        return "yes"
    if parsed == 0:
        return "no"
    if not clean_text(raw):
        return "empty"
    target = answer_block(raw, pcfg) if reasoning else raw
    if target is not None and is_abstention(target, pcfg):
        return "abstain"
    if finish_reason == "length":
        return "truncated"
    return "unparsed"


def answer_position(tokens: Sequence[str], reasoning: bool, pcfg: Dict) -> Optional[int]:
    if not tokens:
        return None
    if not reasoning:
        return 0
    text, ends = "", []
    for tok in tokens:
        text += str(tok)
        ends.append(len(text))
    start = _tag_start(text, pcfg)
    if start < 0:
        return None
    tag_end = start + len(pcfg["answer_tag_open"])
    if not text[tag_end:].strip() or text[tag_end:].lstrip().startswith(pcfg["answer_tag_close"]):
        return None
    for i, end in enumerate(ends):
        if end <= tag_end:
            continue
        begin = ends[i - 1] if i else 0
        if text[max(begin, tag_end):end].strip():
            return i
    return None


def _normalize(token: str) -> str:
    return str(token).replace("Ġ", " ").replace("▁", " ")


def p_yes(candidates: Optional[Sequence[Tuple[str, float]]], position: Optional[int], reasoning: bool, pcfg: Dict) -> Tuple[float, str]:
    if position is None:
        return 0.5, "no_answer_position"
    yes, no = set(_tokens(pcfg, "yes_tokens")), set(_tokens(pcfg, "no_tokens"))
    py = sum(math.exp(float(lp)) for tok, lp in (candidates or []) if _normalize(tok) in yes)
    pn = sum(math.exp(float(lp)) for tok, lp in (candidates or []) if _normalize(tok) in no)
    if py + pn <= 0:
        return 0.5, "none_in_top"
    return float(py / (py + pn)), "answer_tag" if reasoning else "first_token"


def reparse(rec: Dict, reasoning: bool, pcfg: Dict) -> Optional[Dict]:
    raw = rec.get("raw_answer")
    position = answer_position(rec.get("tokens") or [], True, pcfg) if reasoning else rec.get("answer_position")
    if position != rec.get("answer_position"):
        return None
    parsed = parse_output(raw, reasoning, pcfg)
    prob, source = p_yes([(t, lp) for t, lp in rec.get("candidates") or []], position, reasoning, pcfg)
    return {"parsed_answer": parsed, "answer_class": classify(raw, parsed, reasoning, rec.get("finish_reason"), pcfg),
            "p_yes": prob, "p_yes_source": source, "parser_version": parser_version(pcfg)}


OPTION_PARSER_VERSION = 5


def option_parser_version(pcfg: Dict, ocfg: Dict) -> str:
    blob = json.dumps({"parser": pcfg, "options": ocfg}, sort_keys=True, ensure_ascii=True).encode("ascii")
    return f"o{OPTION_PARSER_VERSION}-{hashlib.blake2b(blob, digest_size=6).hexdigest()}"


_STATEMENT = re.compile(r"(?:\b(?i:(?:correct|final|best)\s+)?(?i:answer|option|choice)\s*(?:is\s*:?|:|would\s+be\s*:?)|\bis\s+(?=(?i:option|choice)\b))"
                        r"\s*[*_\s]*(?:(?i:option|choice)\s*[*_\s]*)?(?:\(([A-Ha-h])\)|([A-H])(?![A-Za-z]))")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?([A-Ha-h])\)?(?![A-Za-z])")


def _bold_option(letter: str, text: str):
    body = re.escape(str(text).strip().rstrip("."))
    return re.compile(r"\*\*\s*(?:\((?i:" + re.escape(letter) + r")\)|" + re.escape(letter) + r"[.):])\s*(?i:" + body + r")\s*[.,;:]?\s*\*\*")


def parse_option(text: Optional[str], options: Sequence[Sequence[str]], pcfg: Dict, ocfg: Dict) -> int:
    text = clean_text(text)
    if not text:
        return -1
    letters = [str(o[0]) for o in options]
    word = text.split()[0].strip(_STRIP)
    if word in letters:
        return letters.index(word)
    bare = text.strip(_STRIP).upper()
    if len(bare) == 1 and bare in letters:
        return letters.index(bare)
    stated = [(m.start(), (m.group(1) or m.group(2)).upper()) for m in _STATEMENT.finditer(text)]
    stated += [(m.start(), m.group(1).upper()) for m in _BOXED.finditer(text)]
    if stated and max(stated)[1] in letters:
        return letters.index(max(stated)[1])
    last = re.fullmatch(r"\(?([A-H])\)?(?:[.:)]\s*(.*))?", text.splitlines()[-1].strip().strip("*_ "))
    if last and last.group(1) in letters:
        rest = (last.group(2) or "").strip("*_ .:").lower()
        if not rest or rest == str(options[letters.index(last.group(1))][1]).strip().strip(".").lower():
            return letters.index(last.group(1))
    named = re.search(r"\b(?i:option|answer(?:\s+is)?)\s*:?\s*\(?([A-H])\)?(?![A-Za-z])", text[: int(ocfg["head_chars"])])
    if named and named.group(1) in letters:
        return letters.index(named.group(1))
    bold = {L for L, o in zip(letters, options) if _bold_option(L, o[1]).search(text)}
    if len(bold) == 1:
        return letters.index(bold.pop())
    if is_abstention(text, pcfg):
        return -1
    low = text.lower()
    starts = [i for i, o in enumerate(options) if str(o[1]).strip() and low.startswith(str(o[1]).strip().lower())]
    if len(starts) == 1:
        return starts[0]
    found = {c for c in re.findall(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", text[: int(ocfg["head_chars"])]) if c in letters}
    return letters.index(found.pop()) if len(found) == 1 else -1


def parse_option_output(raw: Optional[str], options: Sequence[Sequence[str]], reasoning: bool, pcfg: Dict, ocfg: Dict) -> int:
    if not reasoning:
        return parse_option(raw, options, pcfg, ocfg)
    block = answer_block(raw, pcfg)
    return -1 if block is None else parse_option(block, options, pcfg, ocfg)


def classify_option(raw: Optional[str], parsed: int, reasoning: bool, finish_reason: Optional[str], pcfg: Dict) -> str:
    return "option" if parsed >= 0 else classify(raw, -1, reasoning, finish_reason, pcfg)


def reparse_option(rec: Dict, reasoning: bool, pcfg: Dict, ocfg: Dict) -> Dict:
    raw = rec.get("raw_answer")
    parsed = parse_option_output(raw, rec.get("options") or [], reasoning, pcfg, ocfg)
    return {"parsed_answer": parsed, "answer_class": classify_option(raw, parsed, reasoning, rec.get("finish_reason"), pcfg),
            "parser_version": option_parser_version(pcfg, ocfg)}
