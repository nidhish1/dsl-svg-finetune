"""Shared helpers: pull DSL v2 JSON out of messy LLM text."""
from __future__ import annotations

import json
import re
from typing import Any, Tuple


def strip_markdown_fences(text: str) -> str:
    t = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return t.replace("```", "").strip()


def extract_dsl_json(text: str) -> Tuple[Any | None, str | None]:
    """
    Return (parsed_dict, single_line_json) for first valid v2-ish object, else (None, None).
    Uses JSONDecoder.raw_decode so a valid object followed by junk is still parsed.
    """
    cleaned = strip_markdown_fences(text)
    start_search = 0
    while True:
        i = cleaned.find("{", start_search)
        if i == -1:
            return None, None
        try:
            obj, end = json.JSONDecoder().raw_decode(cleaned, i)
        except json.JSONDecodeError:
            start_search = i + 1
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
            start_search = i + 1
            continue
        v = obj.get("v")
        if v is not None and v != 2 and v != "2":
            start_search = i + 1
            continue
        fixed = dict(obj)
        fixed["v"] = 2
        fixed.setdefault("canvas", 256)
        line = json.dumps(fixed, separators=(",", ":"), ensure_ascii=True)
        return fixed, line


def extract_dsl_line(text: str) -> str | None:
    _, line = extract_dsl_json(text)
    return line
