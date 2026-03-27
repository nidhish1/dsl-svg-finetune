#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dsl_infer_utils import extract_dsl_json  # noqa: E402
from dsl_v2_to_svg import dsl_to_svg  # noqa: E402


def read_bad_ids(bad_index_csv: Path) -> list[str]:
    ids: list[str] = []
    with bad_index_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                ids.append(rid)
    return ids


def find_raw_path(raw_root: Path, rid: str) -> Path | None:
    direct = raw_root / rid / "raw.txt"
    if direct.is_file():
        return direct
    # Support sharded layout like infer_test_out_shards/shard*/<id>/raw.txt
    matches = list(raw_root.glob(f"*/{rid}/raw.txt"))
    if matches:
        return matches[0]
    return None


def _try_json_candidate(candidate: str) -> str | None:
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        fixed = dict(obj)
        fixed["v"] = 2
        fixed.setdefault("canvas", 256)
        return json.dumps(fixed, separators=(",", ":"), ensure_ascii=True)
    return None


def _extract_balanced_object(text: str) -> list[str]:
    out: list[str] = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1)
    return out


def salvage_dsl(raw_text: str) -> tuple[str | None, str]:
    # 1) Native robust extractor first.
    _obj, line = extract_dsl_json(raw_text)
    if line:
        return line, "extract_dsl_json"

    # 2) Try balanced object candidates.
    for cand in _extract_balanced_object(raw_text):
        fixed = _try_json_candidate(cand)
        if fixed:
            return fixed, "balanced_json"

    # 3) Heuristic cleanup for common model output mistakes.
    cleaned = raw_text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing comma
    for cand in _extract_balanced_object(cleaned):
        fixed = _try_json_candidate(cand)
        if fixed:
            return fixed, "heuristic_cleanup"

    return None, "unrecovered"


def main() -> None:
    p = argparse.ArgumentParser(description="Recover DSL from saved raw.txt for bad IDs.")
    p.add_argument(
        "--bad-index-csv",
        type=Path,
        default=Path("kitchenSink/split_cases/bad/index.csv"),
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("infer_test_out"),
        help="Folder containing per-id subfolders with raw.txt",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("kitchenSink/raw_repairs.csv"),
        help="Repair rows: id,dsl,svg,method,status",
    )
    args = p.parse_args()

    bad_index = args.bad_index_csv.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    out_csv = args.output_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    bad_ids = read_bad_ids(bad_index)
    rows: list[dict[str, str]] = []
    recovered = 0
    missing_raw = 0

    for rid in bad_ids:
        raw_path = find_raw_path(raw_root, rid)
        if raw_path is None:
            rows.append({"id": rid, "dsl": "", "svg": "", "method": "none", "status": "missing_raw"})
            missing_raw += 1
            continue

        raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
        dsl_line, method = salvage_dsl(raw_text)
        if dsl_line:
            try:
                svg = dsl_to_svg(dsl_line)
                rows.append(
                    {"id": rid, "dsl": dsl_line, "svg": svg, "method": method, "status": "recovered"}
                )
                recovered += 1
            except Exception:
                rows.append(
                    {"id": rid, "dsl": "", "svg": "", "method": method, "status": "svg_failed"}
                )
        else:
            rows.append({"id": rid, "dsl": "", "svg": "", "method": method, "status": "unrecovered"})

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dsl", "svg", "method", "status"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_csv}")
    print(f"Bad IDs: {len(bad_ids)}")
    print(f"Recovered: {recovered}")
    print(f"Missing raw.txt: {missing_raw}")


if __name__ == "__main__":
    main()

