#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BLANK_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' "
    "viewBox='0 0 256 256'><rect width='256' height='256' fill='white'/></svg>"
)


def read_test_prompts(test_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                out[rid] = (row.get("prompt") or "").strip()
    return out


def read_submission(submission_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with submission_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                out[rid] = (row.get("svg") or "").strip()
    return out


def read_submission_dsl(submission_dsl_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with submission_dsl_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                out[rid] = (row.get("dsl") or "").strip()
    return out


def is_valid_dsl(dsl_text: str) -> bool:
    if not dsl_text:
        return False
    try:
        obj = json.loads(dsl_text)
    except Exception:
        return False
    return isinstance(obj, dict) and isinstance(obj.get("items"), list)


def write_case(case_dir: Path, prompt: str, dsl: str, svg: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (case_dir / "output.svg").write_text(svg, encoding="utf-8")
    if dsl:
        (case_dir / "dsl.json").write_text(dsl + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Split final outputs into good/bad case folders.")
    p.add_argument("--test-csv", type=Path, default=Path("dl-spring-2026-svg-generation/test.csv"))
    p.add_argument("--submission-csv", type=Path, default=Path("final/submission.csv"))
    p.add_argument("--submission-dsl-csv", type=Path, default=Path("final/submission_dsl.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("kitchenSink/split_cases"))
    args = p.parse_args()

    prompts = read_test_prompts(args.test_csv.expanduser().resolve())
    svg_by_id = read_submission(args.submission_csv.expanduser().resolve())
    dsl_by_id = read_submission_dsl(args.submission_dsl_csv.expanduser().resolve())

    out_dir = args.out_dir.expanduser().resolve()
    good_dir = out_dir / "good"
    bad_dir = out_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    good_rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []

    for rid, prompt in prompts.items():
        dsl = dsl_by_id.get(rid, "")
        svg = svg_by_id.get(rid, "")
        dsl_ok = is_valid_dsl(dsl)
        svg_ok = bool(svg and svg != BLANK_SVG)
        if dsl_ok and svg_ok:
            target = good_dir / rid
            write_case(target, prompt, dsl, svg)
            good_rows.append({"id": rid, "prompt": prompt})
        else:
            target = bad_dir / rid
            write_case(target, prompt, dsl, svg or BLANK_SVG)
            bad_rows.append(
                {
                    "id": rid,
                    "prompt": prompt,
                    "dsl_present": str(bool(dsl)),
                    "svg_nonblank": str(svg_ok),
                }
            )

    with (good_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt"])
        w.writeheader()
        w.writerows(good_rows)

    with (bad_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "dsl_present", "svg_nonblank"])
        w.writeheader()
        w.writerows(bad_rows)

    print(f"Wrote good cases: {len(good_rows)} -> {good_dir}")
    print(f"Wrote bad cases: {len(bad_rows)} -> {bad_dir}")


if __name__ == "__main__":
    main()

