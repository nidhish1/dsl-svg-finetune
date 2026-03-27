#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv_map(path: Path, key: str, val: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get(key) or "").strip()
            if k:
                out[k] = (row.get(val) or "").strip()
    return out


def read_test_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if rid:
                rows.append({"id": rid, "prompt": prompt})
    return rows


def write_case(case_dir: Path, prompt: str, dsl: str, svg: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    if dsl:
        (case_dir / "dsl.json").write_text(dsl + "\n", encoding="utf-8")
    if svg:
        (case_dir / "output.svg").write_text(svg, encoding="utf-8")


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Split into old-good, new-good, and bad based on repaired outputs.")
    p.add_argument("--test-csv", type=Path, default=Path("dl-spring-2026-svg-generation/test.csv"))
    p.add_argument("--base-dsl-csv", type=Path, default=Path("final/submission_dsl.csv"))
    p.add_argument("--base-svg-csv", type=Path, default=Path("final/submission.csv"))
    p.add_argument("--repaired-dsl-csv", type=Path, default=Path("kitchenSink/final_repaired/submission_dsl.csv"))
    p.add_argument("--repaired-svg-csv", type=Path, default=Path("kitchenSink/final_repaired/submission.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("kitchenSink2"))
    args = p.parse_args()

    test_rows = read_test_rows(args.test_csv.expanduser().resolve())
    base_dsl = read_csv_map(args.base_dsl_csv.expanduser().resolve(), "id", "dsl")
    base_svg = read_csv_map(args.base_svg_csv.expanduser().resolve(), "id", "svg")
    repaired_dsl = read_csv_map(args.repaired_dsl_csv.expanduser().resolve(), "id", "dsl")
    repaired_svg = read_csv_map(args.repaired_svg_csv.expanduser().resolve(), "id", "svg")

    out_dir = args.out_dir.expanduser().resolve()
    old_good_dir = out_dir / "older_good"
    new_good_dir = out_dir / "new_good"
    bad_dir = out_dir / "bad"
    old_good_dir.mkdir(parents=True, exist_ok=True)
    new_good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    older_good_rows: list[dict[str, str]] = []
    new_good_rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []

    for row in test_rows:
        rid = row["id"]
        prompt = row["prompt"]
        b_dsl = base_dsl.get(rid, "")
        r_dsl = repaired_dsl.get(rid, "")
        b_svg = base_svg.get(rid, "")
        r_svg = repaired_svg.get(rid, "")

        if b_dsl.strip():
            write_case(old_good_dir / rid, prompt, b_dsl, r_svg or b_svg)
            older_good_rows.append({"id": rid, "prompt": prompt})
        elif r_dsl.strip():
            write_case(new_good_dir / rid, prompt, r_dsl, r_svg or b_svg)
            new_good_rows.append({"id": rid, "prompt": prompt})
        else:
            write_case(bad_dir / rid, prompt, "", r_svg or b_svg)
            bad_rows.append({"id": rid, "prompt": prompt})

    write_index(old_good_dir / "index.csv", older_good_rows)
    write_index(new_good_dir / "index.csv", new_good_rows)
    write_index(bad_dir / "index.csv", bad_rows)

    total = len(test_rows)
    split_total = len(older_good_rows) + len(new_good_rows) + len(bad_rows)
    summary = (
        f"total={total}\n"
        f"older_good={len(older_good_rows)}\n"
        f"new_good={len(new_good_rows)}\n"
        f"bad={len(bad_rows)}\n"
        f"split_total={split_total}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary.strip())
    if split_total != total:
        raise SystemExit(f"Count mismatch: split_total={split_total} total={total}")


if __name__ == "__main__":
    main()

