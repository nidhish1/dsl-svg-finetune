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


def read_repair_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_submission(path: Path, order: list[str], values: dict[str, str], field: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", field])
        w.writeheader()
        for rid in order:
            w.writerow({"id": rid, field: values.get(rid, "")})


def main() -> None:
    p = argparse.ArgumentParser(description="Apply recovered DSL/SVG rows onto final submissions.")
    p.add_argument("--base-submission", type=Path, default=Path("final/submission.csv"))
    p.add_argument("--base-submission-dsl", type=Path, default=Path("final/submission_dsl.csv"))
    p.add_argument("--repairs-csv", type=Path, default=Path("kitchenSink/raw_repairs.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("dl-spring-2026-svg-generation/test.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("kitchenSink/final_repaired"))
    args = p.parse_args()

    base_svg = read_csv_map(args.base_submission.expanduser().resolve(), "id", "svg")
    base_dsl = read_csv_map(args.base_submission_dsl.expanduser().resolve(), "id", "dsl")
    repair_rows = read_repair_rows(args.repairs_csv.expanduser().resolve())

    order: list[str] = []
    with args.test_csv.expanduser().resolve().open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                order.append(rid)

    merged_svg = dict(base_svg)
    merged_dsl = dict(base_dsl)
    applied = 0
    for row in repair_rows:
        rid = (row.get("id") or "").strip()
        dsl = (row.get("dsl") or "").strip()
        svg = (row.get("svg") or "").strip()
        status = (row.get("status") or "").strip()
        if rid and dsl and svg and status == "recovered":
            merged_dsl[rid] = dsl
            merged_svg[rid] = svg
            applied += 1

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sub = out_dir / "submission.csv"
    out_sub_dsl = out_dir / "submission_dsl.csv"
    write_submission(out_sub, order, merged_svg, "svg")
    write_submission(out_sub_dsl, order, merged_dsl, "dsl")

    nonempty_before = sum(1 for rid in order if base_dsl.get(rid, "").strip())
    nonempty_after = sum(1 for rid in order if merged_dsl.get(rid, "").strip())

    print(f"Wrote {out_sub}")
    print(f"Wrote {out_sub_dsl}")
    print(f"Applied recovered rows: {applied}")
    print(f"DSL non-empty before: {nonempty_before}/{len(order)}")
    print(f"DSL non-empty after: {nonempty_after}/{len(order)}")


if __name__ == "__main__":
    main()

