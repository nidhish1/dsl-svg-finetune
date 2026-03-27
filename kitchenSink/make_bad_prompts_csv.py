#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Build id,prompt CSV for bad cases only.")
    p.add_argument(
        "--bad-index-csv",
        type=Path,
        default=Path("kitchenSink/split_cases/bad/index.csv"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("kitchenSink/bad_prompts.csv"),
    )
    args = p.parse_args()

    bad_idx = args.bad_index_csv.expanduser().resolve()
    out_csv = args.output_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with bad_idx.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if rid:
                rows.append({"id": rid, "prompt": prompt})

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_csv} with {len(rows)} rows")


if __name__ == "__main__":
    main()

