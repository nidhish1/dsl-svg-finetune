#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def read_prompts(test_csv: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if rid:
                prompts[rid] = prompt
    return prompts


def read_submission(submission_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with submission_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            svg = (row.get("svg") or "").strip()
            if rid:
                rows.append({"id": rid, "svg": svg})
    return rows


def build_html(rows: list[dict[str, str]], prompts: dict[str, str]) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Prompt + Rendered SVG Preview</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;}",
        ".row{display:grid;grid-template-columns:1fr 300px;gap:16px;align-items:start;",
        "padding:14px 0;border-bottom:1px solid #ddd;}",
        ".prompt{margin:0;line-height:1.4;}",
        ".svgbox{border:1px solid #ccc;padding:8px;min-height:280px;display:flex;",
        "align-items:center;justify-content:center;background:#fff;}",
        ".svgbox svg{max-width:256px;max-height:256px;height:auto;}",
        ".id{font-size:12px;color:#666;margin:0 0 6px 0;}",
        "</style></head><body>",
        "<h1>Prompt + Rendered SVG (100 samples)</h1>",
    ]

    for i, row in enumerate(rows, start=1):
        rid = row["id"]
        prompt = prompts.get(rid, "")
        svg = row["svg"]
        parts.append(
            "<section class='row'>"
            f"<div><p class='id'>{i}. {html.escape(rid)}</p>"
            f"<p class='prompt'>{html.escape(prompt)}</p></div>"
            f"<div class='svgbox'>{svg}</div>"
            "</section>"
        )

    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Create prompt+SVG preview HTML from submission.")
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("dl-spring-2026-svg-generation/test.csv"),
        help="Test CSV with id,prompt",
    )
    p.add_argument(
        "--submission-csv",
        type=Path,
        default=Path("final/submission.csv"),
        help="Submission CSV with id,svg",
    )
    p.add_argument("--limit", type=int, default=100, help="Number of rows to visualize")
    p.add_argument(
        "--output-html",
        type=Path,
        default=Path("kitchenSink/prompt_svg_preview_100.html"),
        help="Output HTML path",
    )
    args = p.parse_args()

    prompts = read_prompts(args.test_csv.expanduser().resolve())
    submission_rows = read_submission(args.submission_csv.expanduser().resolve())
    rows = submission_rows[: max(0, args.limit)]
    html_text = build_html(rows, prompts)

    out = args.output_html.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

