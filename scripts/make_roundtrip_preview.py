#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from dsl_v2_to_svg import dsl_to_svg


def read_svg_by_id(train_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            svg = (row.get("svg") or "").strip()
            if rid and svg:
                out[rid] = svg
    return out


def read_dsl_rows(dsl_jsonl: Path, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with dsl_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            obj = json.loads(line)
            rows.append(
                {
                    "id": obj.get("id", ""),
                    "prompt": obj.get("prompt", ""),
                    "dsl": obj.get("dsl", ""),
                }
            )
    return rows


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_html(rows: List[Dict[str, str]], original_svg_by_id: Dict[str, str]) -> str:
    cards: List[str] = []
    for i, r in enumerate(rows, 1):
        rid = r["id"]
        prompt = r["prompt"]
        dsl = r["dsl"]
        orig = original_svg_by_id.get(rid, "")
        recon = ""
        try:
            recon = dsl_to_svg(dsl)
        except Exception:
            recon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><text x="8" y="20">recon failed</text></svg>'

        card = f"""
<section class="card">
  <h3>Sample {i}: {html_escape(rid)}</h3>
  <p class="prompt">{html_escape(prompt)}</p>
  <div class="grid">
    <div class="col">
      <h4>Original SVG</h4>
      <div class="panel">{orig}</div>
    </div>
    <div class="col">
      <h4>DSL v2</h4>
      <pre class="dsl">{html_escape(dsl)}</pre>
    </div>
    <div class="col">
      <h4>Reconstructed SVG</h4>
      <div class="panel">{recon}</div>
    </div>
  </div>
</section>
"""
        cards.append(card)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SVG -> DSL -> SVG Preview</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f8f9fb; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 18px; }}
    .prompt {{ margin-top: -4px; color: #333; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
    .panel {{ border: 1px solid #ddd; background: #fff; min-height: 280px; display: flex; align-items: center; justify-content: center; }}
    .panel svg {{ width: 256px; height: 256px; }}
    .dsl {{ border: 1px solid #ddd; background: #fafafa; padding: 8px; height: 280px; overflow: auto; font-size: 11px; }}
    h3, h4 {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Roundtrip Preview (Original -> DSL -> Reconstructed)</h1>
  {''.join(cards)}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side preview for SVG->DSL->SVG.")
    parser.add_argument("--train-csv", required=True, help="Path to train.csv with original SVG")
    parser.add_argument("--dsl-jsonl", required=True, help="Path to DSL v2 JSONL")
    parser.add_argument("--samples", type=int, default=6, help="How many samples to include")
    parser.add_argument("--output-html", required=True, help="Output HTML preview file")
    args = parser.parse_args()

    orig = read_svg_by_id(Path(args.train_csv))
    rows = read_dsl_rows(Path(args.dsl_jsonl), args.samples)
    html = build_html(rows, orig)
    Path(args.output_html).write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
