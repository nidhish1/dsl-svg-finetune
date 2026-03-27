#!/usr/bin/env python3
"""
Build gallery HTML from batch output (or any folder tree with dsl.json).
No GPU. Run on your laptop after: scp -r host:.../infer_batch_out .

  python scripts/preview_dsl_folder.py --dir ~/Downloads/infer_batch_out
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dsl_v2_to_svg import dsl_to_svg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Gallery from dsl.json files")
    p.add_argument("--dir", type=Path, required=True, help="infer_batch_out (or similar)")
    p.add_argument("--output", type=Path, default=None, help="gallery.html path")
    args = p.parse_args()

    root = args.dir.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    dsl_files = sorted(root.glob("*/dsl.json"))
    if not dsl_files:
        dsl_files = sorted(root.rglob("dsl.json"))

    out_html = args.output or (root / "gallery_local.html")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>DSL gallery</title>",
        "<style>body{font-family:system-ui;margin:24px;} .row{display:grid;grid-template-columns:1fr 280px;gap:16px;",
        "border-bottom:1px solid #ddd;padding:16px 0;} pre{background:#f5f5f5;font-size:10px;max-height:180px;overflow:auto;}",
        "svg{max-width:256px;border:1px solid #ccc;}</style></head><body><h1>Local gallery</h1>",
    ]

    for dsl_path in dsl_files:
        rel = dsl_path.relative_to(root)
        prompt_txt = dsl_path.parent / "prompt.txt"
        prompt = prompt_txt.read_text(encoding="utf-8").strip() if prompt_txt.is_file() else str(rel)
        dsl_line = dsl_path.read_text(encoding="utf-8").strip()
        err = ""
        svg = ""
        try:
            json.loads(dsl_line)
            svg = dsl_to_svg(dsl_line)
        except Exception as e:
            err = str(e)
        err_html = f"<p style='color:#a00'>{html_mod.escape(err)}</p>" if err else ""
        parts.append(
            f"<section class='row'><div><h3>{html_mod.escape(str(rel.parent.name))}</h3>"
            f"<p>{html_mod.escape(prompt[:500])}</p>{err_html}"
            f"<pre>{html_mod.escape(dsl_line[:2000])}</pre></div><div>{svg or '<em>no SVG</em>'}</div></section>"
        )

    parts.append("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
