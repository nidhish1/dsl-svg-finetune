#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


PALETTE_HEX: Dict[str, str] = {
    "BLACK": "#000000",
    "WHITE": "#FFFFFF",
    "GRAY": "#808080",
    "RED": "#DC143C",
    "GREEN": "#228B22",
    "BLUE": "#1E90FF",
    "YELLOW": "#FFD700",
    "CYAN": "#00B4B4",
    "MAGENTA": "#C800C8",
    "ORANGE": "#FF8C00",
    "PURPLE": "#800080",
    "BROWN": "#8B4513",
    "PINK": "#FF69B4",
    "NONE": "none",
}


def esc_text(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def color_to_svg(token: str) -> str:
    return PALETTE_HEX.get(token, "#000000")


def parse_ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def item_to_svg(item: Dict[str, object]) -> str:
    t = str(item.get("t", ""))
    geom = str(item.get("geom", ""))
    fill = color_to_svg(str(item.get("fill", "BLACK")))
    stroke = color_to_svg(str(item.get("stroke", "NONE")))
    sw = int(item.get("sw", 0))
    op = max(0.0, min(1.0, int(item.get("op", 255)) / 255.0))
    font = int(item.get("font", 0))
    text = esc_text(str(item.get("text", "")))

    common = f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op:.3f}"'

    if t == "PATH":
        return f'<path d="{geom}"{common}/>'
    if t == "RECT":
        vals = parse_ints(geom)
        if len(vals) >= 6:
            x, y, w, h, rx, ry = vals[:6]
            return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{ry}"{common}/>'
    if t == "CIRCLE":
        vals = parse_ints(geom)
        if len(vals) >= 3:
            cx, cy, r = vals[:3]
            return f'<circle cx="{cx}" cy="{cy}" r="{r}"{common}/>'
    if t == "ELLIPSE":
        vals = parse_ints(geom)
        if len(vals) >= 4:
            cx, cy, rx, ry = vals[:4]
            return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"{common}/>'
    if t == "LINE":
        vals = parse_ints(geom)
        if len(vals) >= 4:
            x1, y1, x2, y2 = vals[:4]
            return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{common}/>'
    if t in {"POLYLINE", "POLYGON"}:
        vals = parse_ints(geom)
        if len(vals) >= 4:
            pts = " ".join(f"{vals[i]},{vals[i+1]}" for i in range(0, len(vals) - 1, 2))
            tag = "polygon" if t == "POLYGON" else "polyline"
            return f'<{tag} points="{pts}"{common}/>'
    if t == "TEXT":
        vals = parse_ints(geom)
        if len(vals) >= 2:
            x, y = vals[:2]
            font_attr = f' font-size="{font}"' if font > 0 else ""
            return f'<text x="{x}" y="{y}"{common}{font_attr}>{text}</text>'

    return ""


def dsl_to_svg(dsl_json: str) -> str:
    obj = json.loads(dsl_json)
    canvas = int(obj.get("canvas", 256))
    items = obj.get("items", [])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas} {canvas}" width="{canvas}" height="{canvas}">'
    ]
    for item in items:
        if isinstance(item, dict):
            p = item_to_svg(item)
            if p:
                parts.append(p)
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DSL v2 JSON string/file to SVG.")
    parser.add_argument("--dsl-file", help="Path to file containing a DSL JSON object")
    parser.add_argument("--dsl-json", help="Inline DSL JSON string")
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    if not args.dsl_file and not args.dsl_json:
        raise ValueError("Provide either --dsl-file or --dsl-json")

    dsl = args.dsl_json or Path(args.dsl_file).read_text(encoding="utf-8")
    svg = dsl_to_svg(dsl)
    Path(args.output).write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
