#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


MAX_SVG_LENGTH = 16000
MAX_PATH_COUNT = 256
CANVAS_SIZE = 256
MAX_ITEMS = 512

GEOM_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text"}
TOKEN_RE = re.compile(r"[A-Za-z]|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")

# Closed color vocabulary (no arbitrary hex values in DSL).
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "GRAY": (128, 128, 128),
    "RED": (220, 20, 60),
    "GREEN": (34, 139, 34),
    "BLUE": (30, 144, 255),
    "YELLOW": (255, 215, 0),
    "CYAN": (0, 180, 180),
    "MAGENTA": (200, 0, 200),
    "ORANGE": (255, 140, 0),
    "PURPLE": (128, 0, 128),
    "BROWN": (139, 69, 19),
    "PINK": (255, 105, 180),
}
NONE_COLOR = "NONE"


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def clamp_i(v: float, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, int(round(v))))


def parse_num(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def parse_style(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in style.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def parse_viewbox(root: ET.Element) -> Tuple[float, float, float, float]:
    vb = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if vb:
        parts = re.split(r"[,\s]+", vb.strip())
        if len(parts) == 4:
            vals = [parse_num(p) for p in parts]
            if vals[2] > 0 and vals[3] > 0:
                return vals[0], vals[1], vals[2], vals[3]
    w = parse_num(root.attrib.get("width", str(CANVAS_SIZE)).replace("px", ""))
    h = parse_num(root.attrib.get("height", str(CANVAS_SIZE)).replace("px", ""))
    if w <= 0 or h <= 0:
        return 0.0, 0.0, float(CANVAS_SIZE), float(CANVAS_SIZE)
    return 0.0, 0.0, w, h


def parse_hex(hex_color: str) -> Optional[Tuple[int, int, int]]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    if len(h) != 6:
        return None
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None


def parse_rgb_color(s: str) -> Optional[Tuple[int, int, int]]:
    s = s.strip().lower()
    if s.startswith("rgb(") and s.endswith(")"):
        parts = [p.strip() for p in s[4:-1].split(",")]
        if len(parts) != 3:
            return None
        vals = []
        for p in parts:
            if p.endswith("%"):
                vals.append(clamp_i(parse_num(p[:-1]) * 2.55))
            else:
                vals.append(clamp_i(parse_num(p)))
        return vals[0], vals[1], vals[2]
    return None


CSS_NAMED = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "pink": (255, 192, 203),
}


def to_palette_token(value: str) -> str:
    raw = (value or "").strip()
    low = raw.lower()
    if low in {"", "none", "transparent"}:
        return NONE_COLOR
    if low == "currentcolor":
        return "BLACK"

    rgb = parse_hex(raw) if raw.startswith("#") else None
    if rgb is None:
        rgb = parse_rgb_color(raw)
    if rgb is None:
        rgb = CSS_NAMED.get(low)
    if rgb is None:
        return "BLACK"

    best_name = "BLACK"
    best_dist = 10**18
    for name, prgb in PALETTE.items():
        d = (rgb[0] - prgb[0]) ** 2 + (rgb[1] - prgb[1]) ** 2 + (rgb[2] - prgb[2]) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def get_style(el: ET.Element, parent: Dict[str, str]) -> Dict[str, str]:
    style = dict(parent)
    if "style" in el.attrib:
        style.update(parse_style(el.attrib["style"]))
    for k in ("fill", "stroke", "stroke-width", "opacity", "fill-opacity", "stroke-opacity", "font-size"):
        if k in el.attrib:
            style[k] = el.attrib[k]
    return style


def sx_x(x: float, sx: float, tx: float) -> int:
    return clamp_i(x * sx + tx)


def sy_y(y: float, sy: float, ty: float) -> int:
    return clamp_i(y * sy + ty)


def sw_scale(v: float, sx: float, sy: float) -> int:
    s = (abs(sx) + abs(sy)) / 2.0
    return max(0, int(round(v * s)))


def parse_path_to_abs_int(d: str, sx: float, sy: float, tx: float, ty: float) -> Optional[str]:
    tokens = TOKEN_RE.findall(d)
    if not tokens:
        return None

    i = 0
    steps = 0
    max_steps = max(1024, len(tokens) * 20)
    cmd = ""
    x = y = 0.0
    sx0 = sy0 = 0.0
    prev_c_ctrl: Optional[Tuple[float, float]] = None
    prev_q_ctrl: Optional[Tuple[float, float]] = None
    prev_cmd = ""
    out: List[str] = []

    def is_cmd(t: str) -> bool:
        return len(t) == 1 and t.isalpha()

    def need(n: int) -> bool:
        return i + n <= len(tokens)

    while i < len(tokens):
        steps += 1
        if steps > max_steps:
            return None
        loop_start = i
        if is_cmd(tokens[i]):
            cmd = tokens[i]
            i += 1
        elif not cmd:
            return None

        c = cmd
        lo = c.lower()
        rel = c.islower()

        if lo == "z":
            out.append("Z")
            x, y = sx0, sy0
            prev_c_ctrl = None
            prev_q_ctrl = None
            prev_cmd = "Z"
            continue

        if lo == "m":
            if not need(2):
                return None
            x1 = parse_num(tokens[i]); y1 = parse_num(tokens[i + 1]); i += 2
            if rel:
                x1 += x; y1 += y
            x, y = x1, y1
            sx0, sy0 = x, y
            out.extend(["M", str(sx_x(x, sx, tx)), str(sy_y(y, sy, ty))])
            prev_c_ctrl = None
            prev_q_ctrl = None
            prev_cmd = "M"
            # Implicit lineto for remaining pairs.
            while need(2) and not is_cmd(tokens[i]):
                x1 = parse_num(tokens[i]); y1 = parse_num(tokens[i + 1]); i += 2
                if rel:
                    x1 += x; y1 += y
                x, y = x1, y1
                out.extend(["L", str(sx_x(x, sx, tx)), str(sy_y(y, sy, ty))])
                prev_cmd = "L"
            continue

        if lo == "l":
            advanced = False
            while need(2) and not is_cmd(tokens[i]):
                x1 = parse_num(tokens[i]); y1 = parse_num(tokens[i + 1]); i += 2
                if rel:
                    x1 += x; y1 += y
                x, y = x1, y1
                out.extend(["L", str(sx_x(x, sx, tx)), str(sy_y(y, sy, ty))])
                advanced = True
            if not advanced:
                return None
            prev_c_ctrl = None
            prev_q_ctrl = None
            prev_cmd = "L"
            continue

        if lo == "h":
            advanced = False
            while need(1) and not is_cmd(tokens[i]):
                x1 = parse_num(tokens[i]); i += 1
                if rel:
                    x1 += x
                x = x1
                out.extend(["L", str(sx_x(x, sx, tx)), str(sy_y(y, sy, ty))])
                advanced = True
            if not advanced:
                return None
            prev_c_ctrl = None
            prev_q_ctrl = None
            prev_cmd = "L"
            continue

        if lo == "v":
            advanced = False
            while need(1) and not is_cmd(tokens[i]):
                y1 = parse_num(tokens[i]); i += 1
                if rel:
                    y1 += y
                y = y1
                out.extend(["L", str(sx_x(x, sx, tx)), str(sy_y(y, sy, ty))])
                advanced = True
            if not advanced:
                return None
            prev_c_ctrl = None
            prev_q_ctrl = None
            prev_cmd = "L"
            continue

        if lo == "c":
            advanced = False
            while need(6) and not is_cmd(tokens[i]):
                x1 = parse_num(tokens[i]); y1 = parse_num(tokens[i + 1])
                x2 = parse_num(tokens[i + 2]); y2 = parse_num(tokens[i + 3])
                x3 = parse_num(tokens[i + 4]); y3 = parse_num(tokens[i + 5]); i += 6
                if rel:
                    x1 += x; y1 += y; x2 += x; y2 += y; x3 += x; y3 += y
                out.extend([
                    "C",
                    str(sx_x(x1, sx, tx)), str(sy_y(y1, sy, ty)),
                    str(sx_x(x2, sx, tx)), str(sy_y(y2, sy, ty)),
                    str(sx_x(x3, sx, tx)), str(sy_y(y3, sy, ty)),
                ])
                x, y = x3, y3
                prev_c_ctrl = (x2, y2)
                prev_q_ctrl = None
                prev_cmd = "C"
                advanced = True
            if not advanced:
                return None
            continue

        if lo == "s":
            advanced = False
            while need(4) and not is_cmd(tokens[i]):
                x2 = parse_num(tokens[i]); y2 = parse_num(tokens[i + 1])
                x3 = parse_num(tokens[i + 2]); y3 = parse_num(tokens[i + 3]); i += 4
                if prev_cmd in {"C", "S"} and prev_c_ctrl is not None:
                    x1 = 2 * x - prev_c_ctrl[0]
                    y1 = 2 * y - prev_c_ctrl[1]
                else:
                    x1, y1 = x, y
                if rel:
                    x2 += x; y2 += y; x3 += x; y3 += y
                out.extend([
                    "C",
                    str(sx_x(x1, sx, tx)), str(sy_y(y1, sy, ty)),
                    str(sx_x(x2, sx, tx)), str(sy_y(y2, sy, ty)),
                    str(sx_x(x3, sx, tx)), str(sy_y(y3, sy, ty)),
                ])
                x, y = x3, y3
                prev_c_ctrl = (x2, y2)
                prev_q_ctrl = None
                prev_cmd = "S"
                advanced = True
            if not advanced:
                return None
            continue

        if lo == "q":
            advanced = False
            while need(4) and not is_cmd(tokens[i]):
                x1 = parse_num(tokens[i]); y1 = parse_num(tokens[i + 1])
                x2 = parse_num(tokens[i + 2]); y2 = parse_num(tokens[i + 3]); i += 4
                if rel:
                    x1 += x; y1 += y; x2 += x; y2 += y
                out.extend([
                    "Q",
                    str(sx_x(x1, sx, tx)), str(sy_y(y1, sy, ty)),
                    str(sx_x(x2, sx, tx)), str(sy_y(y2, sy, ty)),
                ])
                x, y = x2, y2
                prev_q_ctrl = (x1, y1)
                prev_c_ctrl = None
                prev_cmd = "Q"
                advanced = True
            if not advanced:
                return None
            continue

        if lo == "t":
            advanced = False
            while need(2) and not is_cmd(tokens[i]):
                x2 = parse_num(tokens[i]); y2 = parse_num(tokens[i + 1]); i += 2
                if prev_cmd in {"Q", "T"} and prev_q_ctrl is not None:
                    x1 = 2 * x - prev_q_ctrl[0]
                    y1 = 2 * y - prev_q_ctrl[1]
                else:
                    x1, y1 = x, y
                if rel:
                    x2 += x; y2 += y
                out.extend([
                    "Q",
                    str(sx_x(x1, sx, tx)), str(sy_y(y1, sy, ty)),
                    str(sx_x(x2, sx, tx)), str(sy_y(y2, sy, ty)),
                ])
                x, y = x2, y2
                prev_q_ctrl = (x1, y1)
                prev_c_ctrl = None
                prev_cmd = "T"
                advanced = True
            if not advanced:
                return None
            continue

        if lo == "a":
            advanced = False
            while need(7) and not is_cmd(tokens[i]):
                rx = parse_num(tokens[i]); ry = parse_num(tokens[i + 1]); xrot = parse_num(tokens[i + 2])
                laf = int(parse_num(tokens[i + 3])); sf = int(parse_num(tokens[i + 4]))
                x2 = parse_num(tokens[i + 5]); y2 = parse_num(tokens[i + 6]); i += 7
                if rel:
                    x2 += x; y2 += y
                out.extend([
                    "A",
                    str(max(0, int(round(abs(rx * sx))))),
                    str(max(0, int(round(abs(ry * sy))))),
                    str(int(round(xrot))),
                    str(1 if laf else 0),
                    str(1 if sf else 0),
                    str(sx_x(x2, sx, tx)),
                    str(sy_y(y2, sy, ty)),
                ])
                x, y = x2, y2
                prev_c_ctrl = None
                prev_q_ctrl = None
                prev_cmd = "A"
                advanced = True
            if not advanced:
                return None
            continue

        return None

        if i == loop_start:
            return None

    return " ".join(out)


def collect_items(
    el: ET.Element,
    inherited_style: Dict[str, str],
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    out: List[Dict[str, object]],
) -> None:
    tag = strip_ns(el.tag)
    style = get_style(el, inherited_style)

    if tag in GEOM_TAGS:
        fill = to_palette_token(style.get("fill", "black"))
        stroke = to_palette_token(style.get("stroke", "none"))
        sw = sw_scale(parse_num(style.get("stroke-width", "0")), sx, sy)
        op = clamp_i(parse_num(style.get("opacity", "1")) * 255.0)
        font = max(0, int(round(parse_num(style.get("font-size", "0")) * ((abs(sx) + abs(sy)) / 2.0))))
        text = ""
        geom = ""

        if tag == "path":
            d = el.attrib.get("d", "")
            geom = parse_path_to_abs_int(d, sx, sy, tx, ty) or ""
        elif tag == "rect":
            x = sx_x(parse_num(el.attrib.get("x", "0")), sx, tx)
            y = sy_y(parse_num(el.attrib.get("y", "0")), sy, ty)
            w = max(0, sw_scale(parse_num(el.attrib.get("width", "0")), sx, sx))
            h = max(0, sw_scale(parse_num(el.attrib.get("height", "0")), sy, sy))
            rx = max(0, sw_scale(parse_num(el.attrib.get("rx", "0")), sx, sx))
            ry = max(0, sw_scale(parse_num(el.attrib.get("ry", "0")), sy, sy))
            geom = f"RECT {x} {y} {w} {h} {rx} {ry}"
        elif tag == "circle":
            cx = sx_x(parse_num(el.attrib.get("cx", "0")), sx, tx)
            cy = sy_y(parse_num(el.attrib.get("cy", "0")), sy, ty)
            r = max(0, int(round(parse_num(el.attrib.get("r", "0")) * ((abs(sx) + abs(sy)) / 2.0))))
            geom = f"CIRCLE {cx} {cy} {r}"
        elif tag == "ellipse":
            cx = sx_x(parse_num(el.attrib.get("cx", "0")), sx, tx)
            cy = sy_y(parse_num(el.attrib.get("cy", "0")), sy, ty)
            rx = max(0, int(round(parse_num(el.attrib.get("rx", "0")) * abs(sx))))
            ry = max(0, int(round(parse_num(el.attrib.get("ry", "0")) * abs(sy))))
            geom = f"ELLIPSE {cx} {cy} {rx} {ry}"
        elif tag == "line":
            x1 = sx_x(parse_num(el.attrib.get("x1", "0")), sx, tx)
            y1 = sy_y(parse_num(el.attrib.get("y1", "0")), sy, ty)
            x2 = sx_x(parse_num(el.attrib.get("x2", "0")), sx, tx)
            y2 = sy_y(parse_num(el.attrib.get("y2", "0")), sy, ty)
            geom = f"LINE {x1} {y1} {x2} {y2}"
        elif tag in {"polyline", "polygon"}:
            pts = re.split(r"[,\s]+", (el.attrib.get("points", "") or "").strip())
            nums = [parse_num(p) for p in pts if p != ""]
            ints: List[int] = []
            for idx in range(0, len(nums) - 1, 2):
                ints.append(sx_x(nums[idx], sx, tx))
                ints.append(sy_y(nums[idx + 1], sy, ty))
            kind = "POLYGON" if tag == "polygon" else "POLYLINE"
            geom = f"{kind} " + " ".join(str(v) for v in ints)
        elif tag == "text":
            x = sx_x(parse_num(el.attrib.get("x", "0")), sx, tx)
            y = sy_y(parse_num(el.attrib.get("y", "0")), sy, ty)
            txt = re.sub(r"\s+", " ", "".join(el.itertext()).strip())
            text = txt[:120]
            geom = f"TEXT {x} {y}"

        if geom:
            item = {
                "t": tag.upper(),
                "geom": geom,
                "fill": fill,
                "stroke": stroke,
                "sw": sw,
                "op": op,
                "font": font,
                "text": text,
            }
            out.append(item)

    for child in list(el):
        collect_items(child, style, sx, sy, tx, ty, out)


def count_paths(el: ET.Element) -> int:
    c = 1 if strip_ns(el.tag) == "path" else 0
    for child in list(el):
        c += count_paths(child)
    return c


def svg_to_dsl(svg_raw: str) -> Optional[str]:
    root = ET.fromstring(svg_raw)
    if strip_ns(root.tag) != "svg":
        return None

    vb_x, vb_y, vb_w, vb_h = parse_viewbox(root)
    sx = CANVAS_SIZE / vb_w
    sy = CANVAS_SIZE / vb_h
    tx = -vb_x * sx
    ty = -vb_y * sy

    items: List[Dict[str, object]] = []
    collect_items(root, {}, sx, sy, tx, ty, items)
    if not items or len(items) > MAX_ITEMS:
        return None

    # Fixed top-level grammar and order.
    dsl_obj = {
        "v": 2,
        "canvas": CANVAS_SIZE,
        "n": len(items),
        "items": items,
    }
    return json.dumps(dsl_obj, separators=(",", ":"), ensure_ascii=True)


def convert(input_csv: Path, output_jsonl: Path, report_json: Path) -> None:
    total = 0
    written = 0
    dropped = 0
    reasons: Dict[str, int] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "prompt" not in reader.fieldnames or "svg" not in reader.fieldnames:
            raise ValueError("Input CSV must include columns: id,prompt,svg")

        for row in reader:
            total += 1
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            svg = (row.get("svg") or "").strip()

            reason = ""
            if not prompt:
                reason = "empty_prompt"
            elif not svg:
                reason = "empty_svg"
            elif len(svg) > MAX_SVG_LENGTH:
                reason = "svg_too_long"
            else:
                try:
                    root = ET.fromstring(svg)
                    if count_paths(root) > MAX_PATH_COUNT:
                        reason = "too_many_paths"
                except Exception:
                    reason = "invalid_svg"

            if reason:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue

            try:
                dsl = svg_to_dsl(svg)
            except Exception:
                dsl = None

            if not dsl:
                dropped += 1
                reasons["dsl_v2_failed"] = reasons.get("dsl_v2_failed", 0) + 1
                continue

            rec = {"id": rid, "prompt": prompt, "dsl": dsl}
            fout.write(json.dumps(rec, ensure_ascii=True) + "\n")
            written += 1

    report = {
        "input_csv": str(input_csv),
        "output_jsonl": str(output_jsonl),
        "total_rows": total,
        "written_rows": written,
        "dropped_rows": dropped,
        "drop_reasons": reasons,
        "dsl_version": 2,
        "constraints": {
            "canvas_size": CANVAS_SIZE,
            "max_svg_length": MAX_SVG_LENGTH,
            "max_path_count": MAX_PATH_COUNT,
            "one_object_per_line": True,
            "self_contained_rows": True,
            "fixed_field_order": True,
            "integer_coordinates_only": True,
            "absolute_coordinates_only": True,
            "closed_color_vocabulary": [NONE_COLOR] + sorted(PALETTE.keys()),
            "no_hex_colors_in_dsl": True,
            "small_command_vocabulary": ["M", "L", "C", "Q", "A", "Z", "RECT", "CIRCLE", "ELLIPSE", "LINE", "POLYLINE", "POLYGON", "TEXT"],
        },
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict SVG DSL v2 for fine-tuning.")
    parser.add_argument("--input", required=True, help="CSV with id,prompt,svg")
    parser.add_argument("--output", required=True, help="Output JSONL for prompt->DSL")
    parser.add_argument("--report", required=True, help="Output report JSON")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output), Path(args.report))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
