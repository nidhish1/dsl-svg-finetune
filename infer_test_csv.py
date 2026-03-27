#!/usr/bin/env python3
"""
Run inference on test.csv and build Kaggle-style submission files.

Outputs:
- out_dir/<id>/prompt.txt, raw.txt, dsl.json, output.svg (when available)
- out_dir/index.html
- out_dir/submission_dsl.csv  (id,dsl)
- out_dir/submission.csv      (id,svg)

Example:
  python infer_test_csv.py \
    --model outputs/qwen-dsl-sft-full/final \
    --test-csv dl-spring-2026-svg-generation/test.csv \
    --out-dir infer_test_out
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dsl_infer_utils import extract_dsl_json  # noqa: E402
from dsl_v2_to_svg import dsl_to_svg  # noqa: E402

DEFAULT_SYSTEM = (
    "You convert scene descriptions into compact scene DSL. "
    "Follow the user's output format exactly."
)
DEFAULT_USER_INSTRUCTION = (
    "Output only the scene DSL as a single JSON object on one line (schema v2: keys "
    "v, canvas, n, items). No markdown, no explanation. After the JSON line, output "
    "a second line containing only the word END."
)


def build_messages(prompt: str, system: str, instruction: str) -> list:
    user = f"{instruction}\n\nScene description:\n{prompt.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_model(model_path: Path, device: str, bf16: bool):
    dtype = torch.bfloat16 if (bf16 and device == "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    load_kw: dict = {"trust_remote_code": True, "device_map": "auto" if device == "cuda" else None}
    dt = dtype if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dt, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dt, **load_kw)
    if device == "cpu":
        model = model.to("cpu")
    return model, tokenizer


def generate_raw(model, tokenizer, messages, device: str, max_new_tokens: int) -> str:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = encoded["input_ids"] if not isinstance(encoded, torch.Tensor) else encoded
    if device == "cuda":
        prompt_ids = prompt_ids.to(model.device)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
    prompt_len = prompt_ids.shape[1]
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(prompt_ids, attention_mask=attention_mask, generation_config=gen_cfg)
    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def blank_svg(canvas: int = 256) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{canvas}' height='{canvas}' "
        f"viewBox='0 0 {canvas} {canvas}'><rect width='{canvas}' height='{canvas}' fill='white'/></svg>"
    )


def write_index(out_dir: Path, entries: list[dict]) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Test Inference</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}",
        ".row{display:grid;grid-template-columns:1fr 280px;gap:16px;align-items:start;border-bottom:1px solid #ddd;padding:16px 0;}",
        "pre{background:#f5f5f5;font-size:11px;overflow:auto;max-height:200px;}",
        "svg{max-width:256px;border:1px solid #ccc;}</style></head><body>",
        "<h1>Test inference preview</h1>",
    ]
    for e in entries:
        rid = html_mod.escape(e["id"])
        prompt = html_mod.escape(e["prompt"])
        err = html_mod.escape(e.get("err") or "")
        dsl_short = html_mod.escape((e.get("dsl") or "")[:1200])
        svg = e.get("svg") or ""
        parts.append(f"<section class='row'><div><h3>{rid}</h3><p>{prompt}</p>")
        if err:
            parts.append(f"<p style='color:#a00'>{err}</p>")
        if dsl_short:
            parts.append(f"<pre>{dsl_short}</pre>")
        parts.append(f"</div><div>{svg or '<em>No SVG</em>'}</div></section>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("".join(parts), encoding="utf-8")


def load_test_rows(test_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if rid:
                rows.append({"id": rid, "prompt": prompt})
    return rows


def select_shard_rows(rows: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if num_shards == 1:
        return rows
    return [row for i, row in enumerate(rows) if (i % num_shards) == shard_index]


def main() -> None:
    p = argparse.ArgumentParser(description="Inference on test.csv -> submission files")
    p.add_argument("--model", type=Path, default=Path("outputs/qwen-dsl-sft/final"))
    p.add_argument("--test-csv", type=Path, default=Path("dl-spring-2026-svg-generation/test.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("infer_test_out"))
    p.add_argument("--limit", type=int, default=None, help="Only run first N test rows (smoke test)")
    p.add_argument("--num-shards", type=int, default=1, help="Total shard count for parallel multi-GPU runs")
    p.add_argument("--shard-index", type=int, default=0, help="This worker shard index [0..num_shards-1]")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    p.add_argument("--user-instruction", default=DEFAULT_USER_INSTRUCTION)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model.expanduser().resolve()
    test_csv = args.test_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not model_path.is_dir():
        sys.exit(f"Model not found: {model_path}")
    if not test_csv.is_file():
        sys.exit(f"Test CSV not found: {test_csv}")

    rows = load_test_rows(test_csv)
    if not rows:
        sys.exit(f"No test rows found in: {test_csv}")
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
        if not rows:
            sys.exit("--limit resulted in zero rows to process")
    try:
        rows = select_shard_rows(rows, args.num_shards, args.shard_index)
    except ValueError as ex:
        sys.exit(str(ex))
    if not rows:
        sys.exit("No rows assigned to this shard")

    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(model_path, device, args.bf16)

    submission_svg_rows: list[dict] = []
    submission_dsl_rows: list[dict] = []
    entries: list[dict] = []

    print(
        f"Running shard {args.shard_index + 1}/{args.num_shards} on {len(rows)} rows "
        f"(device={device}, max_new_tokens={args.max_new_tokens})",
        flush=True,
    )

    for i, row in enumerate(rows, start=1):
        rid = row["id"]
        prompt = row["prompt"]
        sample_dir = out_dir / rid
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        messages = build_messages(prompt, args.system_prompt, args.user_instruction)
        raw = generate_raw(model, tokenizer, messages, device, args.max_new_tokens)
        (sample_dir / "raw.txt").write_text(raw, encoding="utf-8")

        _obj, dsl_line = extract_dsl_json(raw)
        err = ""
        svg = ""
        if dsl_line:
            (sample_dir / "dsl.json").write_text(dsl_line + "\n", encoding="utf-8")
            try:
                svg = dsl_to_svg(dsl_line)
            except Exception as ex:
                err = f"SVG conversion failed: {ex}"
        else:
            err = "No valid DSL JSON in model output"

        if not svg:
            svg = blank_svg(256)
        (sample_dir / "output.svg").write_text(svg, encoding="utf-8")

        submission_dsl_rows.append({"id": rid, "dsl": dsl_line or ""})
        submission_svg_rows.append({"id": rid, "svg": svg})
        entries.append({"id": rid, "prompt": prompt, "dsl": dsl_line or "", "svg": svg, "err": err})
        print(
            f"[shard {args.shard_index + 1}/{args.num_shards}] "
            f"[{i}/{len(rows)}] id={rid} dsl={bool(dsl_line)} svg={bool(svg)}",
            flush=True,
        )

    with (out_dir / "submission_dsl.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dsl"])
        writer.writeheader()
        writer.writerows(submission_dsl_rows)

    with (out_dir / "submission.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "svg"])
        writer.writeheader()
        writer.writerows(submission_svg_rows)

    write_index(out_dir, entries)
    print(f"Wrote {out_dir}/submission.csv")
    print(f"Wrote {out_dir}/submission_dsl.csv")
    print(f"Wrote {out_dir}/index.html")


if __name__ == "__main__":
    main()

