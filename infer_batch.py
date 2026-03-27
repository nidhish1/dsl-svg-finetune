#!/usr/bin/env python3
"""
Batch inference on GPU server: many prompts -> per-sample folder + index.html.

  python infer_batch.py --model outputs/qwen-dsl-sft/final --bf16 --prompts-file prompts.txt

Copy to laptop:
  scp -r user@host:~/dsl-svg-finetune/infer_batch_out ~/Downloads/

Preview locally (no GPU):
  python scripts/preview_dsl_folder.py --dir ~/Downloads/infer_batch_out
"""

from __future__ import annotations

import argparse
import html as html_mod
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
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    load_kw: dict = {"trust_remote_code": True, "device_map": "auto" if device == "cuda" else None}
    dt = dtype if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dt, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dt, **load_kw)
    if device == "cpu":
        model = model.to("cpu")
    return model, tok


def generate_raw(model, tokenizer, messages, device: str, max_new_tokens: int) -> str:
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    if device == "cuda":
        prompt_ids = prompt_ids.to(model.device)
    attn = torch.ones_like(prompt_ids, dtype=torch.long)
    plen = prompt_ids.shape[1]
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(prompt_ids, attention_mask=attn, generation_config=gen_cfg)
    new_tok = out[0, plen:]
    return tokenizer.decode(new_tok, skip_special_tokens=True)


def write_index(out_dir: Path, entries: list[dict]) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Batch DSL</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}",
        ".row{display:grid;grid-template-columns:1fr 280px;gap:16px;align-items:start;border-bottom:1px solid #ddd;padding:16px 0;}",
        "pre{background:#f5f5f5;font-size:11px;overflow:auto;max-height:200px;}",
        "svg{max-width:256px;border:1px solid #ccc;}</style></head><body>",
        "<h1>Batch inference</h1>",
    ]
    for e in entries:
        pid = html_mod.escape(e["id"])
        pr = html_mod.escape(e["prompt"])
        err = html_mod.escape(e.get("err") or "")
        svg_block = e.get("svg") or ""
        dsl_short = html_mod.escape((e.get("dsl") or "")[:1200])
        parts.append(f"<section class='row'><div><h3>{pid}</h3><p>{pr}</p>")
        if e.get("err"):
            parts.append(f"<p style='color:#a00'>{err}</p>")
        if e.get("dsl"):
            parts.append(f"<pre>{dsl_short}</pre></div><div>{svg_block}</div></section>")
        else:
            parts.append(f"</div><div><em>No SVG</em></div></section>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch prompt -> DSL -> SVG")
    p.add_argument("--model", type=Path, default=Path("outputs/qwen-dsl-sft/final"))
    p.add_argument("--prompts-file", type=Path, required=True, help="One prompt per line")
    p.add_argument("--out-dir", type=Path, default=Path("infer_batch_out"))
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    p.add_argument("--user-instruction", default=DEFAULT_USER_INSTRUCTION)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        sys.exit(f"Model not found: {model_path}")

    prompts_path = args.prompts_file.expanduser().resolve()
    if not prompts_path.is_file():
        sys.exit(
            f"Prompts file not found: {prompts_path}\n"
            "Create it in the repo root (see prompts.txt) or pass a full path, e.g.\n"
            "  --prompts-file /home/cc/dsl-svg-finetune/prompts.txt"
        )

    lines = [
        ln.strip()
        for ln in prompts_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        sys.exit("No prompts in file")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(model_path, device, args.bf16)

    entries: list[dict] = []
    for idx, prompt in enumerate(lines):
        sid = f"{idx:03d}"
        sub = args.out_dir / sid
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        messages = build_messages(prompt, args.system_prompt, args.user_instruction)
        raw = generate_raw(model, tokenizer, messages, device, args.max_new_tokens)
        (sub / "raw.txt").write_text(raw, encoding="utf-8")

        _obj, dsl_line = extract_dsl_json(raw)
        err = ""
        svg = ""
        if dsl_line:
            (sub / "dsl.json").write_text(dsl_line + "\n", encoding="utf-8")
            try:
                svg = dsl_to_svg(dsl_line)
                (sub / "output.svg").write_text(svg, encoding="utf-8")
            except Exception as ex:
                err = f"SVG: {ex}"
        else:
            err = "No valid DSL JSON (truncated or malformed). Try --max-new-tokens 8192."

        entries.append(
            {
                "id": sid,
                "prompt": prompt,
                "dsl": dsl_line or "",
                "svg": svg,
                "err": err,
            }
        )
        print(f"[{sid}] {prompt[:60]}... ok={bool(dsl_line)} svg={bool(svg)}", flush=True)

    write_index(args.out_dir, entries)
    print(f"Wrote {args.out_dir}/index.html and {len(entries)} subfolders", flush=True)


if __name__ == "__main__":
    main()
