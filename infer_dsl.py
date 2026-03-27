#!/usr/bin/env python3
"""
Generate scene DSL with a fine-tuned (or base) Qwen chat model, then render DSL v2 to SVG.

Uses the same system + user instruction framing as train.py.

Example (server):
  python infer_dsl.py --model outputs/qwen-dsl-sft/final --prompt "A red circle on white."
  scp user@host:~/dsl-svg-finetune/infer_results/preview.html .
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dsl_v2_to_svg import dsl_to_svg  # noqa: E402

# Match train.py defaults
DEFAULT_SYSTEM = (
    "You convert scene descriptions into compact scene DSL. "
    "Follow the user's output format exactly."
)
DEFAULT_USER_INSTRUCTION = (
    "Output only the scene DSL as a single JSON object on one line (schema v2: keys "
    "v, canvas, n, items). No markdown, no explanation. After the JSON line, output "
    "a second line containing only the word END."
)


def extract_dsl_line(text: str) -> str | None:
    """Pull the first line that is valid DSL v2 JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue
        if line.startswith("{") and '"v"' in line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("v") == 2 and "items" in obj:
                    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
            except json.JSONDecodeError:
                continue
    return None


def build_messages(prompt: str, system: str, instruction: str) -> list:
    user = f"{instruction}\n\nScene description:\n{prompt.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Inference: prompt -> DSL -> SVG preview")
    p.add_argument("--model", type=Path, default=Path("outputs/qwen-dsl-sft/final"))
    p.add_argument("--prompt", default="A simple black smiley face on white background.")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    p.add_argument("--user-instruction", default=DEFAULT_USER_INSTRUCTION)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu | auto (auto = cuda if available)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("infer_results"))
    p.add_argument("--bf16", action="store_true", help="Load weights in bf16 (CUDA)")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    dtype = torch.bfloat16 if (args.bf16 and args.device == "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    messages = build_messages(args.prompt, args.system_prompt, args.user_instruction)
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if args.device == "cuda":
        prompt_ids = prompt_ids.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0, prompt_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    dsl_line = extract_dsl_line(raw)
    svg = ""
    parse_err = ""
    if dsl_line:
        try:
            svg = dsl_to_svg(dsl_line)
        except Exception as e:
            parse_err = str(e)
    else:
        parse_err = "Could not find a valid DSL v2 JSON line in model output."

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt.txt").write_text(args.prompt, encoding="utf-8")
    (args.out_dir / "raw_generation.txt").write_text(raw, encoding="utf-8")
    if dsl_line:
        (args.out_dir / "dsl.json").write_text(dsl_line + "\n", encoding="utf-8")
    if svg:
        (args.out_dir / "output.svg").write_text(svg, encoding="utf-8")

    preview = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>DSL inference preview</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; }}
pre {{ background: #f5f5f5; padding: 12px; overflow: auto; font-size: 12px; }}
.panel {{ border: 1px solid #ccc; padding: 16px; margin: 16px 0; }}
svg {{ max-width: 256px; height: auto; border: 1px solid #ddd; }}
.err {{ color: #a00; }}
</style></head><body>
<h1>Inference preview</h1>
<div class="panel"><h2>Prompt</h2><p>{html.escape(args.prompt)}</p></div>
<div class="panel"><h2>Raw generation</h2><pre>{html.escape(raw)}</pre></div>
"""
    if parse_err:
        preview += f'<div class="panel err"><h2>SVG</h2><p>{html.escape(parse_err)}</p></div>\n'
    else:
        preview += f'<div class="panel"><h2>Reconstructed SVG</h2>{svg}</div>\n'
    if dsl_line:
        preview += f'<div class="panel"><h2>DSL (one line)</h2><pre>{html.escape(dsl_line)}</pre></div>\n'
    preview += "</body></html>"
    (args.out_dir / "preview.html").write_text(preview, encoding="utf-8")

    print(f"Wrote {args.out_dir}/preview.html, raw_generation.txt, prompt.txt")
    if dsl_line:
        print(f"Wrote {args.out_dir}/dsl.json, output.svg")
    if parse_err:
        print(f"Warning: {parse_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
