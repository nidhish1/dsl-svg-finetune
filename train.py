#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-Coder (Instruct) for scene DSL generation.

Uses TRL SFTTrainer with conversational prompt/completion rows (lists of
{role, content} dicts). The tokenizer's built-in chat template is applied
via TRL/HF — do not supply a second custom template.

Dataset: JSONL with columns id, prompt (natural language), dsl (single-line JSON string).

References:
- Qwen2.5-Coder Instruct: chat template + tokenizer eos_token / transformers>=4.37
- TRL SFT: prompt-completion with completion-only loss
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset

try:
    from trl import SFTConfig, SFTTrainer
except ImportError as e:  # pragma: no cover
    print(
        "TRL is required. In a venv: pip install 'transformers>=4.37.0' trl accelerate datasets torch",
        file=sys.stderr,
    )
    raise e


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_DATA = Path("dl-spring-2026-svg-generation/train_dsl_v2.jsonl")
DEFAULT_OUTPUT = Path("outputs/qwen-dsl-sft")

DSL_TERMINATOR = "END"

DEFAULT_SYSTEM = (
    "You convert scene descriptions into compact scene DSL. "
    "Follow the user's output format exactly."
)

DEFAULT_USER_INSTRUCTION = (
    "Output only the scene DSL as a single JSON object on one line (schema v2: keys "
    "v, canvas, n, items). No markdown, no explanation. After the JSON line, output "
    "a second line containing only the word END."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT Qwen2.5-Coder for DSL (TRL, chat template)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL: id, prompt, dsl")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    p.add_argument("--user-instruction", default=DEFAULT_USER_INSTRUCTION)
    p.add_argument("--max-length", type=int, default=4096, help="Max tokens per example (truncate right)")
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on Ampere+ CUDA)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 (if not using bf16)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows for debugging")
    p.add_argument(
        "--strict-json",
        action="store_true",
        help="Drop rows where dsl is not valid JSON (cleaner targets)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p


def assistant_body(dsl_line: str) -> str:
    return f"{dsl_line.strip()}\n{DSL_TERMINATOR}"


def _model_init_kwargs(args) -> dict:
    kwargs: dict = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["torch_dtype"] = torch.float16
    return kwargs


def row_to_messages(example: dict, system: str, user_instruction: str) -> dict:
    user_content = (
        f"{user_instruction}\n\nScene description:\n{str(example['prompt']).strip()}"
    )
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "completion": [{"role": "assistant", "content": assistant_body(example["dsl"])}],
    }


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data not found: {args.data}")

    torch.manual_seed(args.seed)

    # Qwen eos for alignment with chat template (see model card)
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        max_length=args.max_length,
        packing=False,
        shuffle_dataset=True,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        # Use tokenizer default eos (Qwen uses </think> from config). Passing the string
        # literal breaks TRL's vocab check because convert_tokens_to_ids splits multi-byte tokens.
        model_init_kwargs=_model_init_kwargs(args),
    )

    ds = load_dataset("json", data_files=str(args.data), split="train")

    if args.strict_json:
        def valid_json(ex: dict) -> bool:
            try:
                json.loads(ex["dsl"])
                return True
            except json.JSONDecodeError:
                return False

        ds = ds.filter(valid_json)

    if args.max_samples is not None:
        n = min(args.max_samples, len(ds))
        ds = ds.select(range(n))

    ds = ds.map(
        lambda ex: row_to_messages(ex, args.system_prompt, args.user_instruction),
        remove_columns=[c for c in ds.column_names],
    )

    trainer = SFTTrainer(
        model=args.model,
        args=sft_args,
        train_dataset=ds,
    )

    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer.processing_class.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
