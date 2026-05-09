import argparse
import json
from pathlib import Path

import torch

from .sample_action_tokens import generate_tokens, generate_two_stage, load_model
from .visualize import save_strokes_png


DEFAULT_PROMPTS = [
    "在画布中央画一个中等正方形",
    "在画布右边画一个大圆",
    "请画一个中等三角形，放在画布中央",
    "在画布左边画一个中等线段",
    "请画一个大宽矩形，放在画布下方右边",
    "在画布上方画一个中等高矩形",
    "画一个中等椭圆，位置在画布中央",
    "在画布下方左边画一个大直角三角形",
]


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", "，"}:
            keep.append("_")
    name = "".join(keep).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch sample prompts from action-token model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_action_tokens_chinese_mvp/checkpoint.pt")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_action_tokens_chinese_mvp/samples")
    parser.add_argument("--max-steps", type=int, default=170)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    parser.add_argument("--two-stage", action="store_true", help="双阶段推理模式")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Override DEFAULT_PROMPTS. For decoder-only-pretrain checkpoints "
             "the prompt content is ignored anyway — pass placeholder labels "
             "(e.g. shape names) so the output filenames are meaningful.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many samples to generate per prompt (default 1).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)

    base_prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    expanded_prompts = []
    for p in base_prompts:
        for k in range(args.repeats):
            expanded_prompts.append((p, k))

    summary = []
    for idx, (prompt, repeat_idx) in enumerate(expanded_prompts, start=1):
        if args.two_stage:
            strokes = generate_two_stage(model, tokenizer, prompt, max_steps=args.max_steps, device=device)
            tokens = None
        else:
            tokens = generate_tokens(model, tokenizer, prompt, max_steps=args.max_steps, device=device)
            strokes = tokenizer.decode_tokens(tokens)
        suffix = f"_r{repeat_idx}" if args.repeats > 1 else ""
        stem = f"{idx:02d}_{safe_name(prompt)}{suffix}"
        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"
        save_strokes_png(strokes, png_path, title=prompt)
        json_path.write_text(
            json.dumps({"prompt": prompt, "tokens": tokens, "strokes": strokes}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary.append(
            {
                "prompt": prompt,
                "num_steps": len(strokes),
                "last_pen_state": strokes[-1]["pen_state"] if strokes else None,
                "png": str(png_path),
                "json": str(json_path),
            }
        )
        print(f"{idx}. {prompt} -> steps={len(strokes)} last={summary[-1]['last_pen_state']} png={png_path}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
