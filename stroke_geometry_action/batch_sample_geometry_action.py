from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stroke_baseline.visualize import save_strokes_png

from .sample_geometry_action import generate_tokens, load_model


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
    parser = argparse.ArgumentParser(description="Batch sample Chinese prompts from geometry-action model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_geometry_action_chinese_mvp/checkpoint.pt")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_geometry_action_chinese_mvp/samples")
    parser.add_argument("--max-steps", type=int, default=170)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    summary = []
    for idx, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        tokens = generate_tokens(model, tokenizer, prompt, max_steps=args.max_steps, device=device)
        strokes = tokenizer.decode_tokens(tokens)
        stem = f"{idx:02d}_{safe_name(prompt)}"
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
