import argparse
import json
from pathlib import Path

import torch

from .sample_pretrained_encoder_decoder import generate_strokes, load_model
from .visualize import save_strokes_png


DEFAULT_PROMPTS = [
    "draw a small arc on the right",
    "draw a medium regular polygon on the right",
    "draw a medium wide rectangle on the right",
    "draw a medium square near the center",
    "draw a large cubic function graph near the center",
    "draw a large right triangle near the center",
    "draw a large kite shape near the center",
    "draw a large cosine wave near the center",
    "draw a large linear function graph near the center",
]


def safe_name(text: str) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")
    name = "".join(keep).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch sample prompts from the MVP stroke model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_mvp_easy_5000_bs16/checkpoint.pt")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_mvp_easy_5000_bs16/samples")
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    summary = []
    for idx, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        strokes = generate_strokes(model, prompt, max_steps=args.max_steps, device=device)
        stem = f"{idx:02d}_{safe_name(prompt)}"
        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"

        save_strokes_png(strokes, png_path, title=prompt)
        json_path.write_text(
            json.dumps({"prompt": prompt, "strokes": strokes}, indent=2, ensure_ascii=False),
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
