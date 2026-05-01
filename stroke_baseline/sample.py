import argparse
import json
from pathlib import Path

import torch

from .dataset import ID_TO_PEN, START_PEN_ID
from .model import StrokeTransformerBaseline, StrokeTransformerConfig
from .tokenizer import CharTokenizer, HFTokenizer
from .visualize import save_strokes_png


def load_model(checkpoint_dir: str | Path, device: str | torch.device = "cpu"):
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = HFTokenizer.load(tokenizer_dir)
    else:
        tokenizer = CharTokenizer.load(checkpoint_dir / "tokenizer.json")
    cfg = StrokeTransformerConfig(**json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8")))
    model = StrokeTransformerBaseline(cfg)
    state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_strokes(
    model: StrokeTransformerBaseline,
    tokenizer: CharTokenizer,
    prompt: str,
    max_steps: int = 64,
    device: str | torch.device = "cpu",
) -> list[dict]:
    cfg = model.cfg
    text_ids = torch.tensor([tokenizer.encode(prompt, cfg.max_text_len)], dtype=torch.long, device=device)
    text_mask = text_ids != tokenizer.pad_id

    decoder_dxdy = torch.zeros(1, cfg.max_stroke_len, 2, dtype=torch.float32, device=device)
    decoder_pen = torch.full((1, cfg.max_stroke_len), START_PEN_ID, dtype=torch.long, device=device)
    target_mask = torch.zeros(1, cfg.max_stroke_len, dtype=torch.bool, device=device)

    strokes = []
    steps = min(max_steps, cfg.max_stroke_len)
    for step_idx in range(steps):
        target_mask[:, : step_idx + 1] = True
        out = model(text_ids, text_mask, decoder_dxdy, decoder_pen, target_mask=target_mask)
        dxdy = out["pred_dxdy"][0, step_idx].detach().cpu()
        pen_id = int(out["pred_pen_logits"][0, step_idx].argmax().item())
        pen_state = ID_TO_PEN[pen_id]
        step = {"dx": float(dxdy[0]), "dy": float(dxdy[1]), "pen_state": pen_state}
        strokes.append(step)

        if step_idx + 1 < cfg.max_stroke_len:
            decoder_dxdy[0, step_idx + 1] = out["pred_dxdy"][0, step_idx]
            decoder_pen[0, step_idx + 1] = pen_id
        if pen_state == "end_all":
            break

    return strokes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strokes from a trained baseline checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/stroke_baseline")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--png", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint_dir, device=device)
    strokes = generate_strokes(model, tokenizer, args.prompt, max_steps=args.max_steps, device=device)
    print(json.dumps({"prompt": args.prompt, "strokes": strokes}, indent=2, ensure_ascii=False))
    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
