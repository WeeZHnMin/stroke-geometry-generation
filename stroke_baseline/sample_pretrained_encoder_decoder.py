import argparse
import json
from pathlib import Path

import torch

from .dataset import ID_TO_PEN, START_PEN_ID
from .pretrained_encoder_decoder import (
    StrokeDecoderConfig,
    TextConditionedStrokeModel,
)
from .visualize import save_strokes_png


def load_model(checkpoint_path: str | Path, device: torch.device) -> TextConditionedStrokeModel:
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    cfg = StrokeDecoderConfig(**state["decoder_cfg"])
    model = TextConditionedStrokeModel(
        decoder_cfg=cfg,
        text_encoder_dir=state["text_encoder_dir"],
        max_text_len=state["max_text_len"],
    )
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_strokes(
    model: TextConditionedStrokeModel,
    prompt: str,
    max_steps: int = 64,
    device: torch.device | str = "cpu",
) -> list[dict]:
    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}

    cache = None
    last_dxdy = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
    last_pen = torch.full((1, 1), START_PEN_ID, dtype=torch.long, device=device)

    strokes = []
    for step_idx in range(max_steps):
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_dxdy=last_dxdy,
            decoder_pen=last_pen,
            step_idx=step_idx,
            cache=cache,
        )
        dxdy = out["pred_dxdy"][0, 0].detach().cpu()
        pen_id = int(out["pred_pen_logits"][0, 0].argmax().item())
        pen_state = ID_TO_PEN[pen_id]
        strokes.append({"dx": float(dxdy[0]), "dy": float(dxdy[1]), "pen_state": pen_state})

        last_dxdy = out["pred_dxdy"]
        last_pen = torch.tensor([[pen_id]], dtype=torch.long, device=device)
        if pen_state == "end_all":
            break

    return strokes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from frozen Chinese encoder + stroke decoder checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_mvp_easy_5000_bs16/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--png", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    strokes = generate_strokes(model, args.prompt, max_steps=args.max_steps, device=device)
    print(json.dumps({"prompt": args.prompt, "strokes": strokes}, indent=2, ensure_ascii=False))

    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
