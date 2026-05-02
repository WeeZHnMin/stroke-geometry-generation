from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.dataset import ID_TO_PEN
from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from stroke_baseline.sample_action_tokens import mask_logits_for_position
from stroke_baseline.visualize import save_strokes_png

from .geometry_dataset import GEOMETRY_STATE_DIM, _geometry_row
from .geometry_model import GeometryActionDecoderConfig, TextConditionedGeometryActionModel


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    tokenizer = StrokeActionTokenizer(ActionTokenizerConfig(**state["action_tokenizer_cfg"]))
    cfg = GeometryActionDecoderConfig(**state["decoder_cfg"])
    model = TextConditionedGeometryActionModel(
        cfg,
        text_encoder_dir=state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR)),
        max_text_len=state["max_text_len"],
    )
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.to(device)
    model.eval()
    return model, tokenizer


def geometry_tensor(
    *,
    x: float,
    y: float,
    pending_dx: float,
    pending_dy: float,
    start_x: float,
    start_y: float,
    bbox_min_x: float,
    bbox_min_y: float,
    bbox_max_x: float,
    bbox_max_y: float,
    progress: float,
    device: torch.device | str,
) -> torch.Tensor:
    row = _geometry_row(
        x=x,
        y=y,
        pending_dx=pending_dx,
        pending_dy=pending_dy,
        start_x=start_x,
        start_y=start_y,
        bbox_min_x=bbox_min_x,
        bbox_min_y=bbox_min_y,
        bbox_max_x=bbox_max_x,
        bbox_max_y=bbox_max_y,
        progress=progress,
    )
    return torch.tensor(row, dtype=torch.float32, device=device).reshape(1, GEOMETRY_STATE_DIM)


@torch.no_grad()
def generate_tokens(
    model: TextConditionedGeometryActionModel,
    tokenizer: StrokeActionTokenizer,
    prompt: str,
    max_steps: int = 170,
    device: torch.device | str = "cpu",
) -> list[int]:
    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}
    cache = None
    input_id = torch.tensor([tokenizer.start_id], dtype=torch.long, device=device)
    tokens: list[int] = []

    x = y = 0.0
    start_x = start_y = 0.0
    bbox_min_x = bbox_max_x = 0.0
    bbox_min_y = bbox_max_y = 0.0
    pending_dx = pending_dy = 0.0

    for pos in range(max_steps * 3):
        phase = pos % 3
        geom = geometry_tensor(
            x=x,
            y=y,
            pending_dx=pending_dx if phase in {1, 2} else 0.0,
            pending_dy=pending_dy if phase == 2 else 0.0,
            start_x=start_x,
            start_y=start_y,
            bbox_min_x=bbox_min_x,
            bbox_min_y=bbox_min_y,
            bbox_max_x=bbox_max_x,
            bbox_max_y=bbox_max_y,
            progress=pos / max(1, max_steps * 3),
            device=device,
        )
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            input_id=input_id,
            geometry_state=geom,
            step_idx=pos,
            cache=cache,
        )
        logits = mask_logits_for_position(out["logits"][0, 0], tokenizer, pos)
        token_id = int(logits.argmax(dim=-1).item())
        tokens.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

        if phase == 0:
            pending_dx = tokenizer._bin_to_value(token_id - tokenizer.dx_offset)
        elif phase == 1:
            pending_dy = tokenizer._bin_to_value(token_id - tokenizer.dy_offset)
        else:
            pen_id = token_id - tokenizer.pen_offset
            x += pending_dx
            y += pending_dy
            bbox_min_x = min(bbox_min_x, x)
            bbox_min_y = min(bbox_min_y, y)
            bbox_max_x = max(bbox_max_x, x)
            bbox_max_y = max(bbox_max_y, y)
            pending_dx = pending_dy = 0.0
            if 0 <= pen_id < len(ID_TO_PEN):
                pen_state = ID_TO_PEN[pen_id]
                if pen_state in {"end_shape", "end_all"}:
                    start_x = x
                    start_y = y
                if pen_state == "end_all":
                    break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from geometry-state-conditioned action-token stroke model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_geometry_action_chinese_mvp/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=170)
    parser.add_argument("--png", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)
    tokens = generate_tokens(model, tokenizer, args.prompt, max_steps=args.max_steps, device=device)
    strokes = tokenizer.decode_tokens(tokens)
    print(json.dumps({"prompt": args.prompt, "tokens": tokens, "strokes": strokes}, indent=2, ensure_ascii=False))
    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
