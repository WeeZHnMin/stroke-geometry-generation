import argparse
import json
from pathlib import Path

import torch

from .action_model import ActionDecoderConfig, TextConditionedActionModel
from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .visualize import save_strokes_png


def load_model(checkpoint_path: str | Path, device: torch.device, text_encoder_dir: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    tokenizer = StrokeActionTokenizer(ActionTokenizerConfig(**state["action_tokenizer_cfg"]))
    cfg = ActionDecoderConfig(**state["decoder_cfg"])
    enc_dir = text_encoder_dir or state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR))
    model = TextConditionedActionModel(
        cfg,
        text_encoder_dir=enc_dir,
        max_text_len=state["max_text_len"],
    )
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    # 双阶段: 加载 start_head（如果存在）
    if "start_head" in state:
        model.start_head.load_state_dict(state["start_head"])
    model.to(device)
    model.eval()
    return model, tokenizer


def mask_logits_for_position(logits: torch.Tensor, tokenizer: StrokeActionTokenizer, position: int) -> torch.Tensor:
    positions = torch.tensor([position], device=logits.device)
    valid = tokenizer.valid_token_mask(positions)[0]
    return logits.masked_fill(~valid[None, :], torch.finfo(logits.dtype).min)


@torch.no_grad()
def generate_tokens(
    model: TextConditionedActionModel,
    tokenizer: StrokeActionTokenizer,
    prompt: str,
    max_steps: int = 64,
    device: torch.device | str = "cpu",
) -> list[int]:
    """单阶段推理: 自回归预测所有步骤（含第一步 move）。"""
    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}
    cache = None
    input_id = torch.tensor([tokenizer.start_id], dtype=torch.long, device=device)
    tokens: list[int] = []

    for pos in range(max_steps * 3):
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            input_id=input_id,
            step_idx=pos,
            cache=cache,
        )
        logits = mask_logits_for_position(out["logits"][0, 0], tokenizer, pos)
        token_id = int(logits.argmax(dim=-1).item())
        tokens.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

        if pos % 3 == 2:
            pen_id = token_id - tokenizer.pen_offset
            if pen_id >= 0 and pen_id < 4 and tokenizer.decode_tokens(tokens)[-1]["pen_state"] == "end_all":
                break
    return tokens


@torch.no_grad()
def generate_two_stage(
    model: TextConditionedActionModel,
    tokenizer: StrokeActionTokenizer,
    prompt: str,
    max_steps: int = 64,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """双阶段推理:
       1. 回归预测起点位置 (start_x, start_y)
       2. 自回归预测 draw 步的离散 token（tight range + 通用 pen token）
    """
    # 阶段1: 预测起点
    start_pred = model.predict_start([prompt])[0].detach().cpu()  # [2]
    start_x, start_y = float(start_pred[0]), float(start_pred[1])

    strokes = [{"dx": start_x, "dy": start_y, "pen_state": "move"}]

    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}
    cache = None
    input_id = torch.tensor([tokenizer.start_id], dtype=torch.long, device=device)
    tokens: list[int] = []

    for pos in range(max_steps * 3):
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            input_id=input_id,
            step_idx=pos,
            cache=cache,
        )
        logits = mask_logits_for_position(out["logits"][0, 0], tokenizer, pos)
        token_id = int(logits.argmax(dim=-1).item())
        tokens.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

        if pos % 3 == 2:
            # 用 draw tokenizer 解码
            step = tokenizer.decode_draw_step(
                tokens[-3], tokens[-2], tokens[-1]
            )
            strokes.append(step)
            if step["pen_state"] == "end_all":
                break

    return strokes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from OpenVLA-style action-token stroke model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_action_tokens_easy/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--png", type=str, default=None)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    parser.add_argument("--two-stage", action="store_true", help="双阶段推理模式")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)

    if args.two_stage:
        strokes = generate_two_stage(model, tokenizer, args.prompt, max_steps=args.max_steps, device=device)
        print(json.dumps({"prompt": args.prompt, "strokes": strokes}, indent=2, ensure_ascii=False))
    else:
        tokens = generate_tokens(model, tokenizer, args.prompt, max_steps=args.max_steps, device=device)
        strokes = tokenizer.decode_tokens(tokens)
        print(json.dumps({"prompt": args.prompt, "tokens": tokens, "strokes": strokes}, indent=2, ensure_ascii=False))

    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
