import argparse
import json
from pathlib import Path

import torch

from .action_model import ActionDecoderConfig, TextConditionedActionModel
from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .visualize import save_strokes_png


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    tokenizer = StrokeActionTokenizer(ActionTokenizerConfig(**state["action_tokenizer_cfg"]))
    cfg = ActionDecoderConfig(**state["decoder_cfg"])
    model = TextConditionedActionModel(
        cfg,
        text_encoder_dir=state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR)),
        max_text_len=state["max_text_len"],
    )
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from OpenVLA-style action-token stroke model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_action_tokens_easy/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
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
