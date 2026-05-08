import argparse
import json
from pathlib import Path

import torch

from .action_model import ActionDecoderConfig, TextConditionedActionModel
from .action_tokenizer import ActionTokenizerConfig, CompactActionTokenMapper, StrokeActionTokenizer
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .visualize import save_strokes_png


def load_model(checkpoint_path: str | Path, device: torch.device, text_encoder_dir: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    tokenizer = StrokeActionTokenizer(ActionTokenizerConfig(**state["action_tokenizer_cfg"]))
    compact_mapper = CompactActionTokenMapper.from_vocab_file(state["compact_vocab_file"]) if state.get("compact_vocab_file") else None
    cfg = ActionDecoderConfig(**state["decoder_cfg"])
    enc_dir = text_encoder_dir or state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR))
    model = TextConditionedActionModel(
        cfg,
        text_encoder_dir=enc_dir,
        max_text_len=state["max_text_len"],
    )
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.to(device)
    model.eval()
    return model, tokenizer, compact_mapper


@torch.no_grad()
def generate_tokens(
    model: TextConditionedActionModel,
    tokenizer: StrokeActionTokenizer,
    compact_mapper: CompactActionTokenMapper | None,
    prompt: str,
    max_steps: int = 192,
    device: torch.device | str = "cpu",
) -> list[int]:
    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}
    cache = None
    abs_x = 0.0
    abs_y = 0.0
    input_coords = torch.zeros(1, 2, dtype=torch.float32, device=device)
    input_pen_states = torch.tensor([tokenizer.start_input_pen_id], dtype=torch.long, device=device)
    tokens: list[int] = []

    for step_idx in range(max_steps):
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            input_coords=input_coords,
            input_pen_states=input_pen_states,
            step_idx=step_idx,
            cache=cache,
        )
        logits = out["logits"][0, 0].clone()
        if compact_mapper is None:
            logits[tokenizer.pad_id] = torch.finfo(logits.dtype).min
            token_id = int(logits.argmax(dim=-1).item())
            raw_token_id = token_id
        else:
            logits[compact_mapper.pad_id] = torch.finfo(logits.dtype).min
            token_id = int(logits.argmax(dim=-1).item())
            raw_token_id = compact_mapper.decode(token_id)
        tokens.append(raw_token_id)
        x, y, pen_state = tokenizer.decode_action(raw_token_id)
        input_coords = torch.tensor([[x, y]], dtype=torch.float32, device=device)
        input_pen_states = torch.tensor([pen_state], dtype=torch.long, device=device)
        if pen_state == 2:
            break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from CPCF-input Cartesian action-token stroke model.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_action_tokens_cpcf/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--png", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, compact_mapper = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)
    tokens = generate_tokens(model, tokenizer, compact_mapper, args.prompt, max_steps=args.max_steps, device=device)
    strokes = tokenizer.decode_tokens(tokens)
    payload = {"prompt": args.prompt, "tokens": tokens, "strokes": strokes}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
