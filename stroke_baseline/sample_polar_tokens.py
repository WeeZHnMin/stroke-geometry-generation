import argparse
import json
from pathlib import Path

import torch

from .polar_model import PolarDecoderConfig, TextConditionedPolarModel
from .polar_tokenizer import CompactPolarTokenMapper, PolarActionTokenizer, PolarActionTokenizerConfig
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .visualize import save_strokes_png


def load_model(checkpoint_path: str | Path, device: torch.device, text_encoder_dir: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    compact_mapper = None
    if state.get("compact_vocab_file"):
        tokenizer = PolarActionTokenizer.from_vocab_file(state["compact_vocab_file"])
    else:
        tokenizer = PolarActionTokenizer(PolarActionTokenizerConfig(**state["polar_tokenizer_cfg"]))
    if state.get("compact_vocab_file"):
        compact_mapper = CompactPolarTokenMapper.from_vocab_file(state["compact_vocab_file"])
    action_cfg = state["decoder_cfg"]
    cfg = PolarDecoderConfig(
        vocab_size=action_cfg["action_vocab_size"],
        pad_token_id=action_cfg["pad_token_id"],
        d_model=action_cfg["d_model"],
        n_heads=action_cfg["n_heads"],
        num_decoder_layers=action_cfg["num_decoder_layers"],
        ff_mult=action_cfg["ff_mult"],
        dropout=action_cfg["dropout"],
        max_action_len=action_cfg["max_action_len"],
        attention_variant=action_cfg.get("attention_variant", "legacy"),
        trend_kernel_size=action_cfg.get("trend_kernel_size", 5),
    )
    enc_dir = text_encoder_dir or state.get("text_encoder_dir", str(DEFAULT_TEXT_ENCODER_DIR))
    model = TextConditionedPolarModel(cfg, text_encoder_dir=enc_dir, max_text_len=state["max_text_len"])
    model.decoder.load_state_dict(state["decoder"])
    model.context_proj.load_state_dict(state["context_proj"])
    model.to(device)
    model.eval()
    return model, tokenizer, compact_mapper


@torch.no_grad()
def generate_tokens(
    model: TextConditionedPolarModel,
    tokenizer: PolarActionTokenizer,
    compact_mapper: CompactPolarTokenMapper | None,
    prompt: str,
    max_steps: int = 192,
    device: torch.device | str = "cpu",
) -> list[int]:
    text = model.encode_text([prompt])
    text = {key: value.to(device) for key, value in text.items()}
    cache = None
    start_id = compact_mapper.start_id if compact_mapper is not None else tokenizer.start_id
    input_id = torch.tensor([start_id], dtype=torch.long, device=device)
    tokens: list[int] = []
    for pos in range(max_steps):
        out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            input_id=input_id,
            step_idx=pos,
            cache=cache,
        )
        logits = out["logits"][0, 0]
        if compact_mapper is not None:
            logits[compact_mapper.action_vocab_size :] = torch.finfo(logits.dtype).min
        else:
            logits[tokenizer.action_vocab_size :] = torch.finfo(logits.dtype).min
        token_id = int(logits.argmax(dim=-1).item())
        tokens.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)
        raw_token_id = compact_mapper.decode(token_id) if compact_mapper is not None else token_id
        stroke = tokenizer.token_to_stroke(raw_token_id)
        if stroke["pen_state"] == "end_all":
            break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from text-to-polar-action-token baseline.")
    parser.add_argument("--checkpoint", type=str, default="runs/stroke_polar_scale8/checkpoint.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--png", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, compact_mapper = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)
    tokens = generate_tokens(model, tokenizer, compact_mapper, args.prompt, max_steps=args.max_steps, device=device)
    raw_tokens = [compact_mapper.decode(token) for token in tokens] if compact_mapper is not None else tokens
    strokes = tokenizer.decode_tokens(raw_tokens)
    payload = {"prompt": args.prompt, "tokens": tokens, "strokes": strokes}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.png:
        save_strokes_png(strokes, args.png, title=args.prompt)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
