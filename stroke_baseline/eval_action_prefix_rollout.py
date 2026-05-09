import argparse
import json
import random
from pathlib import Path

import torch

from .action_dataset import ActionTokenJsonlDataset
from .sample_action_tokens import load_model, token_from_phase_logits
from .visualize import save_strokes_png


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_sample(dataset_path: str, checkpoint_path: str, sample_index: int, limit: int | None, max_action_len: int):
    device = torch.device("cpu")
    _, tokenizer = load_model(checkpoint_path, device)
    dataset = ActionTokenJsonlDataset(
        dataset_path,
        action_tokenizer=tokenizer,
        max_action_len=max_action_len,
        limit=limit,
        progress_every=0,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} out of range for dataset size {len(dataset)}")
    return dataset[sample_index], tokenizer


def coords_from_prefix_tokens(tokenizer, prefix_tokens: list[int], total_len: int, device: torch.device) -> torch.Tensor:
    coords = torch.zeros(total_len, 2, dtype=torch.float32, device=device)
    x = 0.0
    y = 0.0
    pos = 0
    usable = len(prefix_tokens) - (len(prefix_tokens) % 3)
    for idx in range(0, usable, 3):
        for phase in range(3):
            if pos >= total_len:
                break
            coords[pos, 0] = x
            coords[pos, 1] = y
            pos += 1
        step = tokenizer.decode_step(prefix_tokens[idx], prefix_tokens[idx + 1], prefix_tokens[idx + 2])
        x += float(step["dx"])
        y += float(step["dy"])
    while pos < total_len:
        coords[pos, 0] = x
        coords[pos, 1] = y
        pos += 1
    return coords


@torch.no_grad()
def rollout_from_prefix(
    checkpoint: str,
    dataset_path: str,
    sample_index: int,
    prefix_len: int,
    rollout_steps: int,
    limit: int | None,
    max_action_len: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(checkpoint, device)
    item, _ = load_eval_sample(dataset_path, checkpoint, sample_index, limit, max_action_len)

    target_ids = item["target_ids"]
    valid_target = target_ids[target_ids != -100].tolist()
    prefix_len = min(prefix_len, len(valid_target))
    prefix_tokens = valid_target[:prefix_len]
    gt_tail = valid_target[prefix_len : prefix_len + rollout_steps]

    if not prefix_tokens:
        raise ValueError("prefix_len must be at least 1 after clipping.")

    cache = None
    context = None
    context_mask = None
    input_id = torch.tensor([tokenizer.start_id], dtype=torch.long, device=device)
    generated = []

    for pos, token_id in enumerate(prefix_tokens):
        coords = coords_from_prefix_tokens(tokenizer, generated, pos + 1, device)[pos : pos + 1].unsqueeze(0)
        out, cache = model.decode_step(
            context=context,
            context_mask=context_mask,
            input_id=input_id,
            step_idx=pos,
            cache=cache,
            coords=coords,
        )
        generated.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

    pred_tail = []
    for step in range(rollout_steps):
        pos = prefix_len + step
        coords = coords_from_prefix_tokens(tokenizer, generated, pos + 1, device)[pos : pos + 1].unsqueeze(0)
        out, cache = model.decode_step(
            context=context,
            context_mask=context_mask,
            input_id=input_id,
            step_idx=pos,
            cache=cache,
            coords=coords,
        )
        token_id = token_from_phase_logits(out, tokenizer, pos)
        pred_tail.append(token_id)
        generated.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

    gt_prefix_strokes = tokenizer.decode_tokens(prefix_tokens)
    gt_tail_strokes = tokenizer.decode_tokens(gt_tail)
    pred_tail_strokes = tokenizer.decode_tokens(pred_tail)
    return {
        "prefix_tokens": prefix_tokens,
        "gt_tail_tokens": gt_tail,
        "pred_tail_tokens": pred_tail,
        "prefix_strokes": gt_prefix_strokes,
        "gt_tail_strokes": gt_tail_strokes,
        "pred_tail_strokes": pred_tail_strokes,
        "full_pred_strokes": tokenizer.decode_tokens(generated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate decoder-only autoregressive rollout from a real validation prefix.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=24, help="Prefix token length, not stroke length.")
    parser.add_argument("--rollout-steps", type=int, default=48, help="Number of tokens to autoregressively predict.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-action-len", type=int, default=384)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-png", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    result = rollout_from_prefix(
        checkpoint=args.checkpoint,
        dataset_path=args.data,
        sample_index=args.sample_index,
        prefix_len=args.prefix_len,
        rollout_steps=args.rollout_steps,
        limit=args.limit,
        max_action_len=args.max_action_len,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_png:
        output_png = Path(args.output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        save_strokes_png(result["full_pred_strokes"], str(output_png), title=f"prefix={args.prefix_len} rollout={args.rollout_steps}")
        print(f"saved {output_png}")


if __name__ == "__main__":
    main()
