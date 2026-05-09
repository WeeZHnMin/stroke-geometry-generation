"""Feed real prefix strokes into a trained checkpoint and visualize the rollout.

For each chosen sample index, draws two panels side-by-side:
  - left  : ground-truth full sequence (prefix steps in gray, GT tail in green)
  - right : model rollout from the same prefix (prefix in gray, predicted tail in blue)

Also prints the per-sample tail token accuracy.

Example:
    python -m stroke_baseline.eval_prefix_visualize \
        --checkpoint runs/synth_shapes_v0/checkpoint.pt \
        --data generated_data/synth_shapes_v0.jsonl \
        --sample-indices 0 1 5 13 27 \
        --prefix-steps 5 \
        --rollout-steps 60 \
        --output runs/synth_shapes_v0/prefix_rollout.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .action_dataset import ActionTokenJsonlDataset
from .sample_action_tokens import load_model, token_from_phase_logits
from .visualize import strokes_to_xy


def _coords_from_prefix_tokens(tokenizer, prefix_tokens: list[int], total_len: int, device: torch.device) -> torch.Tensor:
    coords = torch.zeros(total_len, 2, dtype=torch.float32, device=device)
    x = y = 0.0
    pos = 0
    usable = len(prefix_tokens) - (len(prefix_tokens) % 3)
    for idx in range(0, usable, 3):
        for _ in range(3):
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
def _rollout_one(
    model,
    tokenizer,
    item: dict,
    prefix_tokens_count: int,
    rollout_tokens_count: int,
    device: torch.device,
) -> dict:
    """Teacher-force the prefix tokens, then autoregressively roll out for
    rollout_tokens_count more tokens. Returns a dict of tokens & strokes for both
    GT and predicted continuation."""
    target_ids = item["target_ids"]
    valid_target = target_ids[target_ids != -100].tolist()
    n_valid = len(valid_target)
    prefix_tokens_count = min(prefix_tokens_count, n_valid)
    if prefix_tokens_count < 3:
        raise ValueError("prefix must be at least 1 stroke step (3 tokens) after clipping")

    prefix_tokens = valid_target[:prefix_tokens_count]
    gt_tail_tokens = valid_target[prefix_tokens_count : prefix_tokens_count + rollout_tokens_count]

    cache = None
    context = None
    context_mask = None
    input_id = torch.tensor([tokenizer.start_id], dtype=torch.long, device=device)
    generated: list[int] = []

    # 1. Teacher-force the prefix to populate the KV cache.
    for pos, token_id in enumerate(prefix_tokens):
        coords = _coords_from_prefix_tokens(tokenizer, generated, pos + 1, device)[pos : pos + 1].unsqueeze(0)
        _, cache = model.decode_step(
            context=context, context_mask=context_mask,
            input_id=input_id, step_idx=pos, cache=cache, coords=coords,
        )
        generated.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

    # 2. Autoregressive rollout.
    pred_tail_tokens: list[int] = []
    for step in range(rollout_tokens_count):
        pos = prefix_tokens_count + step
        coords = _coords_from_prefix_tokens(tokenizer, generated, pos + 1, device)[pos : pos + 1].unsqueeze(0)
        out, cache = model.decode_step(
            context=context, context_mask=context_mask,
            input_id=input_id, step_idx=pos, cache=cache, coords=coords,
        )
        token_id = token_from_phase_logits(out, tokenizer, pos)
        pred_tail_tokens.append(token_id)
        generated.append(token_id)
        input_id = torch.tensor([token_id], dtype=torch.long, device=device)

    return {
        "prefix_tokens": prefix_tokens,
        "gt_tail_tokens": gt_tail_tokens,
        "pred_tail_tokens": pred_tail_tokens,
        "prefix_strokes": tokenizer.decode_tokens(prefix_tokens),
        "gt_full_strokes": tokenizer.decode_tokens(prefix_tokens + gt_tail_tokens),
        "pred_full_strokes": tokenizer.decode_tokens(generated),
    }


def _draw_panel(ax, strokes, n_prefix_steps: int, title: str, tail_color: str = "tab:blue") -> None:
    points = strokes_to_xy(strokes)
    prev = None
    for i, (x, y, pen) in enumerate(points):
        color = "gray" if i < n_prefix_steps else tail_color
        if pen == "move" or prev is None:
            prev = (x, y)
            continue
        ax.plot([prev[0], x], [prev[1], y], color=color, linewidth=1.6)
        prev = (x, y)
    if points:
        ax.scatter([points[0][0]], [points[0][1]], c="red", s=18, zorder=5, label="start")
        # Mark the prefix→tail boundary with an orange dot.
        if 0 < n_prefix_steps < len(points):
            x, y, _ = points[n_prefix_steps - 1]
            ax.scatter([x], [y], c="orange", s=18, zorder=5, label="prefix end")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0.55, -0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True,
                        help="Dataset sample indices to evaluate (multiple allowed).")
    parser.add_argument("--prefix-steps", type=int, default=5,
                        help="Number of stroke steps to teacher-force as prefix (1 step = 3 tokens).")
    parser.add_argument("--rollout-steps", type=int, default=60,
                        help="Number of stroke steps to autoregressively generate after the prefix.")
    parser.add_argument("--max-action-len", type=int, default=510)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--text-encoder-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device, text_encoder_dir=args.text_encoder_dir)
    dataset = ActionTokenJsonlDataset(
        args.data, action_tokenizer=tokenizer,
        max_action_len=args.max_action_len, progress_every=0,
    )

    prefix_tok = args.prefix_steps * 3
    rollout_tok = args.rollout_steps * 3

    rows = []
    for idx in args.sample_indices:
        if not (0 <= idx < len(dataset)):
            print(f"skip sample_index={idx} (dataset size {len(dataset)})")
            continue
        item = dataset[idx]
        prompt = "?"
        # Read the prompt from raw jsonl for the title.
        with open(args.data, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    prompt = json.loads(line).get("prompt", "?")
                    break

        try:
            result = _rollout_one(model, tokenizer, item, prefix_tok, rollout_tok, device)
        except ValueError as e:
            print(f"skip sample_index={idx}: {e}")
            continue

        gt_t = result["gt_tail_tokens"]
        pred_t = result["pred_tail_tokens"]
        n = min(len(gt_t), len(pred_t))
        tail_acc = sum(int(gt_t[i] == pred_t[i]) for i in range(n)) / max(n, 1)
        # Per-phase tail accuracy for finer signal.
        per_phase = [0, 0, 0]
        per_phase_count = [0, 0, 0]
        for i in range(n):
            phase = (prefix_tok + i) % 3
            per_phase_count[phase] += 1
            per_phase[phase] += int(gt_t[i] == pred_t[i])
        phase_acc = [per_phase[p] / max(per_phase_count[p], 1) for p in range(3)]

        rows.append({
            "idx": idx, "prompt": prompt,
            "gt_full": result["gt_full_strokes"],
            "pred_full": result["pred_full_strokes"],
            "tail_acc": tail_acc,
            "phase_acc": phase_acc,
            "n_pred_steps": len(result["pred_full_strokes"]),
            "n_gt_steps": len(result["gt_full_strokes"]),
        })
        print(f"sample {idx:3d}  prompt={prompt:10s}  "
              f"tail_acc={tail_acc:.3f} (dx={phase_acc[0]:.2f} dy={phase_acc[1]:.2f} pen={phase_acc[2]:.2f})  "
              f"pred_steps={len(result['pred_full_strokes'])}/{len(result['gt_full_strokes'])} (pred/gt)")

    if not rows:
        print("nothing to draw")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 3.4 * len(rows)), dpi=130)
    if len(rows) == 1:
        axes = axes[None, :]
    for r, s in enumerate(rows):
        _draw_panel(axes[r, 0], s["gt_full"], args.prefix_steps,
                    f"#{s['idx']} {s['prompt']} | GT", tail_color="tab:green")
        _draw_panel(axes[r, 1], s["pred_full"], args.prefix_steps,
                    f"#{s['idx']} {s['prompt']} | PRED  tail_acc={s['tail_acc']:.2f}",
                    tail_color="tab:blue")
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
