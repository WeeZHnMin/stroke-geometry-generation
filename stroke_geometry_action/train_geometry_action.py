from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from stroke_baseline.train_action_tokens import compute_loss, update_sums, average

from .geometry_dataset import GeometryActionTokenJsonlDataset
from .geometry_model import (
    DEFAULT_GEOMETRY_ACTION_BINS,
    GeometryActionDecoderConfig,
    TextConditionedGeometryActionModel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def save_checkpoint(
    output_dir: Path,
    model: TextConditionedGeometryActionModel,
    tokenizer: StrokeActionTokenizer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "context_proj": model.context_proj.state_dict(),
            "decoder_cfg": model.decoder.cfg.to_dict(),
            "action_tokenizer_cfg": tokenizer.cfg.to_dict(),
            "max_text_len": model.max_text_len,
            "text_encoder_dir": str(args.text_encoder_dir),
            "epoch": epoch,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def train_one_epoch(model, loader, optimizer, device, args, tokenizer, epoch, step_log_path: Path):
    model.train()
    model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_input_ids=batch["decoder_input_ids"],
            geometry_states=batch["geometry_states"],
            target_mask=batch["target_mask"],
        )
        loss, metrics = compute_loss(batch, out, tokenizer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        update_sums(totals, counts, metrics)
        if step % args.log_every == 0:
            line = (
                f"epoch={epoch} step={step}/{len(loader)} loss={metrics['loss']:.4f} "
                f"acc={metrics['token_acc']:.3f} dx={metrics['dx_acc']:.3f} "
                f"dy={metrics['dy_acc']:.3f} pen={metrics['pen_acc']:.3f}"
            )
            print(line, flush=True)
            append_jsonl(
                step_log_path,
                {
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(loader),
                    "split": "train",
                    **metrics,
                },
            )
    return average(totals, counts)


@torch.no_grad()
def evaluate(model, loader, device, tokenizer):
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_input_ids=batch["decoder_input_ids"],
            geometry_states=batch["geometry_states"],
            target_mask=batch["target_mask"],
        )
        _, metrics = compute_loss(batch, out, tokenizer)
        update_sums(totals, counts, metrics)
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train geometry-state-conditioned action-token stroke model.")
    parser.add_argument("--data", type=str, default="generated_data/chinese_mvp/chinese_mvp_single_basic_20260501_173548.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_geometry_action_chinese_mvp")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=DEFAULT_GEOMETRY_ACTION_BINS)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-action-len", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_tokenizer = StrokeActionTokenizer(
        ActionTokenizerConfig(bins=args.bins, min_value=args.min_value, max_value=args.max_value)
    )
    dataset = GeometryActionTokenJsonlDataset(
        args.data,
        action_tokenizer=action_tokenizer,
        max_action_len=args.max_action_len,
        limit=args.limit,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = GeometryActionDecoderConfig(
        action_vocab_size=action_tokenizer.vocab_size,
        pad_token_id=action_tokenizer.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_action_len=args.max_action_len,
    )
    model = TextConditionedGeometryActionModel(cfg, text_encoder_dir=args.text_encoder_dir, max_text_len=args.max_text_len).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = output_dir / "step_metrics.jsonl"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"

    print(
        f"device={device} train={train_size} val={val_size} vocab={action_tokenizer.vocab_size} "
        f"decoder_params={sum(p.numel() for p in model.decoder.parameters()):,} "
        f"step_log={step_log_path} epoch_log={epoch_log_path}",
        flush=True,
    )
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            action_tokenizer,
            epoch,
            step_log_path,
        )
        val_metrics = evaluate(model, val_loader, device, action_tokenizer)
        epoch_line = (
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['token_acc']:.3f} val_dx={val_metrics['dx_acc']:.3f} "
            f"val_dy={val_metrics['dy_acc']:.3f} val_pen={val_metrics['pen_acc']:.3f}"
        )
        print(epoch_line, flush=True)
        is_best = val_metrics["loss"] < best_val
        append_jsonl(
            epoch_log_path,
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss_before_update": best_val if math.isfinite(best_val) else None,
                "is_best": is_best,
            },
        )
        if is_best:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir, model, action_tokenizer, args, epoch)

    print(f"saved best checkpoint to {output_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
