import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .action_dataset import ActionTokenJsonlDataset, TwoStageActionTokenJsonlDataset
from .action_model import ActionDecoderConfig, TextConditionedActionModel
from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def apply_valid_token_mask(logits: torch.Tensor, tokenizer: StrokeActionTokenizer) -> torch.Tensor:
    seq_len = logits.size(1)
    positions = torch.arange(seq_len, device=logits.device)
    valid = tokenizer.valid_token_mask(positions)
    return logits.masked_fill(~valid[None, :, :], torch.finfo(logits.dtype).min)


def compute_loss(batch: dict, out: dict, tokenizer: StrokeActionTokenizer) -> tuple[torch.Tensor, dict[str, float]]:
    logits = apply_valid_token_mask(out["logits"], tokenizer)
    target = batch["target_ids"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)

    # 如果 batch 中有 start_position，加上回归损失
    if "start_position" in batch:
        start_loss = F.mse_loss(out["start_pred"], batch["start_position"])
        loss = loss + start_loss
        metrics = {"start_loss": float(start_loss.item())}
    else:
        metrics = {}

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        valid = target != -100
        acc = (pred[valid] == target[valid]).float().mean()
        metrics["loss"] = float(loss.item())
        metrics["token_acc"] = float(acc.item())
        positions = torch.arange(target.size(1), device=target.device)[None, :].expand_as(target)
        for name, phase in [("dx_acc", 0), ("dy_acc", 1), ("pen_acc", 2)]:
            phase_mask = valid & (positions % 3 == phase)
            if phase_mask.any():
                metrics[name] = float((pred[phase_mask] == target[phase_mask]).float().mean().item())
            else:
                metrics[name] = float("nan")
    return loss, metrics


def update_sums(totals: dict[str, float], counts: dict[str, int], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if value != value:
            continue
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1


def average(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {key: totals[key] / max(counts.get(key, 0), 1) for key in totals}


def save_checkpoint(
    output_dir: Path,
    model: TextConditionedActionModel,
    tokenizer: StrokeActionTokenizer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "decoder": model.decoder.state_dict(),
        "context_proj": model.context_proj.state_dict(),
        "start_head": model.start_head.state_dict(),
        "decoder_cfg": model.decoder.cfg.to_dict(),
        "action_tokenizer_cfg": tokenizer.cfg.to_dict(),
        "max_text_len": model.max_text_len,
        "text_encoder_dir": str(args.text_encoder_dir),
        "epoch": epoch,
    }
    torch.save(state, output_dir / "checkpoint.pt")
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
        out = model(prompts=list(batch["prompt"]), decoder_input_ids=batch["decoder_input_ids"], target_mask=batch["target_mask"])
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
        out = model(prompts=list(batch["prompt"]), decoder_input_ids=batch["decoder_input_ids"], target_mask=batch["target_mask"])
        _, metrics = compute_loss(batch, out, tokenizer)
        update_sums(totals, counts, metrics)
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenVLA-style per-dimension stroke action tokens.")
    parser.add_argument("--data", type=str, default="generated_data/bulk/stage1_foundation_shapes_v3_easy_20260403_125802.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_action_tokens_easy")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--min-value", type=float, default=-0.5)
    parser.add_argument("--max-value", type=float, default=0.5)
    parser.add_argument("--draw-min-value", type=float, default=-0.5)
    parser.add_argument("--draw-max-value", type=float, default=0.5)
    parser.add_argument("--two-stage", action="store_true", help="双阶段: 回归起点 + tight range draw 步")
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-action-len", type=int, default=510)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-variant", type=str, default="legacy", choices=["legacy_qkv", "legacy", "hetero"])
    parser.add_argument("--trend-kernel-size", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 双阶段模式使用更小的 draw 量化范围
    action_tokenizer = StrokeActionTokenizer(
        ActionTokenizerConfig(
            bins=args.bins,
            min_value=args.min_value,
            max_value=args.max_value,
            draw_min_value=args.draw_min_value,
            draw_max_value=args.draw_max_value,
        )
    )

    if args.two_stage:
        dataset = TwoStageActionTokenJsonlDataset(
            args.data,
            action_tokenizer=action_tokenizer,
            max_action_len=args.max_action_len,
            limit=args.limit,
        )
    else:
        dataset = ActionTokenJsonlDataset(
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

    cfg = ActionDecoderConfig(
        action_vocab_size=action_tokenizer.vocab_size,
        pad_token_id=action_tokenizer.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_action_len=args.max_action_len,
        attention_variant=args.attention_variant,
        trend_kernel_size=args.trend_kernel_size,
    )
    model = TextConditionedActionModel(cfg, text_encoder_dir=args.text_encoder_dir, max_text_len=args.max_text_len).to(device)
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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, action_tokenizer, epoch, step_log_path)
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
