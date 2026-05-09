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
from .dataset import PEN_TO_ID
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_loss(batch: dict, out: dict, tokenizer: StrokeActionTokenizer) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target_ids"]
    positions = torch.arange(target.size(1), device=target.device)[None, :].expand_as(target)
    valid = target != -100
    phase0 = valid & (positions % 3 == 0)
    phase1 = valid & (positions % 3 == 1)
    phase2 = valid & (positions % 3 == 2)

    loss_terms: list[torch.Tensor] = []
    metrics: dict[str, float] = {}

    if phase0.any():
        dx_target = target[phase0] - tokenizer.dx_offset
        dx_loss = F.cross_entropy(out["dx_logits"][phase0], dx_target)
        loss_terms.append(dx_loss)
        metrics["dx_loss"] = float(dx_loss.item())
    else:
        metrics["dx_loss"] = float("nan")

    if phase1.any():
        dy_target = target[phase1] - tokenizer.dy_offset
        dy_loss = F.cross_entropy(out["dy_logits"][phase1], dy_target)
        loss_terms.append(dy_loss)
        metrics["dy_loss"] = float(dy_loss.item())
    else:
        metrics["dy_loss"] = float("nan")

    if phase2.any():
        pen_target = target[phase2] - tokenizer.pen_offset
        pen_loss = F.cross_entropy(out["pen_logits"][phase2], pen_target)
        loss_terms.append(pen_loss)
        metrics["pen_loss"] = float(pen_loss.item())
    else:
        metrics["pen_loss"] = float("nan")

    if not loss_terms:
        raise ValueError("No valid targets found in batch")
    loss = torch.stack(loss_terms).mean()

    # 如果 batch 中有 start_position，加上回归损失
    if "start_position" in batch:
        start_loss = F.mse_loss(out["start_pred"], batch["start_position"])
        loss = loss + start_loss
        metrics["start_loss"] = float(start_loss.item())
    else:
        metrics["start_loss"] = float("nan")

    with torch.no_grad():
        metrics["loss"] = float(loss.item())
        total_correct = 0
        total_count = 0

        if phase0.any():
            dx_pred = out["dx_logits"][phase0].argmax(dim=-1)
            dx_target = target[phase0] - tokenizer.dx_offset
            dx_correct = dx_pred == dx_target
            metrics["dx_acc"] = float(dx_correct.float().mean().item())
            total_correct += int(dx_correct.sum().item())
            total_count += int(dx_correct.numel())
        else:
            metrics["dx_acc"] = float("nan")

        if phase1.any():
            dy_pred = out["dy_logits"][phase1].argmax(dim=-1)
            dy_target = target[phase1] - tokenizer.dy_offset
            dy_correct = dy_pred == dy_target
            metrics["dy_acc"] = float(dy_correct.float().mean().item())
            total_correct += int(dy_correct.sum().item())
            total_count += int(dy_correct.numel())
        else:
            metrics["dy_acc"] = float("nan")

        if phase2.any():
            pen_pred = out["pen_logits"][phase2].argmax(dim=-1)
            pen_target = target[phase2] - tokenizer.pen_offset
            pen_correct = pen_pred == pen_target
            metrics["pen_acc"] = float(pen_correct.float().mean().item())
            total_correct += int(pen_correct.sum().item())
            total_count += int(pen_correct.numel())
        else:
            metrics["pen_acc"] = float("nan")

        metrics["token_acc"] = float(total_correct / max(total_count, 1))
    return loss, metrics


def update_sums(totals: dict[str, float], counts: dict[str, int], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if value != value:
            continue
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1


def average(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {key: totals[key] / max(counts.get(key, 0), 1) for key in totals}


def sanitize_record(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, dict):
            cleaned[key] = sanitize_record(value)
        elif isinstance(value, float) and value != value:
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


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
        "text_encoder_dir": None if args.decoder_only_pretrain else str(args.text_encoder_dir),
        "epoch": epoch,
    }
    torch.save(state, output_dir / "checkpoint.pt")
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_record(record), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def train_one_epoch(model, loader, optimizer, device, args, tokenizer, epoch, step_log_path: Path):
    model.train()
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            prompts=None if args.decoder_only_pretrain else list(batch["prompt"]),
            decoder_input_ids=batch["decoder_input_ids"],
            target_mask=batch["target_mask"],
            coords=batch.get("coords"),
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
def evaluate(model, loader, device, tokenizer, args):
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            prompts=None if args.decoder_only_pretrain else list(batch["prompt"]),
            decoder_input_ids=batch["decoder_input_ids"],
            target_mask=batch["target_mask"],
            coords=batch.get("coords"),
        )
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
    parser.add_argument("--trend-kernel-size", type=int, default=12)
    parser.add_argument("--decoder-only-pretrain", action="store_true", help="Disable text conditioning and cross-attention for decoder-only pretraining.")
    parser.add_argument("--use-distance-bias", action="store_true", help="Enable MLP distance bias in self-attention.")
    parser.add_argument("--distance-bias-hidden", type=int, default=32)
    parser.add_argument("--dataset-check-ranges", action="store_true", help="Validate dx/dy range while indexing the dataset.")
    parser.add_argument("--dataset-index-progress-every", type=int, default=10000)
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
            check_ranges=args.dataset_check_ranges,
            progress_every=args.dataset_index_progress_every,
        )
    else:
        dataset = ActionTokenJsonlDataset(
            args.data,
            action_tokenizer=action_tokenizer,
            max_action_len=args.max_action_len,
            limit=args.limit,
            check_ranges=args.dataset_check_ranges,
            progress_every=args.dataset_index_progress_every,
        )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = ActionDecoderConfig(
        action_vocab_size=action_tokenizer.vocab_size,
        pad_token_id=action_tokenizer.pad_id,
        dx_vocab_size=action_tokenizer.bins,
        dy_vocab_size=action_tokenizer.bins,
        pen_vocab_size=len(PEN_TO_ID),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_action_len=args.max_action_len,
        attention_variant=args.attention_variant,
        trend_kernel_size=args.trend_kernel_size,
        use_cross_attn=not args.decoder_only_pretrain,
        use_distance_bias=args.use_distance_bias,
        distance_bias_hidden=args.distance_bias_hidden,
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
        val_metrics = evaluate(model, val_loader, device, action_tokenizer, args)
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
