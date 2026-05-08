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
from .action_tokenizer import ActionTokenizerConfig, CompactActionTokenMapper, StrokeActionTokenizer
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _mask_pad_logit(logits: torch.Tensor, tokenizer: StrokeActionTokenizer) -> torch.Tensor:
    masked = logits.clone()
    masked[..., tokenizer.pad_id] = torch.finfo(masked.dtype).min
    return masked


def _decode_token_components(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pen = torch.div(token_ids, 10201, rounding_mode="floor")
    offset = token_ids % 10201
    bin_x = torch.div(offset, 101, rounding_mode="floor")
    bin_y = offset % 101
    return bin_x, bin_y, pen


def _decode_target_tokens(token_ids: torch.Tensor, compact_mapper: CompactActionTokenMapper | None) -> torch.Tensor:
    if compact_mapper is None:
        return token_ids
    raw = token_ids.clone()
    valid = raw != -100
    if valid.any():
        flat = raw[valid].detach().cpu().tolist()
        decoded = [compact_mapper.decode(token_id) for token_id in flat]
        raw[valid] = torch.tensor(decoded, dtype=raw.dtype, device=raw.device)
    return raw


def compute_loss(
    batch: dict,
    out: dict,
    tokenizer: StrokeActionTokenizer,
    compact_mapper: CompactActionTokenMapper | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target_ids"]
    valid = target != -100
    target_raw = _decode_target_tokens(target, compact_mapper)
    safe_target_raw = target_raw.clamp_min(0)
    tgt_x_all, tgt_y_all, tgt_pen_all = _decode_token_components(safe_target_raw)
    tgt_x_all = tgt_x_all.masked_fill(~valid, -100)
    tgt_y_all = tgt_y_all.masked_fill(~valid, -100)
    tgt_pen_all = tgt_pen_all.masked_fill(~valid, -100)
    if compact_mapper is None:
        logits = _mask_pad_logit(out["logits"], tokenizer)
    else:
        logits = out["logits"].clone()
        logits[..., compact_mapper.pad_id] = torch.finfo(logits.dtype).min
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred[valid] == target[valid]).float().mean() if valid.any() else torch.tensor(0.0, device=target.device)
        pred_raw = _decode_target_tokens(pred, compact_mapper)
        pred_x, pred_y, pred_pen = _decode_token_components(pred_raw[valid])
        tgt_x = tgt_x_all[valid]
        tgt_y = tgt_y_all[valid]
        tgt_pen = tgt_pen_all[valid]

        x_mae = ((pred_x - tgt_x).abs().float() * 0.01).mean() if valid.any() else torch.tensor(0.0, device=target.device)
        y_mae = ((pred_y - tgt_y).abs().float() * 0.01).mean() if valid.any() else torch.tensor(0.0, device=target.device)
        xy_mae = ((x_mae + y_mae) * 0.5) if valid.any() else torch.tensor(0.0, device=target.device)
        x_acc = (pred_x == tgt_x).float().mean() if valid.any() else torch.tensor(0.0, device=target.device)
        y_acc = (pred_y == tgt_y).float().mean() if valid.any() else torch.tensor(0.0, device=target.device)
        pen_acc = (pred_pen == tgt_pen).float().mean() if valid.any() else torch.tensor(0.0, device=target.device)

        metrics = {
            "loss": float(loss.item()),
            "token_acc": float(acc.item()),
            "x_acc": float(x_acc.item()),
            "y_acc": float(y_acc.item()),
            "pen_acc": float(pen_acc.item()),
            "x_mae": float(x_mae.item()),
            "y_mae": float(y_mae.item()),
            "xy_mae": float(xy_mae.item()),
        }
    return loss, metrics


def update_sums(totals: dict[str, float], counts: dict[str, int], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1


def average(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {key: totals[key] / max(counts.get(key, 0), 1) for key in totals}


def save_checkpoint(
    output_dir: Path,
    model: TextConditionedActionModel,
    tokenizer: StrokeActionTokenizer,
    compact_mapper: CompactActionTokenMapper | None,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "decoder": model.decoder.state_dict(),
        "context_proj": model.context_proj.state_dict(),
        "decoder_cfg": model.decoder.cfg.to_dict(),
        "action_tokenizer_cfg": tokenizer.cfg.to_dict(),
        "compact_vocab_size": None if compact_mapper is None else compact_mapper.action_vocab_size,
        "compact_vocab_file": args.vocab_file,
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


def train_one_epoch(model, loader, optimizer, device, args, tokenizer, compact_mapper, epoch, step_log_path: Path):
    model.train()
    model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_coords=batch["decoder_coords"],
            decoder_pen_states=batch["decoder_pen_states"],
            target_mask=batch["target_mask"],
        )
        loss, metrics = compute_loss(batch, out, tokenizer, compact_mapper)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        update_sums(totals, counts, metrics)
        if step % args.log_every == 0:
            line = (
                f"epoch={epoch} step={step}/{len(loader)} loss={metrics['loss']:.4f} "
                f"acc={metrics['token_acc']:.3f} x_acc={metrics['x_acc']:.3f} "
                f"y_acc={metrics['y_acc']:.3f} pen_acc={metrics['pen_acc']:.3f} "
                f"xy_mae={metrics['xy_mae']:.4f}"
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
def evaluate(model, loader, device, tokenizer, compact_mapper):
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_coords=batch["decoder_coords"],
            decoder_pen_states=batch["decoder_pen_states"],
            target_mask=batch["target_mask"],
        )
        _, metrics = compute_loss(batch, out, tokenizer, compact_mapper)
        update_sums(totals, counts, metrics)
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CPCF-input action model with discrete Cartesian tokens.")
    parser.add_argument("--data", type=str, default="generated_data/bulk/stage1_foundation_shapes_v3_easy_20260403_125802.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_action_tokens_cpcf")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--vocab-file", type=str, default=None)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-variant", type=str, default="hetero", choices=["legacy_qkv", "legacy", "hetero"])
    parser.add_argument("--trend-kernel-size", type=int, default=5)
    parser.add_argument("--input-mode", type=str, default="cpcf", choices=["token", "cpcf"])
    parser.add_argument("--xy-hidden-dim", type=int, default=128)
    parser.add_argument("--pen-emb-dim", type=int, default=32)
    parser.add_argument("--input-kernel-size", type=int, default=3)
    parser.add_argument("--disable-2d-rope", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_tokenizer = StrokeActionTokenizer(ActionTokenizerConfig())
    compact_mapper = CompactActionTokenMapper.from_vocab_file(args.vocab_file) if args.vocab_file else None
    dataset_cls = TwoStageActionTokenJsonlDataset if args.two_stage else ActionTokenJsonlDataset
    dataset = dataset_cls(
        args.data,
        action_tokenizer=action_tokenizer,
        compact_mapper=compact_mapper,
        max_action_len=args.max_action_len,
        limit=args.limit,
    )
    if args.val_ratio <= 0.0:
        train_set = dataset
        val_set = None
        train_size = len(dataset)
        val_size = 0
    else:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Validation split leaves no training samples. Reduce --val-ratio or provide more data.")
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = None if val_set is None else DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = ActionDecoderConfig(
        action_vocab_size=compact_mapper.vocab_size if compact_mapper is not None else action_tokenizer.vocab_size,
        pad_token_id=compact_mapper.pad_id if compact_mapper is not None else action_tokenizer.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_action_len=args.max_action_len,
        attention_variant=args.attention_variant,
        trend_kernel_size=args.trend_kernel_size,
        input_mode=args.input_mode,
        xy_hidden_dim=args.xy_hidden_dim,
        pen_emb_dim=args.pen_emb_dim,
        input_kernel_size=args.input_kernel_size,
        use_2d_rope=not args.disable_2d_rope,
    )
    model = TextConditionedActionModel(cfg, text_encoder_dir=args.text_encoder_dir, max_text_len=args.max_text_len).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = output_dir / "step_metrics.jsonl"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"

    print(
        f"device={device} train={train_size} val={val_size} vocab={cfg.action_vocab_size} "
        f"decoder_params={sum(p.numel() for p in model.decoder.parameters()):,} "
        f"input_mode={args.input_mode} step_log={step_log_path} epoch_log={epoch_log_path}",
        flush=True,
    )
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, action_tokenizer, compact_mapper, epoch, step_log_path)
        val_metrics = None if val_loader is None else evaluate(model, val_loader, device, action_tokenizer, compact_mapper)
        if val_metrics is None:
            epoch_line = (
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['token_acc']:.3f} "
                f"train_x_acc={train_metrics['x_acc']:.3f} train_y_acc={train_metrics['y_acc']:.3f} "
                f"train_pen_acc={train_metrics['pen_acc']:.3f} train_xy_mae={train_metrics['xy_mae']:.4f}"
            )
        else:
            epoch_line = (
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['token_acc']:.3f} val_x_acc={val_metrics['x_acc']:.3f} "
                f"val_y_acc={val_metrics['y_acc']:.3f} val_pen_acc={val_metrics['pen_acc']:.3f} "
                f"val_xy_mae={val_metrics['xy_mae']:.4f}"
            )
        print(epoch_line, flush=True)
        current_score = train_metrics["loss"] if val_metrics is None else val_metrics["loss"]
        is_best = current_score < best_val
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
            best_val = current_score
            save_checkpoint(output_dir, model, action_tokenizer, compact_mapper, args, epoch)

    print(f"saved best checkpoint to {output_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
