"""Pretraining script for DxDyDecoder (decoder-only, no text conditioning)."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .dxdy_dataset import DxDyJsonlDataset, collate_fn
from .dxdy_decoder_model import DxDyDecoder, DxDyDecoderConfig
from .action_tokenizer import DxDyPairTokenizerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def neighbor_smoothed_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    window: int,
) -> torch.Tensor:
    """Gaussian-smoothed cross-entropy over neighboring bins (ordinal targets)."""
    if sigma <= 0:
        return F.cross_entropy(logits, target)
    vocab = logits.size(-1)
    offsets = torch.arange(-window, window + 1, device=logits.device)
    idx = (target[:, None] + offsets[None, :]).clamp(0, vocab - 1)
    w = torch.exp(-0.5 * (offsets.to(logits.dtype) / sigma) ** 2)
    w = w[None, :].expand_as(idx).clone()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.gather(-1, idx) * w).sum(-1).mean()


def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    sigma: float = 0.0,
    window: int = 3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    logits:     [B, T, V]
    target_ids: [B, T]   (-100 = ignore)
    """
    B, T, V = logits.shape
    flat_logits = logits.view(B * T, V)
    flat_target = target_ids.view(B * T)

    valid_mask = flat_target != -100
    if not valid_mask.any():
        dummy = flat_logits.sum() * 0.0
        return dummy, {"loss": 0.0, "acc": float("nan")}

    logits_v = flat_logits[valid_mask]
    target_v = flat_target[valid_mask]

    loss = neighbor_smoothed_ce(logits_v, target_v, sigma=sigma, window=window)

    with torch.no_grad():
        acc = (logits_v.argmax(-1) == target_v).float().mean().item()

    return loss, {"loss": float(loss.item()), "acc": acc}


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: DxDyDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    total_steps: int,
    step_log_path: Path,
) -> tuple[dict[str, float], int]:
    model.train()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for batch in loader:
        batch = move_batch(batch, device)

        lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"])
        loss, metrics = compute_loss(
            logits, batch["target_ids"],
            sigma=args.label_sigma, window=args.label_window,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        metrics["lr"] = lr
        global_step += 1

        for k, v in metrics.items():
            if v == v:
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

        if global_step % args.log_every == 0:
            print(
                f"epoch={epoch} step={global_step} loss={metrics['loss']:.4f} "
                f"acc={metrics['acc']:.3f} lr={lr:.2e}",
                flush=True,
            )
            append_jsonl(step_log_path, {"epoch": epoch, "step": global_step, **metrics})

    avg = {k: totals[k] / counts[k] for k in totals}
    return avg, global_step


@torch.no_grad()
def evaluate(
    model: DxDyDecoder,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"])
        _, metrics = compute_loss(
            logits, batch["target_ids"],
            sigma=args.label_sigma, window=args.label_window,
        )
        for k, v in metrics.items():
            if v == v:
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    model: DxDyDecoder,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain DxDyDecoder (decoder-only).")

    # data
    parser.add_argument("--data", type=str,
        default="generated_data/bulk/stage1_foundation_shapes_v3_mixed_20260403_125802.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/dxdy_decoder_pretrain")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to load (None = all)")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--min-steps", type=int, default=4)

    # tokenizer
    parser.add_argument("--dx-bins", type=int, default=100)
    parser.add_argument("--dy-bins", type=int, default=100)
    parser.add_argument("--log-scale", type=float, default=10.0,
        help="Log1p compression factor C (0 = uniform binning)")

    # model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv-kernel", type=int, default=12)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--label-sigma", type=float, default=1.0,
        help="Neighbor label smoothing sigma (0 = hard CE)")
    parser.add_argument("--label-window", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset (first build compact vocab from training data)
    base_cfg = DxDyPairTokenizerConfig(
        dx_bins=args.dx_bins,
        dy_bins=args.dy_bins,
        log_scale=args.log_scale,
    )
    print("Scanning dataset to build compact vocabulary...", flush=True)
    full_dataset = DxDyJsonlDataset(
        args.data,
        base_cfg=base_cfg,
        max_seq_len=args.max_seq_len,
        limit=args.limit,
        min_steps=args.min_steps,
    )
    tok = full_dataset.tokenizer

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # model
    cfg = DxDyDecoderConfig(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        bos_token_id=tok.bos_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len + 1,
        conv_kernel_size=args.conv_kernel,
    )
    model = DxDyDecoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log = output_dir / "step_metrics.jsonl"
    epoch_log = output_dir / "epoch_metrics.jsonl"

    # save tokenizer vocab alongside checkpoint
    tok.save(output_dir / "vocab.json")

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  train={train_size}  val={val_size}  "
        f"vocab={tok.vocab_size} (action={tok.action_vocab_size})  "
        f"params={n_params/1e6:.2f}M  total_steps={total_steps}",
        flush=True,
    )

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_avg, global_step = train_epoch(
            model, train_loader, optimizer, device, args,
            epoch, global_step, total_steps, step_log,
        )
        val_avg = evaluate(model, val_loader, device, args)

        print(
            f"[epoch {epoch}] train_loss={train_avg['loss']:.4f} "
            f"val_loss={val_avg['loss']:.4f} val_acc={val_avg['acc']:.3f}",
            flush=True,
        )
        is_best = val_avg["loss"] < best_val
        append_jsonl(epoch_log, {
            "epoch": epoch,
            "train": train_avg,
            "val": val_avg,
            "is_best": is_best,
        })
        if is_best:
            best_val = val_avg["loss"]
            save_checkpoint(output_dir, model, args, epoch, best_val)

    print(f"Done. Best val_loss={best_val:.4f}  checkpoint -> {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
