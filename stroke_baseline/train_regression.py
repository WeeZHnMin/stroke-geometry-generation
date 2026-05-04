import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .regression_dataset import StrokeRegressionJsonlDataset
from .regression_model import CoordRegressionDecoderConfig, TextConditionedCoordRegressionModel
from .train_action_tokens import average, update_sums


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid])


def last_valid_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1).clamp_min(1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, values.size(-1))
    return values.gather(dim=1, index=idx).squeeze(1)


def compute_loss(batch: dict, out: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    pred = out["pred_dxdy"]
    target = batch["target_dxdy"]
    mask = batch["target_mask"].bool()
    mask_f = mask.unsqueeze(-1).to(pred.dtype)

    delta_loss = masked_smooth_l1(pred, target, mask)
    pred_xy = torch.cumsum(pred * mask_f, dim=1)
    true_xy = torch.cumsum(target * mask_f, dim=1)
    xy_loss = masked_smooth_l1(pred_xy, true_xy, mask)
    end_loss = F.smooth_l1_loss(last_valid_values(pred_xy, mask), last_valid_values(true_xy, mask))
    loss = delta_loss + args.xy_loss_weight * xy_loss + args.end_loss_weight * end_loss

    with torch.no_grad():
        valid = mask.unsqueeze(-1).expand_as(target)
        delta_mae = (pred - target).abs()[valid].mean()
        xy_mae = (pred_xy - true_xy).abs()[valid].mean()
        end_mae = (last_valid_values(pred_xy, mask) - last_valid_values(true_xy, mask)).abs().mean()
    return loss, {
        "loss": float(loss.item()),
        "delta_loss": float(delta_loss.item()),
        "xy_loss": float(xy_loss.item()),
        "end_loss": float(end_loss.item()),
        "delta_mae": float(delta_mae.item()),
        "xy_mae": float(xy_mae.item()),
        "end_mae": float(end_mae.item()),
    }


def save_checkpoint(output_dir: Path, model: TextConditionedCoordRegressionModel, args: argparse.Namespace, epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "context_proj": model.context_proj.state_dict(),
            "decoder_cfg": model.decoder.cfg.to_dict(),
            "max_text_len": model.max_text_len,
            "text_encoder_dir": str(args.text_encoder_dir),
            "canvas_size": args.canvas_size,
            "draw_only": args.draw_only,
            "epoch": epoch,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def train_one_epoch(model, loader, optimizer, device, args, epoch, step_log_path: Path):
    model.train()
    model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        decoder_coords = batch["decoder_coords"]
        if args.noise_std > 0:
            noise = torch.randn_like(decoder_coords) * args.noise_std
            noise[..., 2] = 0.0
            decoder_coords = decoder_coords + noise * batch["target_mask"].unsqueeze(-1).to(decoder_coords.dtype)

        optimizer.zero_grad(set_to_none=True)
        out = model(prompts=list(batch["prompt"]), decoder_coords=decoder_coords, target_mask=batch["target_mask"])
        loss, metrics = compute_loss(batch, out, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        update_sums(totals, counts, metrics)
        if step % args.log_every == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={metrics['loss']:.5f} "
                f"d_mae={metrics['delta_mae']:.4f} xy_mae={metrics['xy_mae']:.4f} end_mae={metrics['end_mae']:.4f}",
                flush=True,
            )
            append_jsonl(step_log_path, {"epoch": epoch, "step": step, "total_steps": len(loader), "split": "train", **metrics})
    return average(totals, counts)


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(prompts=list(batch["prompt"]), decoder_coords=batch["decoder_coords"], target_mask=batch["target_mask"])
        _, metrics = compute_loss(batch, out, args)
        update_sums(totals, counts, metrics)
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-conditioned coordinate-to-dxdy regression decoder.")
    parser.add_argument("--data", type=str, default="generated_data/polar_balanced/polar_balanced_scale8_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_regression_scale8")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--include-move", action="store_true", help="Train on move steps too. Default trains draw-like steps only.")
    parser.add_argument("--xy-loss-weight", type=float, default=0.5)
    parser.add_argument("--end-loss-weight", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.draw_only = not args.include_move

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StrokeRegressionJsonlDataset(
        args.data,
        max_len=args.max_len,
        canvas_size=args.canvas_size,
        max_delta=args.max_delta,
        draw_only=args.draw_only,
        limit=args.limit,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = CoordRegressionDecoderConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        max_delta=args.max_delta,
    )
    model = TextConditionedCoordRegressionModel(cfg, text_encoder_dir=args.text_encoder_dir, max_text_len=args.max_text_len).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = output_dir / "step_metrics.jsonl"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    print(
        f"device={device} train={train_size} val={val_size} draw_only={args.draw_only} "
        f"decoder_params={sum(p.numel() for p in model.decoder.parameters()):,}",
        flush=True,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch, step_log_path)
        val_metrics = evaluate(model, val_loader, device, args)
        is_best = val_metrics["loss"] < best_val
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_d_mae={val_metrics['delta_mae']:.4f} val_xy_mae={val_metrics['xy_mae']:.4f} "
            f"val_end_mae={val_metrics['end_mae']:.4f}",
            flush=True,
        )
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
            save_checkpoint(output_dir, model, args, epoch)

    print(f"saved best checkpoint to {output_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
