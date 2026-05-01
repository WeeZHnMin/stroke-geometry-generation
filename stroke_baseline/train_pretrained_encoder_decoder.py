import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .dataset import StrokeJsonlDataset
from .pretrained_encoder_decoder import (
    DEFAULT_TEXT_ENCODER_DIR,
    StrokeDecoderConfig,
    TextConditionedStrokeModel,
)
from .sample import generate_strokes
from .visualize import save_strokes_png


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def compute_loss(batch: dict, out: dict, pen_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["target_mask"]

    pred_dxdy = out["pred_dxdy"][mask]
    target_dxdy = batch["target_dxdy"][mask]
    delta_loss = F.smooth_l1_loss(pred_dxdy, target_dxdy)

    pred_pen = out["pred_pen_logits"].reshape(-1, out["pred_pen_logits"].size(-1))
    target_pen = batch["target_pen"].reshape(-1)
    pen_loss = F.cross_entropy(pred_pen, target_pen, ignore_index=-100)

    loss = delta_loss + pen_weight * pen_loss

    with torch.no_grad():
        valid = target_pen != -100
        pred_label = pred_pen.argmax(dim=-1)
        pen_acc = (pred_label[valid] == target_pen[valid]).float().mean()

        class_accs = []
        for class_id in range(out["pred_pen_logits"].size(-1)):
            class_mask = valid & (target_pen == class_id)
            if class_mask.any():
                class_accs.append((pred_label[class_mask] == target_pen[class_mask]).float().mean())
        pen_macro_acc = torch.stack(class_accs).mean() if class_accs else pen_acc

        non_draw = valid & (target_pen != 0)
        if non_draw.any():
            pen_non_draw_acc = (pred_label[non_draw] == target_pen[non_draw]).float().mean()
        else:
            pen_non_draw_acc = torch.tensor(float("nan"), device=pred_pen.device)

    return loss, {
        "loss": float(loss.item()),
        "delta_loss": float(delta_loss.item()),
        "pen_loss": float(pen_loss.item()),
        "pen_acc": float(pen_acc.item()),
        "pen_macro_acc": float(pen_macro_acc.item()),
        "pen_non_draw_acc": float(pen_non_draw_acc.item()),
    }


def update_metric_sums(totals: dict[str, float], counts: dict[str, int], parts: dict[str, float]) -> None:
    for key, value in parts.items():
        if value != value:
            continue
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1


def average_metrics(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {key: totals[key] / max(counts.get(key, 0), 1) for key in totals}


def save_checkpoint(
    output_dir: Path,
    model: TextConditionedStrokeModel,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "context_proj": model.context_proj.state_dict(),
            "decoder_cfg": model.decoder.cfg.to_dict(),
            "max_text_len": model.max_text_len,
            "text_encoder_dir": str(args.text_encoder_dir),
            "epoch": epoch,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_one_epoch(
    model: TextConditionedStrokeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out = model(
            prompts=list(batch["prompt"]),
            decoder_dxdy=batch["decoder_dxdy"],
            decoder_pen=batch["decoder_pen"],
            target_mask=batch["target_mask"],
        )
        loss, parts = compute_loss(batch, out, pen_weight=args.pen_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip)
        optimizer.step()

        update_metric_sums(totals, counts, parts)

        if step % args.log_every == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={parts['loss']:.4f} delta={parts['delta_loss']:.4f} "
                f"pen={parts['pen_loss']:.4f} pen_acc={parts['pen_acc']:.3f} "
                f"pen_macro={parts['pen_macro_acc']:.3f} non_draw={parts['pen_non_draw_acc']:.3f}"
            )

    return average_metrics(totals, counts)


@torch.no_grad()
def evaluate(
    model: TextConditionedStrokeModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_dxdy=batch["decoder_dxdy"],
            decoder_pen=batch["decoder_pen"],
            target_mask=batch["target_mask"],
        )
        _, parts = compute_loss(batch, out, pen_weight=args.pen_weight)
        update_metric_sums(totals, counts, parts)

    return average_metrics(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen Chinese encoder + stroke decoder MVP.")
    parser.add_argument("--data", type=str, default="generated_data/bulk/stage1_foundation_shapes_v3_easy_20260403_125802.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_pretrained_encoder_decoder")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pen-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-stroke-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This dataset still tokenizes text internally for the old baseline, but this
    # trainer only uses prompt + stroke tensors. Keeping it avoids duplicating the
    # stroke shifting/padding logic.
    dataset = StrokeJsonlDataset(
        args.data,
        max_text_len=args.max_text_len,
        max_stroke_len=args.max_stroke_len,
        limit=args.limit,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    decoder_cfg = StrokeDecoderConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_stroke_len=args.max_stroke_len,
    )
    model = TextConditionedStrokeModel(
        decoder_cfg=decoder_cfg,
        text_encoder_dir=args.text_encoder_dir,
        max_text_len=args.max_text_len,
    ).to(device)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"device={device} train={train_size} val={val_size} "
        f"decoder_params={sum(p.numel() for p in model.decoder.parameters()):,}"
    )

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device, args)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_delta={val_metrics['delta_loss']:.4f} val_pen={val_metrics['pen_loss']:.4f} "
            f"val_pen_acc={val_metrics['pen_acc']:.3f} "
            f"val_pen_macro={val_metrics['pen_macro_acc']:.3f} "
            f"val_non_draw={val_metrics['pen_non_draw_acc']:.3f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir, model, args, epoch)

    print(f"saved best checkpoint to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
