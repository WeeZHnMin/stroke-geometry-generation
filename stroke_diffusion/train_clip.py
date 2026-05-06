from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from stroke_baseline.train_action_tokens import average, update_sums

from .clip_model import StrokeCLIPConfig, TextStrokeCLIPModel, clip_contrastive_loss
from .dataset import QuantizedStrokeDiffusionDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def save_checkpoint(output_dir: Path, model: TextStrokeCLIPModel, args: argparse.Namespace, epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stroke_encoder": model.stroke_encoder.state_dict(),
            "text_proj": model.text_proj.state_dict(),
            "cfg": model.cfg.to_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "text_encoder_dir": str(args.text_encoder_dir),
            "epoch": epoch,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")


def run_epoch(model, loader, optimizer, device, epoch, step_log_path: Path | None, log_every: int, train: bool):
    model.train(train)
    model.text_encoder.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(prompts=list(batch["prompt"]), steps=batch["steps"], seq_mask=batch["seq_mask"])
        loss, metrics = clip_contrastive_loss(out["logits_per_text"], out["logits_per_stroke"])
        if train:
            loss.backward()
            optimizer.step()
        update_sums(totals, counts, metrics)
        if train and step_log_path is not None and step % log_every == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={metrics['loss']:.5f} "
                f"text_acc={metrics['text_acc']:.3f} stroke_acc={metrics['stroke_acc']:.3f}",
                flush=True,
            )
            append_jsonl(step_log_path, {"epoch": epoch, "step": step, "split": "train", **metrics})
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-stroke CLIP alignment model.")
    parser.add_argument("--data", type=str, default="generated_data/quantized_grid/test_quantized_continuous.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_clip")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QuantizedStrokeDiffusionDataset(args.data, max_seq_len=args.max_seq_len, limit=args.limit)
    if len(dataset) < 2:
        train_set = dataset
        val_set = dataset
        train_size = len(dataset)
        val_size = len(dataset)
    else:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = StrokeCLIPConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        max_text_len=args.max_text_len,
        proj_dim=args.proj_dim,
    )
    model = TextStrokeCLIPModel(cfg, text_encoder_dir=args.text_encoder_dir).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = output_dir / "step_metrics.jsonl"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"

    print(
        f"device={device} train={train_size} val={val_size} "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, step_log_path, args.log_every, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, epoch, None, args.log_every, train=False)
        is_best = val_metrics["loss"] < best_val
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_text_acc={val_metrics['text_acc']:.3f} val_stroke_acc={val_metrics['stroke_acc']:.3f}",
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
