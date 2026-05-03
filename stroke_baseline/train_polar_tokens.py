import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .polar_dataset import PolarActionJsonlDataset
from .polar_model import PolarDecoderConfig, TextConditionedPolarModel
from .polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR
from .train_action_tokens import average, update_sums


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_loss(batch: dict, out: dict) -> tuple[torch.Tensor, dict[str, float]]:
    logits = out["logits"]
    target = batch["target_ids"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        valid = target != -100
        pred = logits.argmax(dim=-1)
        acc = (pred[valid] == target[valid]).float().mean()
    return loss, {"loss": float(loss.item()), "token_acc": float(acc.item())}


def save_checkpoint(
    output_dir: Path,
    model: TextConditionedPolarModel,
    tokenizer: PolarActionTokenizer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "context_proj": model.context_proj.state_dict(),
            "decoder_cfg": model.decoder.cfg.to_dict(),
            "polar_tokenizer_cfg": tokenizer.cfg.to_dict(),
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


def train_one_epoch(model, loader, optimizer, device, args, epoch, step_log_path: Path):
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
            target_mask=batch["target_mask"],
        )
        loss, metrics = compute_loss(batch, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        update_sums(totals, counts, metrics)
        if step % args.log_every == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={metrics['loss']:.4f} acc={metrics['token_acc']:.3f}", flush=True)
            append_jsonl(step_log_path, {"epoch": epoch, "step": step, "total_steps": len(loader), "split": "train", **metrics})
    return average(totals, counts)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            prompts=list(batch["prompt"]),
            decoder_input_ids=batch["decoder_input_ids"],
            target_mask=batch["target_mask"],
        )
        _, metrics = compute_loss(batch, out)
        update_sums(totals, counts, metrics)
    return average(totals, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-to-polar-action-token baseline.")
    parser.add_argument("--data", type=str, default="generated_data/polar/polar_scale8_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/stroke_polar_scale8")
    parser.add_argument("--text-encoder-dir", type=str, default=str(DEFAULT_TEXT_ENCODER_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distances", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distances = tuple(float(v.strip()) for v in args.distances.split(",") if v.strip())
    tokenizer = PolarActionTokenizer(PolarActionTokenizerConfig(distance_buckets=distances, theta_bins=args.theta_bins))
    dataset = PolarActionJsonlDataset(args.data, tokenizer=tokenizer, max_action_len=args.max_action_len, limit=args.limit)

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    cfg = PolarDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_action_len=args.max_action_len,
    )
    model = TextConditionedPolarModel(cfg, text_encoder_dir=args.text_encoder_dir, max_text_len=args.max_text_len).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = output_dir / "step_metrics.jsonl"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    print(
        f"device={device} train={train_size} val={val_size} vocab={tokenizer.vocab_size} "
        f"decoder_params={sum(p.numel() for p in model.decoder.parameters()):,}",
        flush=True,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch, step_log_path)
        val_metrics = evaluate(model, val_loader, device)
        is_best = val_metrics["loss"] < best_val
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['token_acc']:.3f}",
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
            save_checkpoint(output_dir, model, tokenizer, args, epoch)

    print(f"saved best checkpoint to {output_dir / 'checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
