"""SFT training: BERT encoder + cross-attention stroke decoder.

Loads a pretrained DxDyDecoder checkpoint for decoder weights;
cross-attention sublayers are randomly initialised.

Usage (from project root):
    python -m stroke_baseline.train_sft \\
        --data   generated_data/bulk/core_shapes.jsonl \\
        --vocab  runs/dxdy_decoder_pretrain/vocab.json \\
        --decoder-ckpt runs/dxdy_decoder_pretrain/checkpoint.pt \\
        --bert   models/bert-base-chinese \\
        --output-dir runs/sft
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from .action_tokenizer import CompactDxDyTokenizer
from .sft_dataset import SFTJsonlDataset, collate_fn
from .sft_model import SFTConfig, SFTModel


# ---------------------------------------------------------------------------
# Helpers (reused from train_dxdy_decoder)
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


def get_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    B, T, V = logits.shape
    flat_logits = logits.view(B * T, V)
    flat_target = target_ids.view(B * T)
    valid = flat_target != -100
    if not valid.any():
        return flat_logits.sum() * 0.0, {"loss": 0.0, "acc": float("nan")}
    lv = flat_logits[valid]
    tv = flat_target[valid]
    loss = F.cross_entropy(lv, tv)
    with torch.no_grad():
        acc = (lv.argmax(-1) == tv).float().mean().item()
    return loss, {"loss": float(loss.item()), "acc": acc}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, args, epoch,
                global_step, total_steps, step_log):
    model.train()
    totals, counts = {}, {}
    for batch in loader:
        batch = move_batch(batch, device)
        lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            stroke_input_ids=batch["stroke_input_ids"],
            enc_input_ids=batch["enc_input_ids"],
            enc_attention_mask=batch["enc_attention_mask"],
            coords=batch.get("coords"),
        )
        loss, metrics = compute_loss(logits, batch["stroke_target_ids"])
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
            print(f"epoch={epoch} step={global_step} loss={metrics['loss']:.4f} "
                  f"acc={metrics['acc']:.3f} lr={lr:.2e}", flush=True)
            append_jsonl(step_log, {"epoch": epoch, "step": global_step, **metrics})

    avg = {k: totals[k] / counts[k] for k in totals}
    return avg, global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals, counts = {}, {}
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            stroke_input_ids=batch["stroke_input_ids"],
            enc_input_ids=batch["enc_input_ids"],
            enc_attention_mask=batch["enc_attention_mask"],
            coords=batch.get("coords"),
        )
        _, metrics = compute_loss(logits, batch["stroke_target_ids"])
        for k, v in metrics.items():
            if v == v:
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, model, args, epoch, val_loss):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": model.cfg.to_dict(),
         "epoch": epoch, "val_loss": val_loss},
        output_dir / "checkpoint.pt",
    )
    (output_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT: train BERT encoder + stroke decoder.")

    # data
    parser.add_argument("--data", type=str,
        default="generated_data/bulk/core_shapes.jsonl")
    parser.add_argument("--vocab", type=str, required=True,
        help="Path to vocab JSON (from pretrain run or export_dxdy_vocab.py).")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-stroke-len", type=int, default=256)
    parser.add_argument("--max-text-len", type=int, default=64)

    # encoder
    parser.add_argument("--bert", type=str, default="models/bert-base-chinese")
    parser.add_argument("--adapter-dim", type=int, default=64,
        help="Adapter bottleneck size per BERT layer (0 = fully frozen, no adapters).")

    # decoder init
    parser.add_argument("--decoder-ckpt", type=str, default=None,
        help="Path to pretrained DxDyDecoder checkpoint.pt. "
             "If omitted, decoder is randomly initialised.")

    # model dims (must match --decoder-ckpt if provided)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv-kernel", type=int, default=12)

    # training
    parser.add_argument("--output-dir", type=str, default="runs/sft")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizers
    stroke_tok = CompactDxDyTokenizer.load(args.vocab)
    bert_tok = BertTokenizer.from_pretrained(args.bert)
    print(f"stroke vocab_size={stroke_tok.vocab_size}  bert_vocab={bert_tok.vocab_size}",
          flush=True)

    # dataset
    full_dataset = SFTJsonlDataset(
        args.data,
        tokenizer=stroke_tok,
        bert_tokenizer=bert_tok,
        max_stroke_len=args.max_stroke_len,
        max_text_len=args.max_text_len,
        limit=args.limit,
    )
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
    cfg = SFTConfig(
        vocab_size=stroke_tok.vocab_size,
        pad_token_id=stroke_tok.pad_id,
        bos_token_id=stroke_tok.bos_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_seq_len=args.max_stroke_len + 1,
        conv_kernel_size=args.conv_kernel,
        bert_path=args.bert,
        encoder_dim=768,
        adapter_dim=args.adapter_dim,
    )
    model = SFTModel(cfg).to(device)

    if args.decoder_ckpt:
        model.load_pretrained_decoder(args.decoder_ckpt)
    else:
        print("No --decoder-ckpt given: decoder weights are random.", flush=True)

    # only optimise parameters that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log = output_dir / "step_metrics.jsonl"
    epoch_log = output_dir / "epoch_metrics.jsonl"

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(
        f"device={device}  train={train_size}  val={val_size}\n"
        f"total_params={n_params/1e6:.2f}M  trainable={n_train/1e6:.2f}M  "
        f"total_steps={total_steps}",
        flush=True,
    )

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_avg, global_step = train_epoch(
            model, train_loader, optimizer, device, args,
            epoch, global_step, total_steps, step_log,
        )
        val_avg = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={train_avg['loss']:.4f} "
              f"val_loss={val_avg['loss']:.4f} val_acc={val_avg['acc']:.3f}", flush=True)
        is_best = val_avg["loss"] < best_val
        append_jsonl(epoch_log, {"epoch": epoch, "train": train_avg,
                                 "val": val_avg, "is_best": is_best})
        if is_best:
            best_val = val_avg["loss"]
            save_checkpoint(output_dir, model, args, epoch, best_val)

    print(f"Done. Best val_loss={best_val:.4f}  -> {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
