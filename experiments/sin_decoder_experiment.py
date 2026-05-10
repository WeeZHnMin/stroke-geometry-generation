from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sine_tokens(seq_len: int, bins: int, x_end: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, x_end, seq_len, device=device)
    y = torch.sin(x)
    tokens = torch.round(((y + 1.0) * 0.5) * (bins - 1)).long().clamp(0, bins - 1)
    return x, y, tokens


def tokens_to_y(tokens: torch.Tensor, bins: int) -> torch.Tensor:
    return (tokens.float() / max(bins - 1, 1)) * 2.0 - 1.0


class DecoderOnlySineModel(nn.Module):
    def __init__(
        self,
        *,
        bins: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.bos_id = bins
        self.vocab_size = bins + 1
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, bins)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(x, mask=causal_mask)
        return self.head(self.norm(hidden))


class XToYSineModel(nn.Module):
    def __init__(
        self,
        *,
        bins: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.x_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(seq_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, bins)

    def forward(self, x_values: torch.Tensor) -> torch.Tensor:
        seq_len = x_values.size(1)
        pos = torch.arange(seq_len, device=x_values.device)
        hidden = self.x_proj(x_values[..., None]) + self.pos_emb(pos)[None, :, :]
        hidden = self.blocks(hidden)
        return self.head(self.norm(hidden))


def make_batch(tokens: torch.Tensor, batch_size: int, bos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    target = tokens.unsqueeze(0).expand(batch_size, -1)
    decoder_input = torch.empty_like(target)
    decoder_input[:, 0] = bos_id
    decoder_input[:, 1:] = target[:, :-1]
    return decoder_input, target


@torch.no_grad()
def rollout(
    model: DecoderOnlySineModel,
    seq_len: int,
    device: torch.device,
    prefix_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    generated: list[int] = [] if prefix_tokens is None else [int(t) for t in prefix_tokens.detach().cpu().tolist()]
    input_ids = torch.tensor([[model.bos_id, *generated]], dtype=torch.long, device=device)
    for _ in range(max(0, seq_len - len(generated))):
        logits = model(input_ids)
        next_token = int(logits[0, -1].argmax(dim=-1).item())
        generated.append(next_token)
        input_ids = torch.tensor([[model.bos_id, *generated]], dtype=torch.long, device=device)
    return torch.tensor(generated, dtype=torch.long, device=device)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x, y_true, tokens = build_sine_tokens(args.seq_len, args.bins, args.x_end, device)
    if args.x_mode == "normalized":
        x_feat = (x / max(float(x[-1].item()), 1e-6)) * 2.0 - 1.0
    elif args.x_mode == "raw":
        x_feat = x
    else:
        raise ValueError(f"unsupported x_mode: {args.x_mode}")

    if args.mode == "autoregressive":
        model = DecoderOnlySineModel(
            bins=args.bins,
            seq_len=args.seq_len + 1,
            d_model=args.d_model,
            n_heads=args.n_heads,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    elif args.mode == "x-to-y":
        model = XToYSineModel(
            bins=args.bins,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"unsupported mode: {args.mode}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        target = tokens.unsqueeze(0).expand(args.batch_size, -1)
        if args.mode == "autoregressive":
            decoder_input, target = make_batch(tokens, args.batch_size, model.bos_id)
            logits = model(decoder_input)
        else:
            x_batch = x_feat.unsqueeze(0).expand(args.batch_size, -1)
            logits = model(x_batch)
        loss = F.cross_entropy(logits.reshape(-1, args.bins), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)[0]
                token_acc = (pred == tokens).float().mean().item()
                bin_mae = (pred - tokens).abs().float().mean().item()
                y_mae = (tokens_to_y(pred, args.bins) - y_true).abs().mean().item()
            row = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "token_acc": token_acc,
                "bin_mae": bin_mae,
                "teacher_forced_y_mae": y_mae,
            }
            history.append(row)
            print(
                f"epoch={epoch:04d} loss={row['loss']:.4f} "
                f"acc={token_acc:.3f} bin_mae={bin_mae:.3f} y_mae={y_mae:.4f}",
                flush=True,
            )

    prefix_count = 0
    if args.mode == "autoregressive":
        prefix_count = max(0, min(args.rollout_prefix_steps, args.seq_len))
        rollout_prefix = tokens[:prefix_count] if prefix_count else None
        pred_tokens = rollout(model, args.seq_len, device, prefix_tokens=rollout_prefix)
    else:
        model.eval()
        with torch.no_grad():
            pred_tokens = model(x_feat[None, :]).argmax(dim=-1)[0]
    pred_y = tokens_to_y(pred_tokens, args.bins)
    rollout_correct = pred_tokens == tokens
    rollout_abs_bin = (pred_tokens - tokens).abs().float()
    rollout_abs_y = (pred_y - y_true).abs()
    rollout_token_acc = rollout_correct.float().mean().item()
    rollout_bin_mae = rollout_abs_bin.mean().item()
    rollout_y_mae = rollout_abs_y.mean().item()
    tail_slice = slice(prefix_count, args.seq_len)
    if prefix_count < args.seq_len:
        free_token_acc = rollout_correct[tail_slice].float().mean().item()
        free_bin_mae = rollout_abs_bin[tail_slice].mean().item()
        free_y_mae = rollout_abs_y[tail_slice].mean().item()
    else:
        free_token_acc = float("nan")
        free_bin_mae = float("nan")
        free_y_mae = float("nan")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "device": str(device),
        "args": vars(args),
        "history": history,
        "rollout": {
            "prefix_steps": prefix_count,
            "token_acc": rollout_token_acc,
            "bin_mae": rollout_bin_mae,
            "y_mae": rollout_y_mae,
            "free_token_acc": free_token_acc,
            "free_bin_mae": free_bin_mae,
            "free_y_mae": free_y_mae,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "checkpoint.pt")

    plt.figure(figsize=(8, 3), dpi=140)
    plt.plot(x.detach().cpu(), y_true.detach().cpu(), label="sin(x)", linewidth=2)
    plt.plot(x.detach().cpu(), pred_y.detach().cpu(), label="decoder rollout", linestyle="--")
    if prefix_count:
        plt.axvline(float(x[prefix_count - 1].detach().cpu()), color="orange", linewidth=1, alpha=0.8, label="prefix end")
    plt.ylim(-1.15, 1.15)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rollout.png")
    plt.close()

    print(
        f"rollout acc={rollout_token_acc:.3f} bin_mae={rollout_bin_mae:.3f} "
        f"y_mae={rollout_y_mae:.4f} | free_tail acc={free_token_acc:.3f} "
        f"bin_mae={free_bin_mae:.3f} y_mae={free_y_mae:.4f}",
        flush=True,
    )
    print(f"saved {out_dir / 'rollout.png'}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer to autoregressively learn discretized sin(x).")
    parser.add_argument("--mode", type=str, default="autoregressive", choices=["autoregressive", "x-to-y"])
    parser.add_argument("--output-dir", type=str, default="runs/sin_decoder_experiment")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--x-end", type=float, default=2.0 * math.pi, help="End value of the x range.")
    parser.add_argument("--x-mode", type=str, default="raw", choices=["raw", "normalized"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--rollout-prefix-steps", type=int, default=0, help="Teacher-force this many initial y tokens before free rollout.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
