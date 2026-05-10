from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class RotationPointJsonlDataset(Dataset):
    def __init__(self, path: str, limit: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(path)
        self.offsets: list[int] = []
        with self.path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self.offsets.append(offset)
                if limit is not None and len(self.offsets) >= limit:
                    break
        if not self.offsets:
            raise ValueError(f"No samples found in {path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            raw = json.loads(f.readline())
        inp = raw["input"]
        tgt = raw["target"]
        log_scale = float(inp.get("log_scale", math.log(float(inp.get("scale", 1.0)))))
        return {
            "xy": torch.tensor([inp["x"], inp["y"]], dtype=torch.float32),
            "theta": torch.tensor([inp["theta"]], dtype=torch.float32),
            "log_scale": torch.tensor([log_scale], dtype=torch.float32),
            "target": torch.tensor([tgt["x"], tgt["y"]], dtype=torch.float32),
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class RotationBlock(nn.Module):
    def __init__(self, activation: str = "scaled_tanh", activation_scale: float = 4.0, use_scale: bool = False) -> None:
        super().__init__()
        self.base_angle = nn.Parameter(torch.zeros(()))
        self.theta_scale = nn.Parameter(torch.ones(()) / 64.0)
        self.use_scale = bool(use_scale)
        if self.use_scale:
            self.log_scale_weight = nn.Parameter(torch.ones(()) / 64.0)
            self.log_scale_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_scale_weight", None)
            self.register_parameter("log_scale_bias", None)
        self.activation_scale = float(activation_scale)
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "scaled_tanh":
            self.activation = lambda x: self.activation_scale * torch.tanh(x / self.activation_scale)
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def forward(self, xy: torch.Tensor, theta: torch.Tensor, log_scale: torch.Tensor | None = None) -> torch.Tensor:
        angle = self.base_angle + self.theta_scale * theta.squeeze(-1)
        c = torch.cos(angle)
        s = torch.sin(angle)
        x = xy[:, 0]
        y = xy[:, 1]
        xr = c * x - s * y
        yr = s * x + c * y
        out = torch.stack([xr, yr], dim=-1)
        if self.use_scale:
            if log_scale is None:
                raise ValueError("log_scale is required when use_scale=True")
            layer_log_scale = self.log_scale_weight * log_scale.squeeze(-1) + self.log_scale_bias
            out = out * torch.exp(layer_log_scale)[:, None]
        return self.activation(out)


class RotationStackModel(nn.Module):
    def __init__(
        self,
        layers: int = 64,
        activation: str = "scaled_tanh",
        activation_scale: float = 4.0,
        use_scale: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RotationBlock(activation=activation, activation_scale=activation_scale, use_scale=use_scale)
                for _ in range(layers)
            ]
        )

    def forward(self, xy: torch.Tensor, theta: torch.Tensor, log_scale: torch.Tensor | None = None) -> torch.Tensor:
        h = xy
        for layer in self.layers:
            h = layer(h, theta, log_scale=log_scale)
        return h


@torch.no_grad()
def evaluate(model: RotationStackModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(batch["xy"], batch["theta"], log_scale=batch.get("log_scale"))
        mse = F.mse_loss(pred, batch["target"], reduction="sum")
        mae = (pred - batch["target"]).abs().sum()
        total_mse += float(mse.item())
        total_mae += float(mae.item())
        total += int(batch["target"].numel())
    return {
        "mse": total_mse / max(total, 1),
        "mae": total_mae / max(total, 1),
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = RotationPointJsonlDataset(args.data, limit=args.limit)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = RotationStackModel(
        layers=args.layers,
        activation=args.activation,
        activation_scale=args.activation_scale,
        use_scale=args.use_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            pred = model(batch["xy"], batch["theta"], log_scale=batch.get("log_scale"))
            loss = F.mse_loss(pred, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["target"].size(0))
            total += int(batch["target"].size(0))
        train_metrics = {
            "loss": total_loss / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train_mse={train_metrics['loss']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f}",
                flush=True,
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "device": str(device),
        "args": vars(args),
        "history": history,
        "final_val": val_metrics,
        "layers": args.layers,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "checkpoint.pt")

    # Visualize a small validation slice.
    batch = next(iter(val_loader))
    batch = batch_to_device(batch, device)
    with torch.no_grad():
        pred = model(batch["xy"], batch["theta"], log_scale=batch.get("log_scale")).cpu()
    xy = batch["xy"].cpu()
    target = batch["target"].cpu()
    theta = batch["theta"].cpu().squeeze(-1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=140)
    ax.scatter(target[:, 0], target[:, 1], s=8, alpha=0.5, label="target")
    ax.scatter(pred[:, 0], pred[:, 1], s=8, alpha=0.5, label="pred")
    for i in range(min(20, xy.size(0))):
        ax.plot([xy[i, 0], target[i, 0]], [xy[i, 1], target[i, 1]], color="gray", alpha=0.15)
    ax.set_title(f"rotation stack rollout | layers={args.layers} | theta sample")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rollout.png")
    plt.close(fig)

    print(f"saved {out_dir / 'rollout.png'}", flush=True)
    for i, layer in enumerate(model.layers[:5]):
        print(
            f"layer{i}: base_angle={float(layer.base_angle.item()):.4f} "
            f"theta_scale={float(layer.theta_scale.item()):.4f}",
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stack of learnable 2D rotation matrices with activation.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/rotation_stack")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--activation", type=str, default="scaled_tanh", choices=["scaled_tanh", "tanh", "gelu", "relu"])
    parser.add_argument("--activation-scale", type=float, default=4.0)
    parser.add_argument("--use-scale", action="store_true", help="Learn per-layer conditional scalar scales from input log_scale.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
