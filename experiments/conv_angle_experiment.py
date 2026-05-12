"""
实验：用 1D 卷积学习坐标序列的局部方向趋势。
每个位置嵌入到 128 维，卷积核 size=12 在序列上滑动，
预测下一步的方向角 atan2(dy, dx)。
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── 数据生成 ──────────────────────────────────────────────────────────────────

def make_sine(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.uniform(0.5, 2.0)
    w = rng.uniform(0.5, 3.0)
    phi = rng.uniform(0, 2 * math.pi)
    x = np.linspace(0, 4 * math.pi, n)
    y = A * np.sin(w * x + phi)
    return np.stack([x, y], axis=-1)


def make_circle(n: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(1.0, 3.0)
    start = rng.uniform(0, 2 * math.pi)
    turns = rng.uniform(0.5, 2.0)
    t = np.linspace(start, start + turns * 2 * math.pi, n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=-1)


def make_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    turns = rng.uniform(1.0, 4.0)
    t = np.linspace(0, turns * 2 * math.pi, n)
    r = t / (turns * 2 * math.pi) * 3.0
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=-1)


def make_bezier(n: int, rng: np.random.Generator) -> np.ndarray:
    # 随机三次贝塞尔曲线
    p0 = rng.uniform(-2, 2, 2)
    p1 = rng.uniform(-2, 2, 2)
    p2 = rng.uniform(-2, 2, 2)
    p3 = rng.uniform(-2, 2, 2)
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1-t)**3 * p0 + 3*(1-t)**2*t * p1
           + 3*(1-t)*t**2 * p2 + t**3 * p3)
    return pts


GENERATORS = [make_sine, make_circle, make_spiral, make_bezier]


def generate_dataset(n_seqs: int, seq_len: int, kernel_size: int, seed: int = 42):
    """
    返回 (inputs, targets)：
      inputs:  (N_windows, kernel_size, 2)  每个窗口的原始坐标
      targets: (N_windows,)                  对应的方向角 atan2(dy,dx)
    """
    rng = np.random.default_rng(seed)
    all_inputs, all_targets = [], []

    for _ in range(n_seqs):
        gen = rng.choice(GENERATORS)
        seq = gen(seq_len, rng)                 # (seq_len, 2)

        # 对序列做简单归一化到 [-1, 1]
        seq = seq - seq.mean(axis=0)
        scale = np.abs(seq).max() + 1e-6
        seq = seq / scale

        for t in range(kernel_size - 1, seq_len - 1):
            window = seq[t - kernel_size + 1: t + 1]   # (kernel_size, 2)
            dx = seq[t + 1, 0] - seq[t, 0]
            dy = seq[t + 1, 1] - seq[t, 1]
            angle = math.atan2(dy, dx)
            all_inputs.append(window)
            all_targets.append(angle)

    inputs = torch.tensor(np.array(all_inputs), dtype=torch.float32)
    targets = torch.tensor(np.array(all_targets), dtype=torch.float32)
    return inputs, targets


# ── 模型 ──────────────────────────────────────────────────────────────────────

class ConvAnglePredictor(nn.Module):
    def __init__(self, embed_dim: int = 128, kernel_size: int = 12) -> None:
        super().__init__()
        self.embed = nn.Linear(2, embed_dim)

        # Conv1d 输入 (batch, embed_dim, kernel_size)，kernel 覆盖整个窗口 → 输出长度 1
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, kernel_size, 2)
        h = self.embed(x)                       # (batch, kernel_size, embed_dim)
        h = h.permute(0, 2, 1)                  # (batch, embed_dim, kernel_size)
        h = F.gelu(self.conv(h))                # (batch, embed_dim, 1)
        h = h.squeeze(-1)                        # (batch, embed_dim)
        return self.mlp(h).squeeze(-1)           # (batch,)


# ── 角度损失（处理 ±π 环绕） ──────────────────────────────────────────────────

def angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    # 把差值绕到 (-π, π]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return (diff ** 2).mean()


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("生成数据...")
    inputs, targets = generate_dataset(args.n_seqs, args.seq_len, args.kernel_size, args.seed)
    print(f"样本数: {len(inputs)}")

    dataset = TensorDataset(inputs.to(device), targets.to(device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConvAnglePredictor(embed_dim=args.embed_dim, kernel_size=args.kernel_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, t_batch in loader:
            pred = model(x_batch)
            loss = angle_loss(pred, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        epoch_loss /= len(inputs)
        losses.append(epoch_loss)
        scheduler.step()

        if epoch % max(1, args.epochs // 10) == 0:
            mae = angle_mae(model, inputs.to(device), targets.to(device))
            print(f"epoch {epoch:4d}  loss={epoch_loss:.4e}  MAE={math.degrees(mae):.2f}°")

    torch.save({"model_state": model.state_dict(), "args": vars(args)},
               "conv_angle_model.pt")
    print("模型已保存到 conv_angle_model.pt")

    plot_loss(losses, args)
    visualize(model, inputs.to(device), targets.to(device), args)


def angle_mae(model: ConvAnglePredictor, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
    diff = (pred - targets + math.pi) % (2 * math.pi) - math.pi
    return diff.abs().mean().item()


# ── 可视化 ────────────────────────────────────────────────────────────────────

def plot_loss(losses: list[float], args: argparse.Namespace) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("angle loss (log)")
    plt.title(f"Conv angle — seqs={args.n_seqs}, kernel={args.kernel_size}, embed={args.embed_dim}")
    plt.tight_layout()
    plt.savefig("conv_angle_loss.png", dpi=120)
    print("loss 曲线已保存到 conv_angle_loss.png")


def visualize(model: ConvAnglePredictor, inputs: torch.Tensor,
              targets: torch.Tensor, args: argparse.Namespace, n_show: int = 300) -> None:
    model.eval()
    with torch.no_grad():
        pred = model(inputs[:n_show])
    pred_np = pred.cpu().numpy()
    true_np = targets[:n_show].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：预测角度 vs 真实角度散点
    ax = axes[0]
    ax.scatter(true_np, pred_np, s=8, alpha=0.4, color="steelblue")
    lim = [-math.pi, math.pi]
    ax.plot(lim, lim, "r--", linewidth=1, label="理想对角线")
    ax.set_xlabel("真实角度 (rad)")
    ax.set_ylabel("预测角度 (rad)")
    ax.set_title(f"预测 vs 真实角度（前 {n_show} 个窗口）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：误差分布直方图（角度差，单位°）
    ax2 = axes[1]
    diff_deg = np.degrees(((pred_np - true_np + math.pi) % (2 * math.pi) - math.pi))
    ax2.hist(diff_deg, bins=50, color="tomato", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("预测误差 (°)")
    ax2.set_ylabel("频数")
    ax2.set_title(f"角度误差分布  MAE={np.abs(diff_deg).mean():.2f}°")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("conv_angle_eval.png", dpi=120)
    print("可视化已保存到 conv_angle_eval.png")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seqs", type=int, default=500, help="生成多少条序列")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
