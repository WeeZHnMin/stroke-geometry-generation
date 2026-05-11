"""
实验：100 层 2×2 约束矩阵（元素在 [-1,1]）+ GELU + 残差连接的记忆容量测试。
每个数据点 [x,y] 用不同的随机矩阵 M 变换，模型只看 [x,y]，看能拟合多少。
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── 数据生成 ──────────────────────────────────────────────────────────────────

def generate_data(n: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    xy = torch.empty(n, 2).uniform_(-1.0, 1.0, generator=rng)
    # 每个点各自的随机 2×2 矩阵，元素在 [-1,1]
    M = torch.empty(n, 2, 2).uniform_(-1.0, 1.0, generator=rng)
    # target_i = M_i @ xy_i
    target = torch.bmm(M, xy.unsqueeze(-1)).squeeze(-1)
    return xy, target


# ── 模型 ──────────────────────────────────────────────────────────────────────

class ConstrainedBlock(nn.Module):
    """单层：tanh(W_raw) @ h + b，然后 GELU，加残差。"""
    def __init__(self) -> None:
        super().__init__()
        self.W_raw = nn.Parameter(torch.randn(2, 2) * 0.5)
        self.b = nn.Parameter(torch.zeros(2))
        # 残差缩放因子，初始化为小值让训练初期接近恒等映射
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        W = torch.tanh(self.W_raw)          # 元素约束到 (-1, 1)
        out = F.gelu(h @ W.T + self.b)
        return h + self.alpha * out         # 残差：主路 + 缩放后的变换


class ConstrainedStack(nn.Module):
    def __init__(self, depth: int = 100) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ConstrainedBlock() for _ in range(depth)])
        self.out = nn.Linear(2, 2)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        h = xy
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    xy, target = generate_data(args.n_samples, seed=args.seed)
    xy, target = xy.to(device), target.to(device)

    dataset = TensorDataset(xy, target)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConstrainedStack(depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params}  数据量: {args.n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x_batch, t_batch in loader:
            pred = model(x_batch)
            loss = F.mse_loss(pred, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        epoch_loss /= args.n_samples
        losses.append(epoch_loss)
        if epoch % max(1, args.epochs // 10) == 0:
            mae = (model(xy) - target).abs().mean().item()
            print(f"epoch {epoch:4d}  MSE={epoch_loss:.4e}  MAE={mae:.4e}")

    # 最终结果
    with torch.no_grad():
        pred_all = model(xy)
        mse_final = F.mse_loss(pred_all, target).item()
        mae_final = (pred_all - target).abs().mean().item()
    print(f"\n最终  MSE={mse_final:.4e}  MAE={mae_final:.4e}")

    # 保存模型
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
    }, "matrix_memorize_model.pt")
    print("模型已保存到 matrix_memorize_model.pt")

    # 画 loss 曲线
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE (log scale)")
    plt.title(f"Matrix memorize — depth={args.depth}, n={args.n_samples}, params={n_params}")
    plt.tight_layout()
    plt.savefig("matrix_memorize_loss.png", dpi=120)
    print("loss 曲线已保存到 matrix_memorize_loss.png")

    evaluate(model, xy, target, n_show=20)


def evaluate(model: ConstrainedStack, xy: torch.Tensor, target: torch.Tensor, n_show: int = 200) -> None:
    model.eval()
    with torch.no_grad():
        pred = model(xy)

    # 取前 n_show 个点做可视化
    t_np = target[:n_show].cpu().numpy()
    p_np = pred[:n_show].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：真实 vs 预测散点叠加
    ax = axes[0]
    ax.scatter(t_np[:, 0], t_np[:, 1], s=15, alpha=0.6, label="真实", color="steelblue")
    ax.scatter(p_np[:, 0], p_np[:, 1], s=15, alpha=0.6, label="预测", color="tomato", marker="x")
    ax.set_title(f"真实 vs 预测（前 {n_show} 点）")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 右图：每个点画一条线连接真实和预测（误差线）
    ax2 = axes[1]
    for i in range(n_show):
        ax2.plot([t_np[i, 0], p_np[i, 0]], [t_np[i, 1], p_np[i, 1]],
                 color="gray", alpha=0.3, linewidth=0.8)
    ax2.scatter(t_np[:, 0], t_np[:, 1], s=15, alpha=0.7, label="真实", color="steelblue", zorder=3)
    ax2.scatter(p_np[:, 0], p_np[:, 1], s=15, alpha=0.7, label="预测", color="tomato", marker="x", zorder=3)
    ax2.set_title(f"误差连线（前 {n_show} 点）")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("matrix_memorize_eval.png", dpi=120)
    print(f"可视化已保存到 matrix_memorize_eval.png")

    # 整体误差统计
    mse = ((target.cpu() - pred.cpu()) ** 2).mean().item()
    mae = (target.cpu() - pred.cpu()).abs().mean().item()
    print(f"全量  MSE={mse:.4e}  MAE={mae:.4e}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_show", type=int, default=20, help="eval 模式下展示多少条对比")
    parser.add_argument("--ckpt", type=str, default="matrix_memorize_model.pt")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.ckpt, map_location=device)
        saved = ckpt["args"]
        model = ConstrainedStack(depth=saved["depth"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        xy, target = generate_data(saved["n_samples"], seed=saved["seed"])
        xy, target = xy.to(device), target.to(device)
        evaluate(model, xy, target, n_show=args.n_show)
