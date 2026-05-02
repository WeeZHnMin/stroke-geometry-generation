"""
plot_dxdy_distribution.py
=========================
可视化数据集中 dx, dy 的数值分布，帮助理解训练数据的统计特性。

用法:
  python dataset_code/plot_dxdy_distribution.py
  python dataset_code/plot_dxdy_distribution.py --data path/to/data.jsonl
  python dataset_code/plot_dxdy_distribution.py --data path/to/data.jsonl --save-dir my_plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if limit is not None and len(samples) >= limit:
                    break
    return samples


def collect_dxdy(samples: list[dict]) -> dict:
    """从样本中收集所有 dx, dy 值，按 pen_state 分类。"""
    all_dx, all_dy = [], []
    move_dx, move_dy = [], []
    draw_dx, draw_dy = [], []

    for sample in samples:
        x, y = 0.0, 0.0
        for step in sample["strokes"]:
            dx, dy = float(step["dx"]), float(step["dy"])
            pen = step["pen_state"]
            all_dx.append(dx)
            all_dy.append(dy)
            if pen == "move":
                move_dx.append(dx)
                move_dy.append(dy)
            elif pen == "draw":
                draw_dx.append(dx)
                draw_dy.append(dy)

    return {
        "all_dx": np.array(all_dx),
        "all_dy": np.array(all_dy),
        "move_dx": np.array(move_dx),
        "move_dy": np.array(move_dy),
        "draw_dx": np.array(draw_dx),
        "draw_dy": np.array(draw_dy),
    }


def plot_distribution(data: dict, save_dir: Path, prefix: str = "") -> None:
    """绘制 dx/dy 分布的多面板图。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 图1: 直方图 ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"{prefix}dx/dy 数值分布", fontsize=14)

    titles = [
        ("all_dx", "所有 dx"),
        ("all_dy", "所有 dy"),
        ("move_dx", "move 步 dx"),
        ("move_dy", "move 步 dy"),
        ("draw_dx", "draw 步 dx"),
        ("draw_dy", "draw 步 dy"),
    ]

    for ax, (key, title) in zip(axes.flat, titles):
        values = data[key]
        ax.hist(values, bins=80, color="#2563eb", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{title} (n={len(values)})")
        ax.set_xlabel("dx/dy 值")
        ax.set_ylabel("频次")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        # 标注统计值
        stats = f"均值={np.mean(values):.4f}\n标准差={np.std(values):.4f}\nmin={np.min(values):.4f}\nmax={np.max(values):.4f}"
        ax.text(0.97, 0.93, stats, transform=ax.transAxes, fontsize=7,
                va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}histogram.png")
    plt.close(fig)
    print(f"  已保存: {save_dir / f'{prefix}histogram.png'}")

    # ── 图2: 箱线图 ──
    fig, ax = plt.subplots(figsize=(8, 4))
    keys_plot = ["all_dx", "all_dy", "move_dx", "move_dy", "draw_dx", "draw_dy"]
    labels = ["所有 dx", "所有 dy", "move dx", "move dy", "draw dx", "draw dy"]
    values_list = [data[k] for k in keys_plot]

    bp = ax.boxplot(values_list, labels=labels, patch_artist=True, showfliers=False)
    colors = ["#2563eb", "#60a5fa", "#dc2626", "#f87171", "#16a34a", "#4ade80"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"{prefix}dx/dy 箱线图")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}boxplot.png")
    plt.close(fig)
    print(f"  已保存: {save_dir / f'{prefix}boxplot.png'}")

    # ── 图3: draw 步局部放大直方图 ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{prefix}draw 步局部放大 (|dx|,|dy| < 0.1)", fontsize=12)

    for ax, (values, label) in zip(axes, [(data["draw_dx"], "draw dx"), (data["draw_dy"], "draw dy")]):
        mask = np.abs(values) < 0.1
        filtered = values[mask]
        ax.hist(filtered, bins=60, color="#16a34a", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{label} (n={len(filtered)})")
        ax.set_xlabel("dx/dy 值")
        ax.set_ylabel("频次")
        ax.grid(True, alpha=0.2)
        stats = f"均值={np.mean(filtered):.5f}\n标准差={np.std(filtered):.5f}"
        ax.text(0.97, 0.93, stats, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}draw_zoom.png")
    plt.close(fig)
    print(f"  已保存: {save_dir / f'{prefix}draw_zoom.png'}")


def print_stats(data: dict, title: str = "") -> None:
    """打印统计数据。"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    categories = [
        ("所有步 dx", data["all_dx"]),
        ("所有步 dy", data["all_dy"]),
        ("move 步 dx", data["move_dx"]),
        ("move 步 dy", data["move_dy"]),
        ("draw 步 dx", data["draw_dx"]),
        ("draw 步 dy", data["draw_dy"]),
    ]
    print(f"{'类别':>16} {'数量':>8} {'均值':>10} {'标准差':>10} {'min':>10} {'max':>10}")
    print("-" * 68)
    for name, values in categories:
        print(f"{name:>16} {len(values):>8} {np.mean(values):>10.5f} {np.std(values):>10.5f} {np.min(values):>10.5f} {np.max(values):>10.5f}")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 dx/dy 数值分布")
    parser.add_argument("--data", type=str, default=None, help="JSONL 数据文件路径")
    parser.add_argument("--save-dir", type=str, default=None, help="图片保存目录")
    parser.add_argument("--limit", type=int, default=500, help="最多加载多少条样本")
    args = parser.parse_args()

    # 找数据文件
    if args.data:
        data_path = Path(args.data)
    else:
        candidates = sorted((ROOT / "generated_data").glob("**/*.jsonl"))
        if not candidates:
            print("错误: 未找到数据文件")
            sys.exit(1)
        # 优先用 verified 数据集
        verified = [p for p in candidates if "verified" in p.name]
        data_path = (verified or candidates)[-1]
        print(f"自动选择: {data_path}")

    save_dir = Path(args.save_dir) if args.save_dir else ROOT / "viz_output" / "dxdy_stats"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(data_path, limit=args.limit)
    print(f"加载 {len(samples)} 条样本")

    data = collect_dxdy(samples)
    print_stats(data, title=f"数据集: {data_path.name}")
    plot_distribution(data, save_dir)


if __name__ == "__main__":
    main()
