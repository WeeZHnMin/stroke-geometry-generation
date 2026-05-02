"""
visualize_dataset.py
====================
将数据集中的笔画渲染为 PNG 图片，直观查看图形和对应的提示词。

用法:
  # 查看单条样本
  python dataset_code/visualize_dataset.py --sample-id 0

  # 生成网格概览 (4x4 = 16 张图)
  python dataset_code/visualize_dataset.py --grid 4x4

  # 批量导出所有样本
  python dataset_code/visualize_dataset.py --export-dir viz_output

  # 指定数据文件
  python dataset_code/visualize_dataset.py --data path/to/data.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ── 确保能找到 stroke_baseline ──────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def strokes_to_segments(strokes: list[dict]) -> list[list[tuple[float, float]]]:
    """将 stroke 序列转换为线段列表（跳过 move，遇到 end_shape 断线）"""
    x, y = 0.0, 0.0
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for step in strokes:
        pen = step["pen_state"]
        nx = x + float(step["dx"])
        ny = y + float(step["dy"])
        if pen == "move":
            if current:
                segments.append(current)
                current = []
            x, y = nx, ny
        else:
            if pen == "draw":
                current.append(((x, y), (nx, ny)))
            elif pen == "end_shape" or pen == "end_all":
                current.append(((x, y), (nx, ny)))
                if current:
                    segments.append(current)
                    current = []
            x, y = nx, ny

    if current:
        segments.append(current)
    return segments


def draw_sample_on_ax(ax, sample: dict, show_title: bool = True) -> None:
    """在 matplotlib Axes 上绘制一个样本的笔画。"""
    strokes = sample["strokes"]
    prompt = sample["prompt"]
    canvas_size = float(sample.get("metadata", {}).get("canvas_size", sample.get("scene_spec", {}).get("canvas_size", 0.0)) or 0.0)

    segments = strokes_to_segments(strokes)
    xs: list[float] = [0.0]
    ys: list[float] = [0.0]

    # 为不同 shape 分配不同颜色
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
    color_idx = 0

    for seg in segments:
        color = colors[color_idx % len(colors)]
        color_idx += 1
        for (x1, y1), (x2, y2) in seg:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, solid_capstyle="round")
            xs.extend([x1, x2])
            ys.extend([y1, y2])

    if canvas_size > 0:
        pad = canvas_size * 0.04
        ax.set_xlim(-pad, canvas_size + pad)
        ax.set_ylim(canvas_size + pad, -pad)
    else:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pad = span * 0.08
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)
    ax.set_aspect("equal")
    ax.axis("off")

    if show_title:
        ax.set_title(prompt, fontsize=10, pad=6, loc="center")


def render_single(sample: dict, output: Path) -> None:
    """单条样本 → 独立 PNG"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_sample_on_ax(ax, sample, show_title=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output}")


def render_grid(samples: list[dict], rows: int, cols: int, output: Path) -> None:
    """多条样本 → 网格 PNG"""
    n = min(rows * cols, len(samples))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 0.5))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(n):
        ax = axes_flat[i]
        draw_sample_on_ax(ax, samples[i], show_title=True)
        # 右上角标序号
        ax.text(0.98, 0.02, f"#{i}", transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", color="gray", alpha=0.7)

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(f"数据集预览 ({n} 样本)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"已保存网格图: {output}")


def render_individual(samples: list[dict], output_dir: Path, limit: int | None = None) -> None:
    """批量导出每条样本为独立 PNG"""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(samples), limit or len(samples))

    for i in range(count):
        s = samples[i]
        # 文件名用序号 + prompt 前几个字
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in s["prompt"])[:40]
        path = output_dir / f"{i:04d}_{safe_name}.png"
        render_single(s, path)

    print(f"共导出 {count} 张图片到 {output_dir}")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="可视化笔画数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=None,
                        help="JSONL 数据文件路径 (默认: 最新生成的 verified 数据集)")
    parser.add_argument("--sample-id", type=int, default=None,
                        help="查看单条样本 (如 --sample-id 0)")
    parser.add_argument("--grid", type=str, default=None,
                        help="网格概览，如 4x4 (行x列)")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="批量导出到目录")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多加载多少条样本")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径 (默认: dataset_vis/)")
    args = parser.parse_args()

    # ── 找数据文件 ──
    if args.data:
        data_path = Path(args.data)
    else:
        # 自动找最新生成的 verified 文件
        chinese_dir = ROOT / "generated_data" / "chinese_mvp"
        candidates = sorted(chinese_dir.glob("chinese_mvp_single_basic_verified_*.jsonl"))
        if not candidates:
            # 也尝试旧文件
            candidates = sorted(chinese_dir.glob("*.jsonl"))
        if not candidates:
            # 尝试其他位置
            candidates = sorted((ROOT / "generated_data").glob("**/*.jsonl"))
        if not candidates:
            print("错误: 未找到数据文件，请用 --data 指定路径")
            sys.exit(1)
        data_path = candidates[-1]
        print(f"自动选择数据文件: {data_path}")

    samples = load_jsonl(data_path, limit=args.limit)
    print(f"加载 {len(samples)} 条样本")

    output_dir = Path(args.output) if args.output else ROOT / "viz_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 模式分发 ──
    if args.sample_id is not None:
        idx = args.sample_id
        if idx < 0 or idx >= len(samples):
            print(f"错误: sample-id {idx} 超出范围 [0, {len(samples) - 1}]")
            sys.exit(1)
        png = output_dir / f"sample_{idx:04d}.png"
        render_single(samples[idx], png)
        # 也打印 prompt
        print(f"\nprompt: {samples[idx]['prompt']}")
        print(f"形状: {samples[idx]['shapes'][0]['shape_type']}")
        print(f"笔画步数: {len(samples[idx]['strokes'])}")

    elif args.grid:
        try:
            rows, cols = map(int, args.grid.lower().split("x"))
        except ValueError:
            print("错误: --grid 格式应为 行x列，如 4x4")
            sys.exit(1)
        png = output_dir / f"grid_{rows}x{cols}.png"
        render_grid(samples, rows, cols, png)

    elif args.export_dir:
        export_path = Path(args.export_dir)
        render_individual(samples, export_path, limit=args.limit)

    else:
        # 默认: 生成一张 4x4 网格 + 头 3 条独立图
        print("未指定模式，默认生成 4x4 网格...")
        render_grid(samples, 4, 4, output_dir / "grid_4x4.png")
        render_individual(samples, output_dir / "individual", limit=3)
        print(f"\n提示: 想查看更多？用 --grid 4x8 或 --export-dir 目录名")


if __name__ == "__main__":
    main()
