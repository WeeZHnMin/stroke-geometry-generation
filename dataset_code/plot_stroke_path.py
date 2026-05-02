"""
plot_stroke_path.py
===================
将 dx, dy 从原点开始累积，画出完整的笔画路径（每步一个点），
清晰展示模型是如何"一笔一画"画出图形的。

用法:
  python dataset_code/plot_stroke_path.py --sample-id 0
  python dataset_code/plot_stroke_path.py --sample-id 5 --data path/to/data.jsonl
  python dataset_code/plot_stroke_path.py --grid 4x4
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

# 不同 pen_state 的颜色和样式
PEN_STYLES = {
    "move":     {"color": "#f59e0b", "marker": "s", "label": "move (抬笔移动)", "zorder": 3},
    "draw":     {"color": "#2563eb", "marker": "o", "label": "draw (落笔画线)", "zorder": 2},
    "end_shape": {"color": "#dc2626", "marker": "x", "label": "end_shape (图形结束)", "zorder": 4},
    "end_all":  {"color": "#dc2626", "marker": "*", "label": "end_all (全部结束)", "zorder": 5},
}


def strokes_to_path(strokes: list[dict]) -> list[dict]:
    """将 stroke 序列转为带累积坐标的路径点列表。"""
    x, y = 0.0, 0.0
    path = []
    for i, step in enumerate(strokes):
        dx, dy = float(step["dx"]), float(step["dy"])
        nx, ny = x + dx, y + dy
        path.append({
            "step": i,
            "dx": dx,
            "dy": dy,
            "x": nx,
            "y": ny,
            "pen_state": step["pen_state"],
        })
        x, y = nx, ny
    return path


def plot_single_path(ax, path: list[dict], prompt: str, show_details: bool = True, canvas_size: float = 0.0) -> None:
    """在 Axes 上绘制一条完整的笔画路径。"""
    xs = [p["x"] for p in path]
    ys = [p["y"] for p in path]

    # 画背景网格
    ax.grid(True, alpha=0.15)
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

    # 标注原点
    ax.plot(0, 0, "k+", markersize=10, mew=2, label="原点 (0,0)", zorder=6)
    ax.annotate("原点", (0, 0), (0.02, -0.04), fontsize=7, color="gray")

    # 逐段画线
    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        style = PEN_STYLES.get(curr["pen_state"], PEN_STYLES["draw"])

        if curr["pen_state"] == "move":
            # move 用虚线
            ax.plot([prev["x"], curr["x"]], [prev["y"], curr["y"]],
                    color=style["color"], linestyle="--", linewidth=1.5, alpha=0.7, zorder=style["zorder"])
        else:
            # draw 用实线
            ax.plot([prev["x"], curr["x"]], [prev["y"], curr["y"]],
                    color=style["color"], linestyle="-", linewidth=2.5, alpha=0.9, zorder=style["zorder"])

    # 画每个点
    for p in path:
        style = PEN_STYLES.get(p["pen_state"], PEN_STYLES["draw"])
        ax.plot(p["x"], p["y"], style["marker"], color=style["color"],
                markersize=4 if p["pen_state"] == "draw" else 7, zorder=style["zorder"] + 1)

    # 标注关键点
    # 起点（第一个点）
    first = path[0]
    ax.annotate(f"起点\n({first['x']:.3f}, {first['y']:.3f})",
                (first["x"], first["y"]),
                textcoords="offset points", xytext=(8, 8), fontsize=7, color="#f59e0b")

    # 终点（最后一个点）
    last = path[-1]
    ax.annotate(f"终点\n({last['x']:.3f}, {last['y']:.3f})",
                (last["x"], last["y"]),
                textcoords="offset points", xytext=(8, -10), fontsize=7, color="#dc2626")

    # 如果步数不多，给每步标注序号和 dx/dy
    if show_details and len(path) <= 15:
        for p in path:
            if p["step"] % 1 == 0:  # 标所有步
                offset = (6, 6) if p["pen_state"] != "move" else (-30, 6)
                ax.annotate(f"#{p['step']}\n({p['dx']:.3f}, {p['dy']:.3f})",
                            (p["x"], p["y"]),
                            textcoords="offset points", xytext=offset,
                            fontsize=6, alpha=0.7, color="gray")

    # 图例和标题
    if prompt:
        ax.set_title(prompt, fontsize=10, pad=8)


def render_single(samples: list[dict], idx: int, save_dir: Path) -> None:
    """渲染单条样本的笔画路径图。"""
    sample = samples[idx]
    path = strokes_to_path(sample["strokes"])
    canvas_size = float(sample.get("metadata", {}).get("canvas_size", sample.get("scene_spec", {}).get("canvas_size", 0.0)) or 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # 判断是否显示详细标注（步数少的显示）
    show_details = len(path) <= 15
    plot_single_path(ax, path, sample["prompt"], show_details=show_details, canvas_size=canvas_size)

    # 信息框
    n_steps = len(sample["strokes"])
    n_move = sum(1 for s in sample["strokes"] if s["pen_state"] == "move")
    n_draw = sum(1 for s in sample["strokes"] if s["pen_state"] == "draw")
    shape_type = sample["shapes"][0]["shape_type"]
    info = f"形状: {shape_type}  |  总步数: {n_steps} (move={n_move}, draw={n_draw})"
    fig.suptitle(info, fontsize=9, y=0.01, color="gray")

    fig.tight_layout()
    out_path = save_dir / f"path_{idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


def render_grid(samples: list[dict], rows: int, cols: int, save_dir: Path) -> None:
    """网格展示多条样本。"""
    n = min(rows * cols, len(samples))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows + 0.5))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(n):
        ax = axes_flat[i]
        path = strokes_to_path(samples[i]["strokes"])
        canvas_size = float(samples[i].get("metadata", {}).get("canvas_size", samples[i].get("scene_spec", {}).get("canvas_size", 0.0)) or 0.0)
        plot_single_path(ax, path, samples[i]["prompt"], show_details=False, canvas_size=canvas_size)
        ax.text(0.98, 0.02, f"#{i}", transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", color="gray", alpha=0.7)

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(f"笔画路径预览 ({n} 样本)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = save_dir / f"path_grid_{rows}x{cols}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


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
    parser = argparse.ArgumentParser(description="笔画路径可视化 — 从原点出发，一步步画出图形")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--sample-id", type=int, default=None, help="查看单条 (如 0)")
    parser.add_argument("--grid", type=str, default=None, help="网格 行x列 (如 4x4)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.data:
        data_path = Path(args.data)
    else:
        candidates = sorted((ROOT / "generated_data").glob("**/*verified*.jsonl"))
        if not candidates:
            candidates = sorted((ROOT / "generated_data").glob("**/*.jsonl"))
        if not candidates:
            print("错误: 未找到数据文件")
            sys.exit(1)
        data_path = candidates[-1]

    save_dir = Path(args.save_dir) if args.save_dir else ROOT / "viz_output" / "paths"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(data_path, limit=args.limit)
    print(f"加载 {len(samples)} 条样本")

    if args.sample_id is not None:
        render_single(samples, args.sample_id, save_dir)
        s = samples[args.sample_id]
        print(f"prompt: {s['prompt']}")
        print(f"形状: {s['shapes'][0]['shape_type']}")
    elif args.grid:
        rows, cols = map(int, args.grid.lower().split("x"))
        render_grid(samples, rows, cols, save_dir)
    else:
        # 默认: 展示前 3 条 + 4x4 网格
        for i in range(min(3, len(samples))):
            render_single(samples, i, save_dir)
        render_grid(samples, 4, 4, save_dir)


if __name__ == "__main__":
    main()
