from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def accumulate_strokes(strokes: list[dict]) -> list[dict]:
    x = 0.0
    y = 0.0
    points = [{"x": x, "y": y, "pen_state": "origin"}]
    for step in strokes:
        x += float(step["dx"])
        y += float(step["dy"])
        points.append({"x": x, "y": y, "pen_state": str(step["pen_state"])})
    return points


def draw_sample(ax, sample: dict, index: int) -> None:
    metadata = sample.get("metadata", {})
    canvas_size = float(metadata.get("canvas_size", sample.get("scene_spec", {}).get("canvas_size", 8.0)))
    shape = sample.get("shapes", [{}])[0]
    original = shape.get("points", [])
    closed = bool(shape.get("closed", False))
    if closed and original:
        original = original + [original[0]]

    traced = accumulate_strokes(sample.get("strokes", []))

    if len(original) >= 2:
        ax.plot(
            [p["x"] for p in original],
            [p["y"] for p in original],
            color="#94a3b8",
            linewidth=1.2,
            linestyle="--",
            label="original",
        )

    for prev, curr in zip(traced[:-1], traced[1:]):
        pen = curr["pen_state"]
        if pen == "move":
            color = "#b8b8b8"
            linewidth = 1.1
            alpha = 0.75
        else:
            color = "#111827"
            linewidth = 2.0
            alpha = 1.0
        ax.plot([prev["x"], curr["x"]], [prev["y"], curr["y"]], color=color, linewidth=linewidth, alpha=alpha)

    if len(traced) > 1:
        sampled = traced[1:]
        ax.scatter([p["x"] for p in sampled], [p["y"] for p in sampled], s=8, color="#f59e0b", alpha=0.75)
        ax.scatter([traced[0]["x"]], [traced[0]["y"]], s=32, color="#2563eb", label="origin", zorder=5)
        ax.scatter([sampled[0]["x"]], [sampled[0]["y"]], s=42, marker="s", color="#16a34a", label="start", zorder=6)
        ax.scatter([sampled[-1]["x"]], [sampled[-1]["y"]], s=56, marker="*", color="#dc2626", label="end", zorder=6)

    pad = canvas_size * 0.04
    ax.set_xlim(-pad, canvas_size + pad)
    ax.set_ylim(canvas_size + pad, -pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16)

    shape_type = shape.get("shape_type", "unknown")
    prompt = sample.get("prompt", "")
    num_actions = len(sample.get("action_tokens", []))
    ax.set_title(f"#{index} {shape_type} | actions={num_actions}\n{prompt}", fontsize=8)


def save_grid(records: list[dict], output: Path, rows: int, cols: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), dpi=150)
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    for idx, ax in enumerate(axes_flat):
        if idx < len(records):
            draw_sample(ax, records[idx], idx)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_individual(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(records):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        draw_sample(ax, sample, idx)
        fig.tight_layout()
        fig.savefig(output_dir / f"polar_sample_{idx:04d}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated polar-token training data.")
    parser.add_argument("--data", type=str, required=True, help="Path to polar JSONL dataset.")
    parser.add_argument("--output-dir", type=str, default="viz_output/polar_dataset")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--grid", type=str, default="4x4", help="Grid size, for example 4x4.")
    parser.add_argument("--individual", action="store_true", help="Also save one PNG per sample.")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    rows, cols = [int(v) for v in args.grid.lower().split("x")]
    limit = min(args.limit, rows * cols)
    records = load_jsonl(data_path, limit=limit)

    grid_path = output_dir / "grid.png"
    save_grid(records, grid_path, rows, cols)
    if args.individual:
        save_individual(records, output_dir / "individual")

    print(f"loaded samples: {len(records)}")
    print(f"saved grid: {grid_path}")
    if args.individual:
        print(f"saved individual images: {output_dir / 'individual'}")


if __name__ == "__main__":
    main()
