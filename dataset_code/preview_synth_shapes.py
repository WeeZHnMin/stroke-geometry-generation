"""Render a grid PNG of the first N samples in a synthesized stroke jsonl,
so you can eyeball the data before training.

Usage:
    python -m dataset_code.preview_synth_shapes \
        --data generated_data/synth_shapes_v0.jsonl \
        --output runs/synth_shapes_v0/data_preview.png \
        --grid 6 6
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import sys

# Ensure the local stroke_baseline package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stroke_baseline.visualize import strokes_to_xy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--grid", type=int, nargs=2, default=(6, 6), help="rows cols")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, cols = args.grid
    total = rows * cols

    samples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= total:
                break
    if not samples:
        raise RuntimeError(f"no samples in {args.data}")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4), dpi=110)
    axes = axes.reshape(rows, cols) if rows * cols > 1 else [[axes]]

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if i >= len(samples):
            ax.set_visible(False)
            continue
        sample = samples[i]
        strokes = sample["strokes"]
        prompt = sample.get("prompt", "?")
        points = strokes_to_xy(strokes)
        # Draw segments, breaking on `move`.
        prev = None
        for x, y, pen in points:
            if pen == "move" or prev is None:
                prev = (x, y)
                continue
            ax.plot([prev[0], x], [prev[1], y], color="black", linewidth=1.4)
            prev = (x, y)
        # Mark start point.
        if points:
            ax.scatter([points[0][0]], [points[0][1]], c="red", s=10, zorder=5)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0.55, -0.55)  # invert y so canvas reads top-down
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{prompt} ({len(strokes)})", fontsize=8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}  ({len(samples)} samples drawn)")


if __name__ == "__main__":
    main()
