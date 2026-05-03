"""
Angle-distance sampling experiment.

Generate large-scale geometric shapes, then approximate each shape path with
discrete polar steps:

    dx = d * cos(theta)
    dy = d * sin(theta)

where d is chosen from {0.1, 0.2, 0.3, 0.4, 0.5} and theta is quantized into
360 bins over [0, 2*pi).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_code"))

from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.schema import Point, ShapeSample, StrokeStep
from stroke_data_factory.utils import polygon_bbox


DEFAULT_SHAPES = [
    "square",
    "rectangle",
    "wide_rectangle",
    "tall_rectangle",
    "triangle",
    "right_triangle",
    "circle",
    "ellipse",
    "line",
]

DISTANCE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5]
THETA_BINS = 360


def scale_shape(shape: ShapeSample, canvas_size: float) -> ShapeSample:
    points = [Point(p.x * canvas_size, p.y * canvas_size) for p in shape.points]
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=polygon_bbox(points),
        closed=shape.closed,
    )


def path_points(shape: ShapeSample) -> list[Point]:
    points = list(shape.points)
    if shape.closed and points:
        points.append(points[0])
    return points


def quantize_theta(theta: float, theta_bins: int = THETA_BINS) -> tuple[int, float]:
    full = 2 * math.pi
    theta = theta % full
    idx = int(round(theta / full * theta_bins)) % theta_bins
    return idx, idx * full / theta_bins


def choose_distance(remaining: float, buckets: list[float]) -> float:
    valid = [d for d in buckets if d <= remaining + 1e-9]
    return max(valid) if valid else min(buckets)


def polar_sample_segment(
    start: Point,
    end: Point,
    buckets: list[float],
    theta_bins: int,
    max_steps: int = 1000,
) -> tuple[list[Point], list[dict]]:
    """Approximate one segment with quantized angle and bucketed distance.

    The last step is snapped to the true endpoint if the remaining distance is
    smaller than the minimum bucket. This keeps the shape connected while still
    recording the quantized theta used for the step.
    """
    current = Point(start.x, start.y)
    sampled: list[Point] = []
    actions: list[dict] = []

    for _ in range(max_steps):
        vx = end.x - current.x
        vy = end.y - current.y
        remaining = math.hypot(vx, vy)
        if remaining < 1e-6:
            break

        theta_idx, theta_q = quantize_theta(math.atan2(vy, vx), theta_bins)
        d = choose_distance(remaining, buckets)
        if remaining <= min(buckets):
            next_point = Point(end.x, end.y)
            actual_d = remaining
        else:
            next_point = Point(current.x + d * math.cos(theta_q), current.y + d * math.sin(theta_q))
            actual_d = d

        sampled.append(next_point)
        actions.append(
            {
                "d_bucket": d,
                "actual_d": actual_d,
                "theta_bin": theta_idx,
                "theta": theta_q,
                "dx": next_point.x - current.x,
                "dy": next_point.y - current.y,
            }
        )
        current = next_point

    return sampled, actions


def polar_sample_shape(
    shape: ShapeSample,
    buckets: list[float],
    theta_bins: int,
) -> tuple[list[Point], list[dict]]:
    points = path_points(shape)
    if len(points) < 2:
        return points, []

    sampled = [points[0]]
    actions: list[dict] = []
    for a, b in zip(points[:-1], points[1:]):
        seg_points, seg_actions = polar_sample_segment(a, b, buckets, theta_bins)
        sampled.extend(seg_points)
        actions.extend(seg_actions)
    return sampled, actions


def sampled_points_to_strokes(points: list[Point], closed: bool) -> list[StrokeStep]:
    if not points:
        return []
    steps = [StrokeStep(dx=points[0].x, dy=points[0].y, pen_state="move")]
    prev = points[0]
    for idx, point in enumerate(points[1:], start=1):
        pen_state = "draw" if idx < len(points) - 1 else "end_all"
        steps.append(StrokeStep(dx=point.x - prev.x, dy=point.y - prev.y, pen_state=pen_state))
        prev = point
    return steps


def sample_record(rng: random.Random, shape_types: list[str], canvas_size: float, buckets: list[float], theta_bins: int) -> dict:
    shape_type = rng.choice(shape_types)
    shape = scale_shape(build_shape_from_type(shape_type, rng), canvas_size)
    sampled_points, actions = polar_sample_shape(shape, buckets, theta_bins)
    strokes = sampled_points_to_strokes(sampled_points, shape.closed)

    return {
        "prompt": f"angle-distance sampled {shape.shape_type}",
        "shape": {
            "shape_type": shape.shape_type,
            "points": [asdict(p) for p in shape.points],
            "bbox": shape.bbox,
            "closed": shape.closed,
        },
        "sampled_points": [asdict(p) for p in sampled_points],
        "strokes": [asdict(s) for s in strokes],
        "polar_actions": actions,
        "metadata": {
            "canvas_size": canvas_size,
            "distance_buckets": buckets,
            "theta_bins": theta_bins,
            "num_steps": len(actions),
        },
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def draw_record(ax, record: dict, show_title: bool = True) -> None:
    original = [Point(**p) for p in record["shape"]["points"]]
    if record["shape"]["closed"] and original:
        original = original + [original[0]]
    sampled = [Point(**p) for p in record["sampled_points"]]
    canvas_size = float(record["metadata"]["canvas_size"])

    if len(original) >= 2:
        ax.plot([p.x for p in original], [p.y for p in original], color="#94a3b8", linewidth=1.2, linestyle="--", label="original")
    if len(sampled) >= 2:
        ax.plot([p.x for p in sampled], [p.y for p in sampled], color="#2563eb", linewidth=2.0, label="polar sampled")
        ax.scatter([p.x for p in sampled], [p.y for p in sampled], s=8, color="#f59e0b", alpha=0.75)

    if sampled:
        ax.scatter([sampled[0].x], [sampled[0].y], s=40, marker="s", color="#16a34a", zorder=5)
        ax.scatter([sampled[-1].x], [sampled[-1].y], s=50, marker="*", color="#dc2626", zorder=5)

    pad = canvas_size * 0.04
    ax.set_xlim(-pad, canvas_size + pad)
    ax.set_ylim(canvas_size + pad, -pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    if show_title:
        ax.set_title(f"{record['shape']['shape_type']} | steps={record['metadata']['num_steps']}", fontsize=9)


def render_grid(records: list[dict], output: Path, rows: int, cols: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    for idx, ax in enumerate(axes_flat):
        if idx < len(records):
            draw_record(ax, records[idx], show_title=True)
            ax.text(0.98, 0.02, f"#{idx}", transform=ax.transAxes, ha="right", va="bottom", color="gray", fontsize=8)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def render_individual(records: list[dict], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records[:limit]):
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_record(ax, record, show_title=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"angle_distance_{idx:04d}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and visualize angle-distance sampled shapes.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--distances", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None)
    parser.add_argument("--grid", type=str, default="4x4")
    parser.add_argument("--individual", type=int, default=12)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shape_types = [s.strip() for s in args.shapes.split(",") if s.strip()]
    buckets = [float(v.strip()) for v in args.distances.split(",") if v.strip()]

    records = [
        sample_record(rng, shape_types, args.canvas_size, buckets, args.theta_bins)
        for _ in range(args.num_samples)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "generated_data" / "angle_distance" / f"angle_distance_scale{args.canvas_size:g}_{timestamp}.jsonl"
    viz_dir = Path(args.viz_dir) if args.viz_dir else ROOT / "viz_output" / "angle_distance"

    save_jsonl(records, output)
    rows, cols = map(int, args.grid.lower().split("x"))
    render_grid(records, viz_dir / f"grid_{rows}x{cols}.png", rows, cols)
    render_individual(records, viz_dir / "individual", args.individual)

    print(f"saved data: {output}")
    print(f"saved grid: {viz_dir / f'grid_{rows}x{cols}.png'}")
    print(f"saved individual images: {viz_dir / 'individual'}")


if __name__ == "__main__":
    main()
