from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_code"))

from stroke_baseline.polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig
from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.metadata_annotator import annotate_metadata
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

SHAPE_ZH = {
    "square": "正方形",
    "rectangle": "矩形",
    "wide_rectangle": "宽矩形",
    "tall_rectangle": "高矩形",
    "triangle": "三角形",
    "right_triangle": "直角三角形",
    "circle": "圆",
    "ellipse": "椭圆",
    "line": "线段",
}


def scale_shape_to_canvas(shape: ShapeSample, canvas_size: float) -> ShapeSample:
    points = [Point(p.x * canvas_size, p.y * canvas_size) for p in shape.points]
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=polygon_bbox(points),
        closed=shape.closed,
    )


def size_zh(bbox: dict[str, float], canvas_size: float) -> str:
    span = max(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"]) / canvas_size
    if span < 0.18:
        return "小"
    if span < 0.30:
        return "中等"
    return "大"


def position_zh(bbox: dict[str, float], canvas_size: float) -> str:
    cx = ((bbox["x_min"] + bbox["x_max"]) / 2) / canvas_size
    cy = ((bbox["y_min"] + bbox["y_max"]) / 2) / canvas_size
    horiz = "左边" if cx < 0.35 else "右边" if cx > 0.65 else "中间"
    vert = "上方" if cy < 0.35 else "下方" if cy > 0.65 else "中部"
    if horiz == "中间" and vert == "中部":
        return "画布中央"
    if horiz == "中间":
        return f"画布{vert}"
    if vert == "中部":
        return f"画布{horiz}"
    return f"画布{vert}{horiz}"


def make_prompt(shape_type: str, bbox: dict[str, float], rng: random.Random, canvas_size: float) -> str:
    shape = SHAPE_ZH[shape_type]
    size = size_zh(bbox, canvas_size)
    position = position_zh(bbox, canvas_size)
    return rng.choice(
        [
            f"画一个{size}{shape}，位置在{position}",
            f"在{position}画一个{size}{shape}",
            f"请画一个{size}{shape}，放在{position}",
        ]
    )


def quantize_theta(theta: float, theta_bins: int) -> int:
    return int(round((theta % (2 * math.pi)) / (2 * math.pi) * theta_bins)) % theta_bins


def choose_distance_id(remaining: float, buckets: tuple[float, ...], min_remainder: float) -> int | None:
    if remaining < min_remainder:
        return None
    valid = [idx for idx, d in enumerate(buckets) if d <= remaining + 1e-9]
    if valid:
        return max(valid)
    return 0


def append_polar_segment(
    actions: list[dict],
    start: Point,
    end: Point,
    pen_state: str,
    tokenizer: PolarActionTokenizer,
    min_remainder: float,
    max_steps: int = 1000,
) -> Point:
    current = Point(start.x, start.y)
    for _ in range(max_steps):
        vx = end.x - current.x
        vy = end.y - current.y
        remaining = math.hypot(vx, vy)
        distance_id = choose_distance_id(remaining, tokenizer.distance_buckets, min_remainder)
        if distance_id is None:
            break
        distance = tokenizer.distance_buckets[distance_id]
        theta_id = quantize_theta(math.atan2(vy, vx), tokenizer.theta_bins)
        theta = 2 * math.pi * theta_id / tokenizer.theta_bins
        next_point = Point(current.x + distance * math.cos(theta), current.y + distance * math.sin(theta))
        actions.append(
            {
                "distance_id": distance_id,
                "distance": distance,
                "theta_id": theta_id,
                "theta": theta,
                "pen_state": pen_state,
                "dx": next_point.x - current.x,
                "dy": next_point.y - current.y,
            }
        )
        current = next_point
    return current


def shape_path(shape: ShapeSample) -> list[Point]:
    points = list(shape.points)
    if shape.closed and points:
        points.append(points[0])
    return points


def polar_actions_to_strokes(actions: list[dict]) -> list[StrokeStep]:
    return [StrokeStep(dx=a["dx"], dy=a["dy"], pen_state=a["pen_state"]) for a in actions]


def build_polar_actions(shape: ShapeSample, tokenizer: PolarActionTokenizer, min_remainder: float) -> list[dict]:
    points = shape_path(shape)
    if not points:
        return []
    actions: list[dict] = []
    cursor = Point(0.0, 0.0)
    cursor = append_polar_segment(actions, cursor, points[0], "move", tokenizer, min_remainder)
    for target in points[1:]:
        cursor = append_polar_segment(actions, cursor, target, "draw", tokenizer, min_remainder)
    if actions:
        actions[-1]["pen_state"] = "end_all"
    return actions


def sample_scene(
    rng: random.Random,
    shape_types: list[str],
    tokenizer: PolarActionTokenizer,
    canvas_size: float,
    min_remainder: float,
) -> dict:
    shape = scale_shape_to_canvas(build_shape_from_type(rng.choice(shape_types), rng), canvas_size)
    polar_actions = build_polar_actions(shape, tokenizer, min_remainder)
    tokens = [
        tokenizer.encode_action(action["distance_id"], action["theta_id"], action["pen_state"])
        for action in polar_actions
    ]
    strokes = polar_actions_to_strokes(polar_actions)
    prompt = make_prompt(shape.shape_type, shape.bbox, rng, canvas_size)
    scene_spec = {
        "scene_type": "single_basic_polar",
        "difficulty": "easy",
        "num_shapes": 1,
        "allowed_shapes": shape_types,
        "relation_type": None,
        "anchor_position": None,
        "recipe_name": "polar_single_basic",
        "canvas_size": canvas_size,
        "distance_buckets": list(tokenizer.distance_buckets),
        "theta_bins": tokenizer.theta_bins,
        "min_remainder": min_remainder,
    }
    metadata = annotate_metadata({**scene_spec, "scene_type": "single_basic"}, [shape], strokes)
    metadata.update(
        {
            "scene_type": scene_spec["scene_type"],
            "prompt_language": "zh",
            "canvas_size": canvas_size,
            "distance_buckets": list(tokenizer.distance_buckets),
            "theta_bins": tokenizer.theta_bins,
            "min_remainder": min_remainder,
            "num_polar_actions": len(tokens),
        }
    )
    return {
        "scene_spec": scene_spec,
        "prompt": prompt,
        "shapes": [
            {
                "shape_type": shape.shape_type,
                "prompt_fragment": prompt,
                "points": [asdict(p) for p in shape.points],
                "bbox": shape.bbox,
                "closed": shape.closed,
            }
        ],
        "polar_actions": polar_actions,
        "action_tokens": tokens,
        "strokes": [asdict(step) for step in strokes],
        "metadata": metadata,
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_report(samples: list[dict]) -> None:
    lengths = [len(s["action_tokens"]) for s in samples]
    shape_counts: dict[str, int] = {}
    for sample in samples:
        shape_type = sample["shapes"][0]["shape_type"]
        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
    print(f"samples={len(samples)}")
    print(f"shape_counts={dict(sorted(shape_counts.items()))}")
    print(f"action_len min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polar action token stroke dataset.")
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--distances", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--min-remainder", type=float, default=0.05)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shape_types = [s.strip() for s in args.shapes.split(",") if s.strip()]
    distances = tuple(float(v.strip()) for v in args.distances.split(",") if v.strip())
    tokenizer = PolarActionTokenizer(PolarActionTokenizerConfig(distance_buckets=distances, theta_bins=args.theta_bins))
    samples = [
        sample_scene(rng, shape_types, tokenizer, args.canvas_size, args.min_remainder)
        for _ in range(args.num_samples)
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else ROOT / "generated_data" / "polar" / f"polar_scale{args.canvas_size:g}_{timestamp}.jsonl"
    save_jsonl(samples, output)
    print_report(samples)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
