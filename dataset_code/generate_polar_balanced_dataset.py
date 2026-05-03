from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_code"))

from stroke_baseline.polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig
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

POSITION_CENTERS = {
    "top_left": (0.24, 0.24),
    "top": (0.50, 0.22),
    "top_right": (0.76, 0.24),
    "left": (0.23, 0.50),
    "center": (0.50, 0.50),
    "right": (0.77, 0.50),
    "bottom_left": (0.24, 0.76),
    "bottom": (0.50, 0.78),
    "bottom_right": (0.76, 0.76),
}

SIZE_SPANS = {
    "small": 1.15,
    "medium": 1.75,
    "large": 2.35,
}

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

SIZE_ZH = {
    "small": "小",
    "medium": "中等",
    "large": "大",
}

POSITION_ZH = {
    "top_left": "画布左上方",
    "top": "画布上方",
    "top_right": "画布右上方",
    "left": "画布左边",
    "center": "画布中央",
    "right": "画布右边",
    "bottom_left": "画布左下方",
    "bottom": "画布下方",
    "bottom_right": "画布右下方",
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def quantize_theta(theta: float, theta_bins: int) -> int:
    return int(round((theta % (2 * math.pi)) / (2 * math.pi) * theta_bins)) % theta_bins


def make_shape(shape_type: str, size_name: str, position_name: str, canvas_size: float, rng: random.Random) -> ShapeSample:
    span = SIZE_SPANS[size_name]
    cx_ratio, cy_ratio = POSITION_CENTERS[position_name]
    margin = span * 0.62
    cx = clamp(cx_ratio * canvas_size + rng.uniform(-0.16, 0.16), margin, canvas_size - margin)
    cy = clamp(cy_ratio * canvas_size + rng.uniform(-0.16, 0.16), margin, canvas_size - margin)

    points: list[Point]
    closed = True
    if shape_type == "square":
        half = span / 2
        points = [
            Point(cx - half, cy - half),
            Point(cx + half, cy - half),
            Point(cx + half, cy + half),
            Point(cx - half, cy + half),
        ]
    elif shape_type == "rectangle":
        half_w = span * 0.58
        half_h = span * 0.40
        points = [
            Point(cx - half_w, cy - half_h),
            Point(cx + half_w, cy - half_h),
            Point(cx + half_w, cy + half_h),
            Point(cx - half_w, cy + half_h),
        ]
    elif shape_type == "wide_rectangle":
        half_w = span * 0.72
        half_h = span * 0.30
        points = [
            Point(cx - half_w, cy - half_h),
            Point(cx + half_w, cy - half_h),
            Point(cx + half_w, cy + half_h),
            Point(cx - half_w, cy + half_h),
        ]
    elif shape_type == "tall_rectangle":
        half_w = span * 0.32
        half_h = span * 0.74
        points = [
            Point(cx - half_w, cy - half_h),
            Point(cx + half_w, cy - half_h),
            Point(cx + half_w, cy + half_h),
            Point(cx - half_w, cy + half_h),
        ]
    elif shape_type == "triangle":
        r = span * 0.68
        rot = rng.uniform(-0.18, 0.18)
        points = [
            Point(cx + r * math.cos(rot - math.pi / 2), cy + r * math.sin(rot - math.pi / 2)),
            Point(cx + r * math.cos(rot + math.pi * 5 / 6), cy + r * math.sin(rot + math.pi * 5 / 6)),
            Point(cx + r * math.cos(rot + math.pi / 6), cy + r * math.sin(rot + math.pi / 6)),
        ]
    elif shape_type == "right_triangle":
        w = span * 1.15
        h = span * 1.10
        x0 = cx - w / 2
        y0 = cy + h / 2
        points = [Point(x0, y0), Point(x0 + w, y0), Point(x0, y0 - h)]
    elif shape_type == "circle":
        r = span * 0.50
        n = 32
        points = [Point(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elif shape_type == "ellipse":
        rx = span * 0.62
        ry = span * 0.42
        n = 32
        points = [Point(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elif shape_type == "line":
        closed = False
        angle = rng.choice([0, math.pi / 4, math.pi / 2, math.pi * 3 / 4, -math.pi / 4])
        half = span * 0.62
        points = [
            Point(cx - half * math.cos(angle), cy - half * math.sin(angle)),
            Point(cx + half * math.cos(angle), cy + half * math.sin(angle)),
        ]
    else:
        raise ValueError(f"unsupported shape type: {shape_type}")

    points = [Point(clamp(p.x, 0.0, canvas_size), clamp(p.y, 0.0, canvas_size)) for p in points]
    return ShapeSample(shape_type, points, SHAPE_ZH[shape_type], polygon_bbox(points), closed)


def path_points(shape: ShapeSample) -> list[Point]:
    points = list(shape.points)
    if shape.closed and points:
        points.append(points[0])
    return points


def choose_distance_id(
    remaining: float,
    buckets: tuple[float, ...],
    allowed_ids: list[int],
    min_remainder: float,
) -> int | None:
    if remaining < min_remainder:
        return None
    valid = [idx for idx in allowed_ids if buckets[idx] <= remaining + 1e-9]
    if valid:
        return max(valid)
    return min(allowed_ids, key=lambda idx: buckets[idx])


def append_segment(
    actions: list[dict],
    start: Point,
    end: Point,
    pen_state: str,
    tokenizer: PolarActionTokenizer,
    allowed_distance_ids: list[int],
    min_remainder: float,
    max_steps: int,
) -> Point | None:
    current = Point(start.x, start.y)
    best_remaining = math.inf
    stalled = 0
    min_allowed_distance = min(tokenizer.distance_buckets[idx] for idx in allowed_distance_ids)
    for _ in range(max_steps):
        vx = end.x - current.x
        vy = end.y - current.y
        remaining = math.hypot(vx, vy)
        distance_id = choose_distance_id(remaining, tokenizer.distance_buckets, allowed_distance_ids, min_remainder)
        if distance_id is None:
            return current
        if remaining > best_remaining - 1e-6:
            stalled += 1
        else:
            stalled = 0
            best_remaining = remaining
        if stalled >= 6:
            return None

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
        if remaining < min_allowed_distance:
            return current
    return None


def build_actions(
    shape: ShapeSample,
    tokenizer: PolarActionTokenizer,
    draw_distance_ids: list[int],
    move_distance_ids: list[int],
    min_remainder: float,
    max_move_steps: int,
    max_segment_steps: int,
) -> list[dict] | None:
    points = path_points(shape)
    if not points:
        return None

    actions: list[dict] = []
    cursor = append_segment(
        actions,
        Point(0.0, 0.0),
        points[0],
        "move",
        tokenizer,
        move_distance_ids,
        min_remainder,
        max_move_steps,
    )
    if cursor is None:
        return None

    for target in points[1:]:
        cursor = append_segment(
            actions,
            cursor,
            target,
            "draw",
            tokenizer,
            draw_distance_ids,
            min_remainder,
            max_segment_steps,
        )
        if cursor is None:
            return None

    if not actions:
        return None
    actions[-1]["pen_state"] = "end_all"
    return actions


def actions_to_strokes(actions: list[dict]) -> list[StrokeStep]:
    return [StrokeStep(dx=a["dx"], dy=a["dy"], pen_state=a["pen_state"]) for a in actions]


def actions_to_drawn_shape(shape_type: str, actions: list[dict], closed: bool) -> ShapeSample:
    x = 0.0
    y = 0.0
    drawn: list[Point] = []
    for action in actions:
        x += float(action["dx"])
        y += float(action["dy"])
        if action["pen_state"] != "move":
            drawn.append(Point(x, y))

    if closed and len(drawn) > 1:
        first = drawn[0]
        last = drawn[-1]
        if math.hypot(last.x - first.x, last.y - first.y) < 1e-6:
            drawn = drawn[:-1]

    if not drawn:
        drawn = [Point(x, y)]

    return ShapeSample(
        shape_type=shape_type,
        points=drawn,
        prompt_fragment=SHAPE_ZH[shape_type],
        bbox=polygon_bbox(drawn),
        closed=False,
    )


def make_prompt(shape_type: str, size_name: str, position_name: str, rng: random.Random) -> str:
    shape = SHAPE_ZH[shape_type]
    size = SIZE_ZH[size_name]
    position = POSITION_ZH[position_name]
    templates = [
        f"画一个{size}{shape}，位置在{position}",
        f"在{position}画一个{size}{shape}",
        f"请画一个{size}{shape}，放在{position}",
        f"生成一个{position}的{size}{shape}",
        f"把一个{size}{shape}画在{position}",
    ]
    return rng.choice(templates)


def make_sample(
    shape_type: str,
    size_name: str,
    position_name: str,
    rng: random.Random,
    tokenizer: PolarActionTokenizer,
    canvas_size: float,
    draw_distance_ids: list[int],
    move_distance_ids: list[int],
    min_remainder: float,
    max_move_steps: int,
    max_segment_steps: int,
) -> dict | None:
    shape = make_shape(shape_type, size_name, position_name, canvas_size, rng)
    actions = build_actions(
        shape,
        tokenizer,
        draw_distance_ids,
        move_distance_ids,
        min_remainder,
        max_move_steps,
        max_segment_steps,
    )
    if actions is None:
        return None

    token_shape = actions_to_drawn_shape(shape_type, actions, shape.closed)
    tokens = [tokenizer.encode_action(a["distance_id"], a["theta_id"], a["pen_state"]) for a in actions]
    strokes = actions_to_strokes(actions)
    prompt = make_prompt(shape_type, size_name, position_name, rng)
    scene_spec = {
        "scene_type": "single_basic_polar_balanced",
        "difficulty": "easy",
        "num_shapes": 1,
        "recipe_name": "polar_balanced_single_basic",
        "canvas_size": canvas_size,
        "distance_buckets": list(tokenizer.distance_buckets),
        "draw_distance_ids": draw_distance_ids,
        "move_distance_ids": move_distance_ids,
        "theta_bins": tokenizer.theta_bins,
        "min_remainder": min_remainder,
        "max_move_steps": max_move_steps,
    }
    metadata = annotate_metadata({**scene_spec, "scene_type": "single_basic"}, [token_shape], strokes)
    metadata.update(
        {
            "scene_type": scene_spec["scene_type"],
            "prompt_language": "zh",
            "shape_type": shape_type,
            "size_name": size_name,
            "position_name": position_name,
            "canvas_size": canvas_size,
            "distance_buckets": list(tokenizer.distance_buckets),
            "draw_distance_ids": draw_distance_ids,
            "move_distance_ids": move_distance_ids,
            "theta_bins": tokenizer.theta_bins,
            "min_remainder": min_remainder,
            "num_polar_actions": len(tokens),
            "num_move_actions": sum(1 for a in actions if a["pen_state"] == "move"),
            "num_draw_actions": sum(1 for a in actions if a["pen_state"] == "draw"),
        }
    )
    return {
        "scene_spec": scene_spec,
        "prompt": prompt,
        "shapes": [
            {
                "shape_type": shape.shape_type,
                "prompt_fragment": prompt,
                "points": [asdict(p) for p in token_shape.points],
                "bbox": token_shape.bbox,
                "closed": token_shape.closed,
            }
        ],
        "polar_actions": actions,
        "action_tokens": tokens,
        "strokes": [asdict(step) for step in strokes],
        "metadata": metadata,
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in text.split(",") if v.strip())


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def print_report(samples: list[dict], rejected: int) -> None:
    lengths = [len(s["action_tokens"]) for s in samples]
    move_lengths = [s["metadata"]["num_move_actions"] for s in samples]
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample["metadata"]["shape_type"]
        counts[key] = counts.get(key, 0) + 1
    print(f"samples={len(samples)} rejected={rejected}")
    print(f"shape_counts={dict(sorted(counts.items()))}")
    print(f"action_len min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"move_len min={min(move_lengths)} avg={sum(move_lengths)/len(move_lengths):.1f} max={max(move_lengths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced polar action token dataset with short move phase.")
    parser.add_argument("--samples-per-combo", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--distances", type=str, default="0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0")
    parser.add_argument("--draw-distance-ids", type=str, default="0,1,2,3,4")
    parser.add_argument("--move-distance-ids", type=str, default="4,5,6,7")
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--min-remainder", type=float, default=0.05)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--max-move-steps", type=int, default=8)
    parser.add_argument("--max-segment-steps", type=int, default=128)
    parser.add_argument("--max-attempts-per-sample", type=int, default=50)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--sizes", type=str, default="small,medium,large")
    parser.add_argument("--positions", type=str, default=",".join(POSITION_CENTERS.keys()))
    parser.add_argument("--output", type=str, default="generated_data/polar_balanced/polar_balanced_scale8_train.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shape_types = [s.strip() for s in args.shapes.split(",") if s.strip()]
    size_names = [s.strip() for s in args.sizes.split(",") if s.strip()]
    position_names = [s.strip() for s in args.positions.split(",") if s.strip()]
    tokenizer = PolarActionTokenizer(
        PolarActionTokenizerConfig(
            distance_buckets=parse_float_tuple(args.distances),
            theta_bins=args.theta_bins,
        )
    )
    draw_distance_ids = parse_int_list(args.draw_distance_ids)
    move_distance_ids = parse_int_list(args.move_distance_ids)

    samples: list[dict] = []
    rejected = 0
    combos = [(shape, size, pos) for shape in shape_types for size in size_names for pos in position_names]
    for shape_type, size_name, position_name in combos:
        made = 0
        attempts = 0
        while made < args.samples_per_combo:
            attempts += 1
            if attempts > args.samples_per_combo * args.max_attempts_per_sample:
                raise RuntimeError(f"too many rejected samples for {shape_type}/{size_name}/{position_name}")
            sample = make_sample(
                shape_type,
                size_name,
                position_name,
                rng,
                tokenizer,
                args.canvas_size,
                draw_distance_ids,
                move_distance_ids,
                args.min_remainder,
                args.max_move_steps,
                args.max_segment_steps,
            )
            if sample is None or len(sample["action_tokens"]) > args.max_action_len:
                rejected += 1
                continue
            samples.append(sample)
            made += 1

    rng.shuffle(samples)
    save_jsonl(samples, Path(args.output))
    print_report(samples, rejected)
    print(f"saved: {args.output}")
    print(f"train distances arg: --distances {args.distances}")


if __name__ == "__main__":
    main()
