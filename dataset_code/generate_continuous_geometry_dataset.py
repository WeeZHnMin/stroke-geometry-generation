from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_code"))

from stroke_data_factory.metadata_annotator import annotate_metadata
from stroke_data_factory.schema import Point, ShapeSample, StrokeStep
from stroke_data_factory.utils import polygon_bbox


DEFAULT_SHAPES = [
    "square",
    "rectangle",
    "wide_rectangle",
    "tall_rectangle",
    "quadrilateral",
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
    "quadrilateral": "四边形",
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


def rotate_point(p: Point, angle: float) -> Point:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return Point(p.x * ca - p.y * sa, p.x * sa + p.y * ca)


def translate_point(p: Point, cx: float, cy: float) -> Point:
    return Point(p.x + cx, p.y + cy)


def transform_points(points: list[Point], angle: float, cx: float, cy: float) -> list[Point]:
    return [translate_point(rotate_point(p, angle), cx, cy) for p in points]


def bbox_size(points: list[Point]) -> tuple[float, float]:
    bbox = polygon_bbox(points)
    return bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"]


def fit_transform(
    local_points: list[Point],
    angle: float,
    cx: float,
    cy: float,
    canvas_size: float,
    margin: float,
) -> tuple[list[Point], float]:
    transformed_at_origin = transform_points(local_points, angle, 0.0, 0.0)
    width, height = bbox_size(transformed_at_origin)
    allowed = max(canvas_size - 2.0 * margin, 0.5)
    scale = min(1.0, allowed / max(width, height, 1e-6))
    scaled = [Point(p.x * scale, p.y * scale) for p in local_points]
    transformed = transform_points(scaled, angle, cx, cy)

    bbox = polygon_bbox(transformed)
    shift_x = 0.0
    shift_y = 0.0
    if bbox["x_min"] < margin:
        shift_x = margin - bbox["x_min"]
    elif bbox["x_max"] > canvas_size - margin:
        shift_x = canvas_size - margin - bbox["x_max"]
    if bbox["y_min"] < margin:
        shift_y = margin - bbox["y_min"]
    elif bbox["y_max"] > canvas_size - margin:
        shift_y = canvas_size - margin - bbox["y_max"]

    return [Point(p.x + shift_x, p.y + shift_y) for p in transformed], scale


def make_local_shape(shape_type: str, span: float, rng: random.Random) -> tuple[list[Point], bool]:
    if shape_type == "square":
        half = span / 2.0
        return [Point(-half, -half), Point(half, -half), Point(half, half), Point(-half, half)], True
    if shape_type == "rectangle":
        half_w = span * 0.58
        half_h = span * 0.40
        return [Point(-half_w, -half_h), Point(half_w, -half_h), Point(half_w, half_h), Point(-half_w, half_h)], True
    if shape_type == "wide_rectangle":
        half_w = span * 0.72
        half_h = span * 0.30
        return [Point(-half_w, -half_h), Point(half_w, -half_h), Point(half_w, half_h), Point(-half_w, half_h)], True
    if shape_type == "tall_rectangle":
        half_w = span * 0.32
        half_h = span * 0.74
        return [Point(-half_w, -half_h), Point(half_w, -half_h), Point(half_w, half_h), Point(-half_w, half_h)], True
    if shape_type == "quadrilateral":
        half = span / 2.0
        jitter = span * 0.18
        return [
            Point(-half + rng.uniform(-jitter, jitter), -half + rng.uniform(-jitter, jitter)),
            Point(half + rng.uniform(-jitter, jitter), -half + rng.uniform(-jitter, jitter)),
            Point(half + rng.uniform(-jitter, jitter), half + rng.uniform(-jitter, jitter)),
            Point(-half + rng.uniform(-jitter, jitter), half + rng.uniform(-jitter, jitter)),
        ], True
    if shape_type == "triangle":
        radius = span * 0.68
        return [
            Point(0.0, -radius),
            Point(-radius * 0.92, radius * 0.55),
            Point(radius * 0.92, radius * 0.55),
        ], True
    if shape_type == "right_triangle":
        w = span * 1.15
        h = span * 1.10
        return [Point(-w / 2, h / 2), Point(w / 2, h / 2), Point(-w / 2, -h / 2)], True
    if shape_type == "circle":
        radius = span * 0.50
        return [Point(-radius, -radius), Point(radius, radius)], True
    if shape_type == "ellipse":
        rx = span * 0.62
        ry = span * 0.42
        return [Point(-rx, -ry), Point(rx, -ry), Point(rx, ry), Point(-rx, ry)], True
    if shape_type == "line":
        half = span * 0.62
        return [Point(-half, 0.0), Point(half, 0.0)], False
    raise ValueError(f"unsupported shape type: {shape_type}")


def sample_segment(a: Point, b: Point, step: float, include_end: bool = False) -> list[Point]:
    dist = math.hypot(b.x - a.x, b.y - a.y)
    n = max(1, int(math.ceil(dist / step)))
    pts = [Point(a.x + (b.x - a.x) * (i / n), a.y + (b.y - a.y) * (i / n)) for i in range(n)]
    if include_end:
        pts.append(Point(b.x, b.y))
    return pts


def sample_polyline(vertices: list[Point], closed: bool, step: float) -> list[Point]:
    if len(vertices) < 2:
        return list(vertices)
    pairs = list(zip(vertices, vertices[1:]))
    if closed:
        pairs.append((vertices[-1], vertices[0]))
    pts: list[Point] = []
    for idx, (a, b) in enumerate(pairs):
        include_end = (not closed) and idx == len(pairs) - 1
        pts.extend(sample_segment(a, b, step, include_end=include_end))
    if closed:
        pts.append(Point(vertices[0].x, vertices[0].y))
    return pts


def sample_circle(cx: float, cy: float, radius: float, step: float) -> list[Point]:
    circumference = 2.0 * math.pi * radius
    n = max(24, int(math.ceil(circumference / step)))
    pts = [
        Point(cx + radius * math.cos(2.0 * math.pi * i / n), cy + radius * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]
    pts.append(Point(pts[0].x, pts[0].y))
    return pts


def sample_ellipse(cx: float, cy: float, rx: float, ry: float, angle: float, step: float) -> list[Point]:
    h = ((rx - ry) ** 2) / max((rx + ry) ** 2, 1e-9)
    circumference = math.pi * (rx + ry) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
    n = max(32, int(math.ceil(circumference / step)))
    ca = math.cos(angle)
    sa = math.sin(angle)
    pts: list[Point] = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        pts.append(Point(cx + x * ca - y * sa, cy + x * sa + y * ca))
    pts.append(Point(pts[0].x, pts[0].y))
    return pts


def points_to_strokes(points: list[Point], move_step: float, max_delta: float) -> list[StrokeStep]:
    strokes: list[StrokeStep] = []
    x = 0.0
    y = 0.0
    if not points:
        return strokes

    start = points[0]
    n_move = max(1, int(math.ceil(math.hypot(start.x, start.y) / move_step)))
    for i in range(n_move):
        nx = start.x * (i + 1) / n_move
        ny = start.y * (i + 1) / n_move
        strokes.append(StrokeStep(nx - x, ny - y, "move"))
        x, y = nx, ny

    for idx, point in enumerate(points[1:], start=1):
        sx, sy = x, y
        dx = point.x - sx
        dy = point.y - sy
        chunks = max(1, int(math.ceil(max(abs(dx), abs(dy)) / max_delta)))
        for chunk_idx in range(chunks):
            frac = (chunk_idx + 1) / chunks
            nx = sx + dx * frac
            ny = sy + dy * frac
            is_last = idx == len(points) - 1 and chunk_idx == chunks - 1
            strokes.append(StrokeStep(nx - x, ny - y, "end" if is_last else "draw"))
            x, y = nx, ny
    return strokes


def accumulated_points(strokes: list[StrokeStep], include_move: bool = False) -> list[Point]:
    x = 0.0
    y = 0.0
    pts: list[Point] = []
    for stroke in strokes:
        x += stroke.dx
        y += stroke.dy
        if include_move or stroke.pen_state != "move":
            pts.append(Point(x, y))
    return pts


def max_abs_delta(strokes: list[StrokeStep]) -> float:
    return max((max(abs(s.dx), abs(s.dy)) for s in strokes), default=0.0)


def closure_error(points: list[Point], closed: bool) -> float | None:
    if not closed or len(points) < 2:
        return None
    return math.hypot(points[-1].x - points[0].x, points[-1].y - points[0].y)


def make_prompt(shape_type: str, size_name: str, position_name: str, rotated: bool, rng: random.Random) -> str:
    shape = SHAPE_ZH[shape_type]
    size = SIZE_ZH[size_name]
    position = POSITION_ZH[position_name]
    rotate_word = "旋转的" if rotated and shape_type not in {"circle", "line"} else ""
    templates = [
        f"画一个{position}的{size}{rotate_word}{shape}",
        f"在{position}画一个{size}{rotate_word}{shape}",
        f"请画一个{size}{rotate_word}{shape}，放在{position}",
        f"生成一个{position}的{size}{rotate_word}{shape}",
        f"把一个{size}{rotate_word}{shape}画在{position}",
    ]
    return rng.choice(templates)


def sample_shape_points(
    shape_type: str,
    span: float,
    angle: float,
    cx: float,
    cy: float,
    canvas_size: float,
    margin: float,
    sample_step: float,
    rng: random.Random,
) -> tuple[list[Point], bool]:
    local, closed = make_local_shape(shape_type, span, rng)
    if shape_type == "circle":
        fitted, scale = fit_transform(local, 0.0, cx, cy, canvas_size, margin)
        bbox = polygon_bbox(fitted)
        radius = span * 0.50 * scale
        return sample_circle((bbox["x_min"] + bbox["x_max"]) / 2.0, (bbox["y_min"] + bbox["y_max"]) / 2.0, radius, sample_step), True
    if shape_type == "ellipse":
        fitted, scale = fit_transform(local, angle, cx, cy, canvas_size, margin)
        bbox = polygon_bbox(fitted)
        return sample_ellipse(
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
            span * 0.62 * scale,
            span * 0.42 * scale,
            angle,
            sample_step,
        ), True
    vertices, _scale = fit_transform(local, angle, cx, cy, canvas_size, margin)
    return sample_polyline(vertices, closed, sample_step), closed


def make_sample(
    shape_type: str,
    size_name: str,
    position_name: str,
    rng: random.Random,
    canvas_size: float,
    sample_step: float,
    move_step: float,
    max_delta: float,
    rotate: bool,
) -> dict:
    span = SIZE_SPANS[size_name]
    cx_ratio, cy_ratio = POSITION_CENTERS[position_name]
    cx = cx_ratio * canvas_size + rng.uniform(-0.12, 0.12)
    cy = cy_ratio * canvas_size + rng.uniform(-0.12, 0.12)
    angle = rng.uniform(0.0, 2.0 * math.pi) if rotate and shape_type != "circle" else 0.0
    if shape_type == "line" and rotate:
        angle = rng.choice([0.0, math.pi / 4, math.pi / 2, math.pi * 3 / 4, -math.pi / 4])
    margin = max(0.35, sample_step * 2.0)

    sampled, closed = sample_shape_points(shape_type, span, angle, cx, cy, canvas_size, margin, sample_step, rng)
    strokes = points_to_strokes(sampled, move_step=move_step, max_delta=max_delta)
    traced_points = accumulated_points(strokes)
    shape_points = sampled[:-1] if closed and len(sampled) > 1 else sampled
    prompt = make_prompt(shape_type, size_name, position_name, rotate and abs(angle) > 1e-6, rng)
    bbox = polygon_bbox(shape_points)
    shape = ShapeSample(
        shape_type,
        shape_points,
        SHAPE_ZH[shape_type],
        bbox,
        closed,
    )

    scene_spec = {
        "scene_type": "continuous_geometry_single",
        "difficulty": "easy",
        "num_shapes": 1,
        "recipe_name": "continuous_geometry_single",
        "canvas_size": canvas_size,
        "sample_step": sample_step,
        "move_step": move_step,
        "max_delta": max_delta,
        "shape_type": shape_type,
        "size_name": size_name,
        "position_name": position_name,
        "rotation": angle,
    }
    metadata = annotate_metadata({**scene_spec, "scene_type": "single_basic"}, [shape], strokes)
    metadata.update(
        {
            "scene_type": scene_spec["scene_type"],
            "prompt_language": "zh",
            "shape_type": shape_type,
            "size_name": size_name,
            "position_name": position_name,
            "canvas_size": canvas_size,
            "sample_step": sample_step,
            "move_step": move_step,
            "max_delta": max_delta,
            "rotation": angle,
            "num_move_actions": sum(1 for s in strokes if s.pen_state == "move"),
            "num_draw_actions": sum(1 for s in strokes if s.pen_state == "draw"),
            "num_end_actions": sum(1 for s in strokes if s.pen_state == "end"),
            "max_abs_delta": max_abs_delta(strokes),
            "closure_error": closure_error(sampled, closed),
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
        "strokes": [asdict(step) for step in strokes],
        "metadata": metadata,
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_list(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def print_report(samples: list[dict], rejected: int) -> None:
    lengths = [len(s["strokes"]) for s in samples]
    max_abs = [s["metadata"]["max_abs_delta"] for s in samples]
    closures = [s["metadata"]["closure_error"] for s in samples if s["metadata"]["closure_error"] is not None]
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample["metadata"]["shape_type"]
        counts[key] = counts.get(key, 0) + 1
    print(f"samples={len(samples)}")
    print(f"shape_counts={dict(sorted(counts.items()))}")
    print(f"stroke_len min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"max_abs_delta max={max(max_abs):.6f}")
    if closures:
        print(f"closure_error max={max(closures):.8f} avg={sum(closures)/len(closures):.8f}")
    print(f"rejected={rejected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-precision continuous geometry strokes for GMM regression.")
    parser.add_argument("--samples-per-combo", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=8.0)
    parser.add_argument("--sample-step", type=float, default=0.1)
    parser.add_argument("--move-step", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--max-attempts-per-sample", type=int, default=50)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--sizes", type=str, default="small,medium,large")
    parser.add_argument("--positions", type=str, default=",".join(POSITION_CENTERS.keys()))
    parser.add_argument("--output", type=str, default="generated_data/continuous_geometry/continuous_geometry_scale8_train.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shape_types = parse_list(args.shapes)
    size_names = parse_list(args.sizes)
    position_names = parse_list(args.positions)
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
                args.canvas_size,
                args.sample_step,
                args.move_step,
                args.max_delta,
                args.rotate,
            )
            if len(sample["strokes"]) > args.max_action_len or sample["metadata"]["max_abs_delta"] > args.max_delta + 1e-6:
                rejected += 1
                continue
            samples.append(sample)
            made += 1

    rng.shuffle(samples)
    save_jsonl(samples, Path(args.output))
    print_report(samples, rejected)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
