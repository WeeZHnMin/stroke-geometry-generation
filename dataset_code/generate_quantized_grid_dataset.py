from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.metadata_annotator import annotate_metadata
from stroke_data_factory.schema import Point, ShapeSample, StrokeStep
from stroke_data_factory.utils import polygon_bbox, resample_dense


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "generated_data" / "quantized_grid"

GRID_SIZE = 0.01
MAX_DELTA = 0.5
MAX_DELTA_UNITS = int(round(MAX_DELTA / GRID_SIZE))

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

SHAPE_EN = {
    "square": "square",
    "rectangle": "rectangle",
    "wide_rectangle": "wide rectangle",
    "tall_rectangle": "tall rectangle",
    "triangle": "triangle",
    "right_triangle": "right triangle",
    "circle": "circle",
    "ellipse": "ellipse",
    "line": "line",
}

SHAPE_ZH = {
    "square": "正方形",
    "rectangle": "矩形",
    "wide_rectangle": "宽矩形",
    "tall_rectangle": "高矩形",
    "triangle": "三角形",
    "right_triangle": "直角三角形",
    "circle": "圆形",
    "ellipse": "椭圆",
    "line": "线段",
}


def pen_state_to_label(pen_state: str) -> int:
    if pen_state == "move":
        return 0
    if pen_state == "draw":
        return 1
    if pen_state == "end_all":
        return 2
    raise ValueError(f"unsupported pen_state: {pen_state}")


def quantize_to_grid(value: float) -> float:
    return round(value / GRID_SIZE) * GRID_SIZE


def quantize_point(point: Point, canvas_size: float) -> Point:
    x = min(max(quantize_to_grid(point.x), 0.0), canvas_size)
    y = min(max(quantize_to_grid(point.y), 0.0), canvas_size)
    return Point(x=x, y=y)


def scale_shape_to_canvas(shape: ShapeSample, canvas_size: float) -> ShapeSample:
    points = [Point(p.x * canvas_size, p.y * canvas_size) for p in shape.points]
    quantized = [quantize_point(p, canvas_size) for p in points]
    return ShapeSample(
        shape_type=shape.shape_type,
        points=quantized,
        prompt_fragment=shape.prompt_fragment,
        bbox=polygon_bbox(quantized),
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
    horiz = "左侧" if cx < 0.35 else "右侧" if cx > 0.65 else "中间"
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
    templates = [
        f"画一个位于{position}的{size}{shape}",
        f"在{position}画一个{size}{shape}",
        f"请画一个{size}{shape}，位置在{position}",
    ]
    return rng.choice(templates)


def point_to_units(point: Point) -> tuple[int, int]:
    return int(round(point.x / GRID_SIZE)), int(round(point.y / GRID_SIZE))


def units_to_point(x_units: int, y_units: int) -> Point:
    return Point(x=x_units * GRID_SIZE, y=y_units * GRID_SIZE)


def interpolate_units(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    steps = max(1, math.ceil(max(abs(dx), abs(dy)) / MAX_DELTA_UNITS))
    out: list[tuple[int, int]] = []
    prev_x, prev_y = start
    for i in range(1, steps + 1):
        next_x = round(start[0] + dx * i / steps)
        next_y = round(start[1] + dy * i / steps)
        step_dx = next_x - prev_x
        step_dy = next_y - prev_y
        if abs(step_dx) > MAX_DELTA_UNITS or abs(step_dy) > MAX_DELTA_UNITS:
            raise ValueError("step exceeds configured max delta after interpolation")
        out.append((next_x, next_y))
        prev_x, prev_y = next_x, next_y
    return out


def append_segment_steps(
    strokes: list[StrokeStep],
    cursor_units: tuple[int, int],
    target_units: tuple[int, int],
    *,
    pen_state: str,
) -> tuple[int, int]:
    points = interpolate_units(cursor_units, target_units)
    prev_x, prev_y = cursor_units
    for next_x, next_y in points:
        strokes.append(
            StrokeStep(
                dx=(next_x - prev_x) * GRID_SIZE,
                dy=(next_y - prev_y) * GRID_SIZE,
                pen_state=pen_state,
            )
        )
        prev_x, prev_y = next_x, next_y
    return prev_x, prev_y


def quantized_path_points(shape: ShapeSample, dense_step: float, canvas_size: float) -> list[Point]:
    dense = resample_dense(shape.points, shape.closed, step=dense_step)
    quantized: list[Point] = []
    for point in dense:
        qp = quantize_point(point, canvas_size)
        if not quantized or qp.x != quantized[-1].x or qp.y != quantized[-1].y:
            quantized.append(qp)

    if shape.closed and quantized:
        first = quantized[0]
        last = quantized[-1]
        if first.x != last.x or first.y != last.y:
            quantized.append(first)
    return quantized


def compile_quantized_strokes(shape: ShapeSample, dense_step: float, canvas_size: float) -> list[StrokeStep]:
    points = quantized_path_points(shape, dense_step=dense_step, canvas_size=canvas_size)
    if len(points) < 2:
        raise ValueError("shape path is too short after quantization")

    strokes: list[StrokeStep] = []
    cursor_units = (0, 0)
    cursor_units = append_segment_steps(strokes, cursor_units, point_to_units(points[0]), pen_state="move")
    for point in points[1:]:
        cursor_units = append_segment_steps(strokes, cursor_units, point_to_units(point), pen_state="draw")
    strokes[-1].pen_state = "end_all"
    return strokes


def verify_strokes(strokes: list[StrokeStep]) -> None:
    if not strokes:
        raise ValueError("empty stroke sequence")
    if strokes[-1].pen_state != "end_all":
        raise ValueError("last pen_state must be end_all")
    valid_pen = {"move", "draw", "end_all"}
    for step in strokes:
        if step.pen_state not in valid_pen:
            raise ValueError(f"invalid pen_state: {step.pen_state}")
        if step.dx < -MAX_DELTA or step.dx > MAX_DELTA:
            raise ValueError(f"dx out of range: {step.dx}")
        if step.dy < -MAX_DELTA or step.dy > MAX_DELTA:
            raise ValueError(f"dy out of range: {step.dy}")
        if abs(step.dx / GRID_SIZE - round(step.dx / GRID_SIZE)) > 1e-6:
            raise ValueError(f"dx is not on the {GRID_SIZE} grid: {step.dx}")
        if abs(step.dy / GRID_SIZE - round(step.dy / GRID_SIZE)) > 1e-6:
            raise ValueError(f"dy is not on the {GRID_SIZE} grid: {step.dy}")


def build_continuous_sequence(strokes: list[StrokeStep]) -> list[dict[str, float | int]]:
    sequence: list[dict[str, float | int]] = []
    x = 0.0
    y = 0.0
    for step in strokes:
        x += step.dx
        y += step.dy
        sequence.append(
            {
                "x": x,
                "y": y,
                "pen_id": pen_state_to_label(step.pen_state),
                "pen_state": step.pen_state,
            }
        )
    return sequence


def sample_scene(
    rng: random.Random,
    shape_types: list[str],
    *,
    canvas_size: float,
    dense_step: float,
) -> dict:
    shape_type = rng.choice(shape_types)
    base_shape = build_shape_from_type(shape_type, rng)
    shape = scale_shape_to_canvas(base_shape, canvas_size)
    strokes = compile_quantized_strokes(shape, dense_step=dense_step, canvas_size=canvas_size)
    verify_strokes(strokes)
    continuous_sequence = build_continuous_sequence(strokes)

    scene_spec = {
        "scene_type": "single_basic_quantized_grid",
        "difficulty": "easy",
        "num_shapes": 1,
        "allowed_shapes": shape_types,
        "relation_type": None,
        "anchor_position": None,
        "recipe_name": "quantized_grid_single_basic",
        "canvas_size": canvas_size,
        "dense_step": dense_step,
        "grid_size": GRID_SIZE,
        "max_delta": MAX_DELTA,
        "pen_states": ["move", "draw", "end_all"],
        "step_format": ["x", "y", "pen_id"],
    }
    prompt = make_prompt(shape.shape_type, shape.bbox, rng, canvas_size)
    metadata = annotate_metadata({**scene_spec, "scene_type": "single_basic"}, [shape], strokes)
    metadata.update(
        {
            "scene_type": scene_spec["scene_type"],
            "prompt_language": "zh",
            "grid_size": GRID_SIZE,
            "max_delta": MAX_DELTA,
            "pen_states": scene_spec["pen_states"],
            "num_move_actions": sum(1 for step in strokes if step.pen_state == "move"),
            "num_draw_actions": sum(1 for step in strokes if step.pen_state == "draw"),
            "num_end_all_actions": sum(1 for step in strokes if step.pen_state == "end_all"),
            "continuous_step_dim": 3,
        }
    )
    return {
        "scene_spec": scene_spec,
        "prompt": prompt,
        "shapes": [
            {
                "shape_type": shape.shape_type,
                "prompt_fragment": prompt,
                "points": [asdict(point) for point in shape.points],
                "bbox": shape.bbox,
                "closed": shape.closed,
            }
        ],
        "strokes": [asdict(step) for step in strokes],
        "continuous_sequence": continuous_sequence,
        "metadata": metadata,
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_report(samples: list[dict]) -> None:
    lengths = [len(sample["strokes"]) for sample in samples]
    shape_counts: dict[str, int] = {}
    for sample in samples:
        shape_type = sample["shapes"][0]["shape_type"]
        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
    print(f"samples={len(samples)}")
    print(f"shape_counts={dict(sorted(shape_counts.items()))}")
    print(f"stroke_len min={min(lengths)} avg={sum(lengths) / len(lengths):.1f} max={max(lengths)}")
    print(f"grid_size={GRID_SIZE} max_delta={MAX_DELTA}")
    print(f"example_prompt={samples[0]['prompt']}")
    print(f"example_strokes={samples[0]['strokes'][:5]}")
    print(f"example_continuous_sequence={samples[0]['continuous_sequence'][:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quantized single-shape stroke data with dx/dy on a 0.01 grid in [-0.5, 0.5]."
    )
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--canvas-size", type=float, default=6.0)
    parser.add_argument("--dense-step", type=float, default=0.08)
    args = parser.parse_args()

    shape_types = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]
    unknown = [shape for shape in shape_types if shape not in SHAPE_EN]
    if unknown:
        raise ValueError(f"unsupported shapes: {unknown}")

    rng = random.Random(args.seed)
    samples = [
        sample_scene(
            rng,
            shape_types,
            canvas_size=args.canvas_size,
            dense_step=args.dense_step,
        )
        for _ in range(args.num_samples)
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output)
        if args.output
        else OUTPUT_DIR / f"quantized_grid_single_basic_{timestamp}.jsonl"
    )
    save_jsonl(samples, output)
    print_report(samples)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
