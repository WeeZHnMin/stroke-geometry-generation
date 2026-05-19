"""Generate a focused dataset with 5000 samples per core shape type.

Core shapes: rectangle, circle, triangle variants, rhombus
Each shape is generated as a single-shape scene (no relations).

Usage:
    cd dataset_code
    python generate_core_shapes_dataset.py
    python generate_core_shapes_dataset.py --count 5000 --output ../generated_data/bulk/core_shapes.jsonl
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.stroke_compiler import compile_strokes
from stroke_data_factory.schema import StrokeStep


CORE_SHAPES = [
    # 矩形类
    "rectangle",
    "wide_rectangle",
    "tall_rectangle",
    "rotated_rectangle",
    # 圆形类
    "circle",
    "ellipse",
    "wide_ellipse",
    "tall_ellipse",
    # 三角形类（含角度分类）
    "triangle",
    "equilateral_triangle",
    "isosceles_triangle",
    "right_triangle",
    "acute_triangle",
    "obtuse_triangle",
    # 菱形及近亲
    "rhombus",
    "parallelogram",
    "kite",
    "trapezoid",
]


def steps_to_dict(steps: list[StrokeStep]) -> list[dict]:
    return [{"dx": s.dx, "dy": s.dy, "pen_state": s.pen_state} for s in steps]


def generate_samples(shape_type: str, count: int, rng: random.Random) -> list[dict]:
    samples = []
    for _ in range(count):
        shape = build_shape_from_type(shape_type, rng)
        steps = compile_strokes([shape])
        samples.append({
            "shape_type": shape_type,
            "prompt": shape.prompt_fragment,
            "strokes": steps_to_dict(steps),
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000, help="Samples per shape type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str,
        default=str(ROOT / "generated_data" / "bulk" / "core_shapes.jsonl"),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total = 0

    with output_path.open("w", encoding="utf-8") as f:
        for shape_type in CORE_SHAPES:
            samples = generate_samples(shape_type, args.count, rng)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total += len(samples)
            print(f"  {shape_type:30s}  {len(samples):>5} samples", flush=True)

    print(f"\ntotal: {total} samples  ->  {output_path}")


if __name__ == "__main__":
    main()
