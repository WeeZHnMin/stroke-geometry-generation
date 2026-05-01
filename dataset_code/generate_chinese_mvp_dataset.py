import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.metadata_annotator import annotate_metadata
from stroke_data_factory.stroke_compiler import compile_strokes


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "generated_data" / "chinese_mvp"


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


def size_zh(bbox: dict[str, float]) -> str:
    span = max(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"])
    if span < 0.18:
        return "小"
    if span < 0.30:
        return "中等"
    return "大"


def position_zh(bbox: dict[str, float]) -> str:
    cx = (bbox["x_min"] + bbox["x_max"]) / 2
    cy = (bbox["y_min"] + bbox["y_max"]) / 2
    horiz = "左边" if cx < 0.35 else "右边" if cx > 0.65 else "中间"
    vert = "上方" if cy < 0.35 else "下方" if cy > 0.65 else "中部"

    if horiz == "中间" and vert == "中部":
        return "画布中央"
    if horiz == "中间":
        return f"画布{vert}"
    if vert == "中部":
        return f"画布{horiz}"
    return f"画布{vert}{horiz}"


def make_chinese_prompt(shape_type: str, bbox: dict[str, float], rng: random.Random) -> str:
    shape = SHAPE_ZH[shape_type]
    size = size_zh(bbox)
    position = position_zh(bbox)
    templates = [
        f"画一个{size}{shape}，位置在{position}",
        f"在{position}画一个{size}{shape}",
        f"请画一个{size}{shape}，放在{position}",
    ]
    return rng.choice(templates)


def sample_chinese_scene(rng: random.Random, shape_types: list[str]) -> dict:
    shape_type = rng.choice(shape_types)
    shape = build_shape_from_type(shape_type, rng)
    strokes = compile_strokes([shape])
    scene_spec = {
        "scene_type": "single_basic_chinese_mvp",
        "difficulty": "easy",
        "num_shapes": 1,
        "allowed_shapes": shape_types,
        "relation_type": None,
        "anchor_position": None,
        "recipe_name": "chinese_mvp_single_basic",
    }
    prompt = make_chinese_prompt(shape.shape_type, shape.bbox, rng)
    metadata = annotate_metadata(
        {**scene_spec, "scene_type": "single_basic"},
        [shape],
        strokes,
    )
    metadata["scene_type"] = scene_spec["scene_type"]
    metadata["prompt_language"] = "zh"
    metadata["mvp_family"] = "single_basic_convergence"

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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a clean Chinese MVP stroke dataset.")
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    args = parser.parse_args()

    shape_types = [s.strip() for s in args.shapes.split(",") if s.strip()]
    unknown = [s for s in shape_types if s not in SHAPE_ZH]
    if unknown:
        raise ValueError(f"unsupported shapes for Chinese MVP: {unknown}")

    rng = random.Random(args.seed)
    samples = [sample_chinese_scene(rng, shape_types) for _ in range(args.num_samples)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else OUTPUT_DIR / f"chinese_mvp_single_basic_{timestamp}.jsonl"
    save_jsonl(samples, output)

    print(f"wrote {len(samples)} samples to {output}")
    print("shapes:", ", ".join(shape_types))
    print("example prompt:", samples[0]["prompt"])
    print("example metadata:", samples[0]["metadata"])


if __name__ == "__main__":
    main()
