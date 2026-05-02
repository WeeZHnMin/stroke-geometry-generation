"""
generate_verified_dataset.py
============================
正确、可靠的中文笔画数据集生成器。

数据流水线:
  scene spec → normalized geometry → canvas-scale geometry → dense resampling → stroke steps → text + metadata

核心保证:
  1. 图形坐标位于更大的画布尺度, 默认 [0, 6] x [0, 6] ✓
  2. draw 步位移由 dense step 控制, 默认 ≈ 0.1 (一点点连续画) ✓
  3. move 步也按 dense step 抬笔移动到起点, 不再大跳跃 ✓
  4. pen_state 序列合法 ✓
  5. 输出可验证、可视化 ✓

用法:
  python dataset_code/generate_verified_dataset.py
  python dataset_code/generate_verified_dataset.py --num-samples 10000 --seed 42
  python dataset_code/generate_verified_dataset.py --verify-only path/to/file.jsonl
"""

import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── 路径设置 ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "dataset_code"))

from stroke_data_factory.geometry_builder import build_shape_from_type
from stroke_data_factory.metadata_annotator import annotate_metadata
from stroke_data_factory.schema import Point, ShapeSample, StrokeStep
from stroke_data_factory.stroke_compiler import compile_strokes
from stroke_data_factory.utils import polygon_bbox

# ── 形状配置 ─────────────────────────────────────────────────────
DEFAULT_SHAPES = [
    "square", "rectangle", "wide_rectangle", "tall_rectangle",
    "triangle", "right_triangle",
    "circle", "ellipse",
    "line",
]

SHAPE_ZH = {
    "square": "正方形", "rectangle": "矩形", "wide_rectangle": "宽矩形",
    "tall_rectangle": "高矩形", "triangle": "三角形", "right_triangle": "直角三角形",
    "circle": "圆", "ellipse": "椭圆", "line": "线段",
}


# ── 中文 prompt 生成 ─────────────────────────────────────────────
def size_zh(bbox: Dict[str, float], canvas_size: float) -> str:
    span = max(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"])
    norm_span = span / canvas_size
    if norm_span < 0.18:
        return "小"
    if norm_span < 0.30:
        return "中等"
    return "大"


def position_zh(bbox: Dict[str, float], canvas_size: float) -> str:
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


def make_chinese_prompt(shape_type: str, bbox: Dict[str, float], rng: random.Random, canvas_size: float) -> str:
    shape = SHAPE_ZH[shape_type]
    size = size_zh(bbox, canvas_size)
    position = position_zh(bbox, canvas_size)
    templates = [
        f"画一个{size}{shape}，位置在{position}",
        f"在{position}画一个{size}{shape}",
        f"请画一个{size}{shape}，放在{position}",
    ]
    return rng.choice(templates)


# ── 单样本生成 ───────────────────────────────────────────────────
def scale_shape_to_canvas(shape: ShapeSample, canvas_size: float) -> ShapeSample:
    points = [Point(p.x * canvas_size, p.y * canvas_size) for p in shape.points]
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=polygon_bbox(points),
        closed=shape.closed,
    )


def sample_chinese_scene(
    rng: random.Random,
    shape_types: List[str],
    canvas_size: float = 6.0,
    dense_step: float = 0.1,
) -> Dict[str, Any]:
    """生成一个单图元样本，确保所有 dx/dy 正确。"""
    shape_type = rng.choice(shape_types)
    normalized_shape = build_shape_from_type(shape_type, rng)
    shape = scale_shape_to_canvas(normalized_shape, canvas_size)

    # 核心: 在大尺度画布上沿轨迹密采样, 让 draw dx/dy 成为可控小增量。
    strokes = compile_strokes([shape], dense_step=dense_step)

    scene_spec = {
        "scene_type": "single_basic_chinese_mvp",
        "difficulty": "easy",
        "num_shapes": 1,
        "allowed_shapes": shape_types,
        "relation_type": None,
        "anchor_position": None,
        "recipe_name": "chinese_mvp_single_basic",
        "canvas_size": canvas_size,
        "dense_step": dense_step,
    }
    prompt = make_chinese_prompt(shape.shape_type, shape.bbox, rng, canvas_size)
    metadata = annotate_metadata(
        {**scene_spec, "scene_type": "single_basic"},
        [shape],
        strokes,
    )
    metadata["scene_type"] = scene_spec["scene_type"]
    metadata["prompt_language"] = "zh"
    metadata["mvp_family"] = "single_basic_convergence"
    metadata["canvas_size"] = canvas_size
    metadata["dense_step"] = dense_step

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


# ── 质量验证 ─────────────────────────────────────────────────────
def verify_sample(sample: Dict[str, Any], index: int) -> List[str]:
    """验证单条样本的 dx/dy 合法性，返回所有问题列表。"""
    issues: List[str] = []
    strokes = sample["strokes"]

    if not strokes:
        issues.append(f"[#{index}] 空笔画序列")
        return issues

    dense_step = float(sample.get("metadata", {}).get("dense_step", sample.get("scene_spec", {}).get("dense_step", 0.1)))

    # 1. 所有 action-token 位移都应该是小步, 包括抬笔 move。
    for i, s in enumerate(strokes):
        dx, dy = s["dx"], s["dy"]
        limit = 0.5
        if not (-limit <= dx <= limit):
            issues.append(f"[#{index}] 步{i}: dx={dx:.4f} 超出 [-{limit}, {limit}]")
        if not (-limit <= dy <= limit):
            issues.append(f"[#{index}] 步{i}: dy={dy:.4f} 超出 [-{limit}, {limit}]")

    # 2. 检查 move/draw/end 是否都是连续小步。
    local_steps = [s for s in strokes if s["pen_state"] in {"move", "draw", "end_shape", "end_all"}]
    if local_steps:
        max_draw_delta = max(max(abs(s["dx"]), abs(s["dy"])) for s in local_steps)
        allowed = min(0.5, dense_step * 1.5)
        if max_draw_delta > allowed:
            issues.append(f"[#{index}] 局部步最大位移={max_draw_delta:.4f} > {allowed:.4f} (可能未按 dense_step 重采样)")

    # 3. 检查最后一步是 end_all
    if strokes[-1]["pen_state"] != "end_all":
        issues.append(f"[#{index}] 最后一步不是 end_all")

    # 4. 检查 pen_state 序列合法性
    valid_pen = {"move", "draw", "end_shape", "end_all"}
    for i, s in enumerate(strokes):
        if s["pen_state"] not in valid_pen:
            issues.append(f"[#{index}] 步{i}: 非法 pen_state={s['pen_state']}")

    return issues


def compute_statistics(strokes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算单条样本的统计信息。"""
    if not strokes:
        return {}
    dxs = [s["dx"] for s in strokes]
    dys = [s["dy"] for s in strokes]
    move_steps = [s for s in strokes if s["pen_state"] == "move"]
    draw_steps = [s for s in strokes if s["pen_state"] == "draw"]

    draw_dxs = [abs(s["dx"]) for s in draw_steps] if draw_steps else [0.0]
    draw_dys = [abs(s["dy"]) for s in draw_steps] if draw_steps else [0.0]

    return {
        "n_steps": len(strokes),
        "n_draw": len(draw_steps),
        "dx_min": min(dxs),
        "dx_max": max(dxs),
        "dy_min": min(dys),
        "dy_max": max(dys),
        "draw_dx_mean": sum(draw_dxs) / len(draw_dxs) if draw_dxs else 0.0,
        "draw_dy_mean": sum(draw_dys) / len(draw_dys) if draw_dys else 0.0,
        "draw_dx_max": max(draw_dxs),
        "draw_dy_max": max(draw_dys),
    }


def print_verification_report(samples: List[Dict[str, Any]], title: str = "验证报告"):
    """打印数据集质量验证报告。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    all_issues: List[str] = []
    stats_list: List[Dict[str, Any]] = []
    shape_type_counts: Dict[str, int] = {}

    for i, sample in enumerate(samples):
        issues = verify_sample(sample, i)
        all_issues.extend(issues)
        stats = compute_statistics(sample["strokes"])
        stats_list.append(stats)

        shape_type = sample["shapes"][0]["shape_type"]
        shape_type_counts[shape_type] = shape_type_counts.get(shape_type, 0) + 1

    n_samples = len(samples)
    print(f"\n样本总数: {n_samples}")
    print(f"形状分布: {dict(sorted(shape_type_counts.items()))}")

    # 总体 dx/dy 范围
    all_dx = [s["dx"] for sample in samples for s in sample["strokes"]]
    all_dy = [s["dy"] for sample in samples for s in sample["strokes"]]
    print(f"\n所有 dx  范围: [{min(all_dx):.4f}, {max(all_dx):.4f}]")
    print(f"所有 dy  范围: [{min(all_dy):.4f}, {max(all_dy):.4f}]")

    # draw 步统计
    draw_dxs = [abs(s["dx"]) for sample in samples for s in sample["strokes"] if s["pen_state"] == "draw"]
    draw_dys = [abs(s["dy"]) for sample in samples for s in sample["strokes"] if s["pen_state"] == "draw"]
    if draw_dxs:
        print(f"\n--- draw 步统计 ---")
        print(f"draw |dx| 均值: {sum(draw_dxs)/len(draw_dxs):.4f}")
        print(f"draw |dx| 最大: {max(draw_dxs):.4f}")
        print(f"draw |dy| 均值: {sum(draw_dys)/len(draw_dys):.4f}")
        print(f"draw |dy| 最大: {max(draw_dys):.4f}")

    # 步数分布
    step_counts = [s["n_steps"] for s in stats_list]
    draw_counts = [s["n_draw"] for s in stats_list]
    if step_counts:
        print(f"\n笔画步数:   min={min(step_counts)} max={max(step_counts)} avg={sum(step_counts)/len(step_counts):.0f}")
        print(f"draw步数:   min={min(draw_counts)} max={max(draw_counts)} avg={sum(draw_counts)/len(draw_counts):.0f}")

    # 问题汇总
    print(f"\n问题总数: {len(all_issues)}")
    if all_issues:
        print("前 10 个问题:")
        for issue in all_issues[:10]:
            print(f"  WARN {issue}")
    else:
        print("OK 无问题 - 数据质量合格!")

    print(f"{'='*60}\n")
    return len(all_issues)


# ── 文件读写 ─────────────────────────────────────────────────────
def save_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ── 主入口 ───────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="生成中文笔画数据集 (带质量验证)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_code/generate_verified_dataset.py
  python dataset_code/generate_verified_dataset.py --num-samples 10000 --seed 42
  python dataset_code/generate_verified_dataset.py --verify-only generated_data/chinese_mvp/xxx.jsonl
        """,
    )
    parser.add_argument("--num-samples", type=int, default=6000, help="生成样本数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES), help="形状列表")
    parser.add_argument("--canvas-size", type=float, default=6.0, help="画布尺度, 默认坐标范围约为 [0, 6]")
    parser.add_argument("--dense-step", type=float, default=0.1, help="轨迹密采样步长, 默认 draw 位移约为 0.1")
    parser.add_argument("--verify-only", type=str, default=None, help="仅验证已有文件")
    args = parser.parse_args()

    # ── 仅验证模式 ──
    if args.verify_only:
        samples = load_jsonl(Path(args.verify_only))
        print_verification_report(samples, title=f"验证: {args.verify_only}")
        return

    # ── 生成模式 ──
    shape_types = [s.strip() for s in args.shapes.split(",") if s.strip()]
    unknown = [s for s in shape_types if s not in SHAPE_ZH]
    if unknown:
        raise ValueError(f"不支持的形状: {unknown}")

    print(f"生成 {args.num_samples} 个样本...")
    print(f"形状池: {shape_types}")
    print(f"种子: {args.seed}")
    print(f"画布尺度: {args.canvas_size}")
    print(f"采样步长: {args.dense_step}")

    rng = random.Random(args.seed)
    samples = [
        sample_chinese_scene(
            rng,
            shape_types,
            canvas_size=args.canvas_size,
            dense_step=args.dense_step,
        )
        for _ in range(args.num_samples)
    ]

    # ── 验证 ──
    n_issues = print_verification_report(samples, title="生成数据质量报告")

    # ── 保存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "generated_data" / "chinese_mvp"
    output_path = Path(args.output) if args.output else output_dir / f"chinese_mvp_single_basic_scale{args.canvas_size:g}_step{args.dense_step:g}_{timestamp}.jsonl"
    save_jsonl(samples, output_path)

    print(f"已保存: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    if n_issues > 0:
        print(f"\nWARN 发现 {n_issues} 个问题，请检查!")
    else:
        print("\nOK 数据集生成成功，所有样本通过质量检查!")


if __name__ == "__main__":
    main()
