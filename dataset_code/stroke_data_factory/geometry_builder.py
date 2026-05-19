import math
import random
from typing import Dict, List

from .schema import Point, ShapeSample
from .utils import bbox_center, clamp, polygon_bbox, rotate_point, scale_shape, translate_shape


def describe_position(cx: float, cy: float) -> str:
    horiz = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    vert = "top" if cy < 0.38 else "bottom" if cy > 0.62 else "middle"
    _pos_map = {
        ("center", "middle"): "位于中心",
        ("center", "top"):    "位于上方",
        ("center", "bottom"): "位于下方",
        ("left",   "middle"): "位于左侧",
        ("right",  "middle"): "位于右侧",
        ("left",   "top"):    "位于左上角",
        ("right",  "top"):    "位于右上角",
        ("left",   "bottom"): "位于左下角",
        ("right",  "bottom"): "位于右下角",
    }
    return _pos_map.get((horiz, vert), "位于中心")


def describe_size(scale: float) -> str:
    if scale < 0.18:
        return "小"
    if scale < 0.3:
        return "中等"
    return "大"


def _sanitize_shape(shape: ShapeSample) -> ShapeSample:
    points = [Point(clamp(p.x), clamp(p.y)) for p in shape.points]
    bbox = polygon_bbox(points)
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=bbox,
        closed=shape.closed,
    )


ANCHOR_TARGETS = {
    "top_left": Point(0.22, 0.22),
    "top_right": Point(0.78, 0.22),
    "bottom_left": Point(0.22, 0.78),
    "bottom_right": Point(0.78, 0.78),
    "center": Point(0.50, 0.50),
    "top": Point(0.50, 0.20),
    "bottom": Point(0.50, 0.80),
    "left": Point(0.20, 0.50),
    "right": Point(0.80, 0.50),
    "upper_left": Point(0.32, 0.24),
    "upper_right": Point(0.68, 0.24),
    "lower_left": Point(0.32, 0.76),
    "lower_right": Point(0.68, 0.76),
    "center_left": Point(0.28, 0.50),
    "center_right": Point(0.72, 0.50),
}


def apply_anchor_layout(shape: ShapeSample, anchor_position: str | None) -> ShapeSample:
    if not anchor_position or anchor_position not in ANCHOR_TARGETS:
        return shape
    center = bbox_center(shape.bbox)
    target = ANCHOR_TARGETS[anchor_position]
    laid_out = translate_shape(shape, target.x - center.x, target.y - center.y)
    return _sanitize_shape(laid_out)


def _line_shape(points: List[Point], shape_type: str, prompt: str) -> ShapeSample:
    return ShapeSample(shape_type, points, prompt, polygon_bbox(points), closed=False)


def _anchor_point(name: str) -> Point:
    return ANCHOR_TARGETS.get(name, Point(0.5, 0.5))


def _function_points_from_callable(fn, x_start: float, x_end: float, n: int) -> List[Point]:
    xs = [x_start + (x_end - x_start) * i / max(n - 1, 1) for i in range(n)]
    ys = [fn(x) for x in xs]
    y_min = min(ys)
    y_max = max(ys)
    y_span = max(y_max - y_min, 1e-6)
    points = []
    for x, y in zip(xs, ys):
        px = 0.1 + 0.8 * ((x - x_start) / max(x_end - x_start, 1e-6))
        py = 0.15 + 0.7 * ((y - y_min) / y_span)
        points.append(Point(clamp(px), clamp(py)))
    return points


def build_contrast_pair_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    family = scene_spec["contrast_family"]
    family_map = {
        "rectangle": ["rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle"],
        "triangle": ["triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle", "acute_triangle", "obtuse_triangle"],
        "square": ["square", "rotated_square"],
        "ellipse": ["circle", "ellipse", "wide_ellipse", "tall_ellipse"],
    }
    candidates = family_map[family]
    t1 = rng.choice(candidates)
    t2 = rng.choice([t for t in candidates if t != t1] or candidates)
    shapes = [build_shape_from_type(t1, rng), build_shape_from_type(t2, rng)]
    return apply_relation_layout(shapes, "separate", rng)


def build_equation_function_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    spec = scene_spec["equation_spec"]
    if spec["family"] == "linear":
        m, b = spec["m"], spec["b"]
        points = _function_points_from_callable(lambda x: m * x + b, -3.0, 3.0, rng.randint(18, 28))
        shape = ShapeSample("linear_function", points, "a linear function graph", polygon_bbox(points), closed=False)
        return [_sanitize_shape(shape)]
    if spec["family"] == "quadratic":
        a, b, c = spec["a"], spec["b"], spec["c"]
        points = _function_points_from_callable(lambda x: a * x * x + b * x + c, -3.0, 3.0, rng.randint(22, 34))
        shape = ShapeSample("parabola", points, "a quadratic function graph", polygon_bbox(points), closed=False)
        return [_sanitize_shape(shape)]
    a, b, c, d = spec["a"], spec["b"], spec["c"], spec["d"]
    points = _function_points_from_callable(lambda x: a * x**3 + b * x**2 + c * x + d, -2.5, 2.5, rng.randint(22, 34))
    shape = ShapeSample("cubic_function", points, "a cubic function graph", polygon_bbox(points), closed=False)
    return [_sanitize_shape(shape)]


def build_triple_composition_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    family = scene_spec["triple_family"]
    if family == "rect_tri_square":
        shape_types = ["rectangle", "triangle", "square"]
    elif family == "circles_and_triangle":
        shape_types = ["circle", "wide_ellipse", "triangle"]
    else:
        pool = ["rectangle", "triangle", "square", "circle", "ellipse", "pentagon", "hexagon"]
        shape_types = rng.sample(pool, 3)
    shapes = [build_shape_from_type(t, rng) for t in shape_types]
    targets = [Point(0.22, 0.30), Point(0.50, 0.72), Point(0.78, 0.34)]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_compound_relation_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    shape_a = build_shape_from_type(rng.choice(["triangle", "rectangle", "ellipse", "square"]), rng)
    shape_b = build_shape_from_type(rng.choice(["circle", "rectangle", "pentagon", "wide_ellipse"]), rng)
    relation = scene_spec["compound_relation"]
    target_b = Point(0.62, 0.62)
    offsets = {
        "above_left_of": Point(-0.26, -0.24),
        "above_right_of": Point(0.26, -0.24),
        "below_left_of": Point(-0.26, 0.24),
        "below_right_of": Point(0.26, 0.24),
    }
    target_a = Point(target_b.x + offsets[relation].x, target_b.y + offsets[relation].y)
    center_a = bbox_center(shape_a.bbox)
    center_b = bbox_center(shape_b.bbox)
    shape_a = _sanitize_shape(translate_shape(shape_a, target_a.x - center_a.x, target_a.y - center_a.y))
    shape_b = _sanitize_shape(translate_shape(shape_b, target_b.x - center_b.x, target_b.y - center_b.y))
    return [shape_a, shape_b]


def build_constraint_curve_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    start = _anchor_point(scene_spec["start_anchor"])
    end = _anchor_point(scene_spec["end_anchor"])
    constraint = scene_spec["constraint_type"]
    if constraint == "through_center":
        control = Point(0.50, 0.50)
    elif constraint == "touch_top_edge":
        control = Point(rng.uniform(0.35, 0.65), 0.04)
    else:
        control = Point(rng.uniform(0.35, 0.65), 0.96)

    points: List[Point] = []
    if scene_spec["curve_style"] == "s_bend":
        control2 = Point(clamp((start.x + end.x) / 2 + rng.uniform(-0.12, 0.12)), clamp((start.y + end.y) / 2 + rng.uniform(-0.18, 0.18)))
        n = rng.randint(14, 20)
        for i in range(n):
            t = i / max(n - 1, 1)
            x = (1 - t) ** 3 * start.x + 3 * (1 - t) ** 2 * t * control.x + 3 * (1 - t) * t * t * control2.x + t ** 3 * end.x
            y = (1 - t) ** 3 * start.y + 3 * (1 - t) ** 2 * t * control.y + 3 * (1 - t) * t * t * control2.y + t ** 3 * end.y
            points.append(Point(clamp(x), clamp(y)))
    else:
        n = rng.randint(12, 18)
        for i in range(n):
            t = i / max(n - 1, 1)
            x = (1 - t) * (1 - t) * start.x + 2 * (1 - t) * t * control.x + t * t * end.x
            y = (1 - t) * (1 - t) * start.y + 2 * (1 - t) * t * control.y + t * t * end.y
            points.append(Point(clamp(x), clamp(y)))
    shape = ShapeSample("guided_curve", points, "a constrained curve", polygon_bbox(points), closed=False)
    return [_sanitize_shape(shape)]


def build_counting_group_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    family = scene_spec["count_family"]
    type_map = {
        "circles": ["circle", "wide_ellipse", "tall_ellipse"],
        "squares": ["square", "rotated_square"],
        "rectangles": ["rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle"],
    }
    shape_types = [rng.choice(type_map[family]) for _ in range(scene_spec["num_shapes"])]
    shapes = [build_shape_from_type(t, rng) for t in shape_types]
    mode = scene_spec["alignment_mode"]
    targets: List[Point]
    if mode == "column":
        xs = [0.50] * scene_spec["num_shapes"]
        ys = [0.22 + i * (0.56 / max(scene_spec["num_shapes"] - 1, 1)) for i in range(scene_spec["num_shapes"])]
        targets = [Point(x, y) for x, y in zip(xs, ys)]
    elif mode == "grid" and scene_spec["num_shapes"] == 4:
        targets = [Point(0.34, 0.34), Point(0.66, 0.34), Point(0.34, 0.66), Point(0.66, 0.66)]
    else:
        xs = [0.18 + i * (0.64 / max(scene_spec["num_shapes"] - 1, 1)) for i in range(scene_spec["num_shapes"])]
        ys = [0.50] * scene_spec["num_shapes"]
        targets = [Point(x, y) for x, y in zip(xs, ys)]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_alignment_group_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    n = scene_spec["num_shapes"]
    shape_types = [rng.choice(["circle", "square", "rectangle", "triangle"]) for _ in range(n)]
    shapes = [build_shape_from_type(t, rng) for t in shape_types]
    mode = scene_spec["alignment_mode"]
    if mode == "aligned_horizontally":
        targets = [Point(0.20 + i * (0.60 / max(n - 1, 1)), 0.50) for i in range(n)]
    elif mode == "aligned_vertically":
        targets = [Point(0.50, 0.20 + i * (0.60 / max(n - 1, 1))) for i in range(n)]
    else:
        if n == 2:
            targets = [Point(0.32, 0.50), Point(0.68, 0.50)]
        elif n == 3:
            targets = [Point(0.25, 0.50), Point(0.50, 0.50), Point(0.75, 0.50)]
        else:
            targets = [Point(0.22, 0.50), Point(0.40, 0.50), Point(0.60, 0.50), Point(0.78, 0.50)]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_edge_contact_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    shape = build_shape_from_type(rng.choice(["circle", "ellipse", "rectangle", "square", "triangle"]), rng)
    mode = scene_spec["edge_mode"]
    bbox = shape.bbox
    center = bbox_center(bbox)
    width = bbox["x_max"] - bbox["x_min"]
    height = bbox["y_max"] - bbox["y_min"]
    if mode == "touch_left_edge":
        target = Point(width / 2, 0.50)
    elif mode == "touch_right_edge":
        target = Point(1 - width / 2, 0.50)
    elif mode == "touch_top_edge":
        target = Point(0.50, height / 2)
    elif mode == "touch_bottom_edge":
        target = Point(0.50, 1 - height / 2)
    else:
        target = Point(1 - width / 2, height / 2)
    moved = _sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y))
    return [moved]


def build_relative_size_pair_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    family = rng.choice(["circle", "ellipse", "rectangle", "triangle", "square"])
    family_map = {
        "circle": ["circle", "wide_ellipse", "tall_ellipse"],
        "ellipse": ["ellipse", "wide_ellipse", "tall_ellipse"],
        "rectangle": ["rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle"],
        "triangle": ["triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle", "acute_triangle", "obtuse_triangle"],
        "square": ["square", "rotated_square"],
    }
    shapes = [build_shape_from_type(rng.choice(family_map[family]), rng), build_shape_from_type(rng.choice(family_map[family]), rng)]
    if scene_spec["size_relation"] == "larger_than":
        shapes[0] = scale_shape(shapes[0], 1.35, 1.35)
        shapes[1] = scale_shape(shapes[1], 0.72, 0.72)
    else:
        shapes[0] = scale_shape(shapes[0], 0.72, 0.72)
        shapes[1] = scale_shape(shapes[1], 1.35, 1.35)
    shapes = [_sanitize_shape(s) for s in shapes]
    return apply_relation_layout(shapes, scene_spec["relation_type"], rng)


def build_same_diff_pair_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    if scene_spec["sameness_mode"] == "same_shape":
        family = rng.choice(["rectangle", "triangle", "circle", "square"])
        family_map = {
            "rectangle": ["rectangle", "wide_rectangle", "tall_rectangle"],
            "triangle": ["triangle", "isosceles_triangle", "right_triangle"],
            "circle": ["circle", "ellipse"],
            "square": ["square", "rotated_square"],
        }
        t = rng.choice(family_map[family])
        shapes = [build_shape_from_type(t, rng), build_shape_from_type(t, rng)]
        shapes[1] = scale_shape(shapes[1], rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15))
    else:
        pool = ["rectangle", "triangle", "circle", "square", "pentagon", "hexagon"]
        t1, t2 = rng.sample(pool, 2)
        shapes = [build_shape_from_type(t1, rng), build_shape_from_type(t2, rng)]
    shapes = [_sanitize_shape(s) for s in shapes]
    return apply_relation_layout(shapes, scene_spec["relation_type"], rng)


def build_through_points_curve_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    start = _anchor_point(scene_spec["start_anchor"])
    end = _anchor_point(scene_spec["end_anchor"])
    pc = scene_spec["point_constraint"]
    through_map = {
        "through_center": Point(0.50, 0.50),
        "through_top_right": Point(0.82, 0.18),
        "through_bottom_left": Point(0.18, 0.82),
    }
    mid = through_map[pc]
    points: List[Point] = []
    n = rng.randint(16, 22)
    for i in range(n):
        t = i / max(n - 1, 1)
        x = (1 - t) * (1 - t) * start.x + 2 * (1 - t) * t * mid.x + t * t * end.x
        y = (1 - t) * (1 - t) * start.y + 2 * (1 - t) * t * mid.y + t * t * end.y
        points.append(Point(clamp(x), clamp(y)))
    return [_sanitize_shape(ShapeSample("guided_curve", points, "a through-points curve", polygon_bbox(points), closed=False))]


def build_region_occupancy_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    family = scene_spec["count_family"]
    if family == "circles":
        pool = ["circle", "wide_ellipse", "tall_ellipse"]
    elif family == "squares":
        pool = ["square", "rotated_square"]
    else:
        pool = ["circle", "square", "triangle", "rectangle"]
    shapes = [build_shape_from_type(rng.choice(pool), rng) for _ in range(scene_spec["num_shapes"])]
    region = scene_spec["occupancy_region"]
    targets: List[Point] = []
    if region == "upper_half":
        ys = [0.20 + (i % 2) * 0.16 for i in range(scene_spec["num_shapes"])]
        xs = [0.25 + (i // 2) * 0.25 if scene_spec["num_shapes"] > 2 else 0.30 + i * 0.28 for i in range(scene_spec["num_shapes"])]
    elif region == "lower_half":
        ys = [0.66 + (i % 2) * 0.12 for i in range(scene_spec["num_shapes"])]
        xs = [0.25 + (i // 2) * 0.25 if scene_spec["num_shapes"] > 2 else 0.30 + i * 0.28 for i in range(scene_spec["num_shapes"])]
    elif region == "left_half":
        xs = [0.20 + (i % 2) * 0.14 for i in range(scene_spec["num_shapes"])]
        ys = [0.30 + (i // 2) * 0.22 if scene_spec["num_shapes"] > 2 else 0.32 + i * 0.22 for i in range(scene_spec["num_shapes"])]
    else:
        xs = [0.66 + (i % 2) * 0.10 for i in range(scene_spec["num_shapes"])]
        ys = [0.30 + (i // 2) * 0.22 if scene_spec["num_shapes"] > 2 else 0.32 + i * 0.22 for i in range(scene_spec["num_shapes"])]
    targets = [Point(x, y) for x, y in zip(xs, ys)]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_step_instruction_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    plan = scene_spec["step_plan"]
    if plan == "circle_then_rectangle_right":
        shape1 = build_shape_from_type("circle", rng)
        shape2 = build_shape_from_type(rng.choice(["rectangle", "wide_rectangle", "tall_rectangle"]), rng)
        targets = [Point(0.30, 0.50), Point(0.72, 0.50)]
    elif plan == "square_then_triangle_below":
        shape1 = build_shape_from_type(rng.choice(["square", "rotated_square"]), rng)
        shape2 = build_shape_from_type(rng.choice(["triangle", "isosceles_triangle", "right_triangle"]), rng)
        targets = [Point(0.50, 0.28), Point(0.50, 0.74)]
    else:
        shape1 = build_shape_from_type(rng.choice(["ellipse", "wide_ellipse", "tall_ellipse"]), rng)
        shape2 = build_shape_from_type("line", rng)
        targets = [Point(0.50, 0.64), Point(0.50, 0.24)]
    shapes = [shape1, shape2]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_negation_constraint_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    mode = scene_spec["negation_mode"]
    if mode == "two_non_intersecting_circles":
        shapes = [build_shape_from_type(rng.choice(["circle", "wide_ellipse", "tall_ellipse"]), rng), build_shape_from_type(rng.choice(["circle", "wide_ellipse", "tall_ellipse"]), rng)]
        return apply_relation_layout(shapes, "separate", rng)
    if mode == "two_shapes_not_touching_boundary":
        shapes = [
            build_shape_from_type(rng.choice(["rectangle", "triangle", "circle"]), rng),
            build_shape_from_type(rng.choice(["square", "ellipse", "pentagon"]), rng),
        ]
        targets = [Point(0.35, 0.50), Point(0.68, 0.50)]
        out = []
        for shape, target in zip(shapes, targets):
            center = bbox_center(shape.bbox)
            out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
        return out
    shapes = [
        build_shape_from_type(rng.choice(["rectangle", "triangle", "circle"]), rng),
        build_shape_from_type(rng.choice(["square", "ellipse", "hexagon"]), rng),
    ]
    return apply_relation_layout(shapes, "separate", rng)


def build_style_precision_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    target = scene_spec["style_target"]
    if target == "nearly_square_rectangle":
        shape = build_shape_from_type("rectangle", rng)
        shape = scale_shape(shape, 1.02, 0.95)
    elif target == "very_flat_ellipse":
        shape = build_shape_from_type("wide_ellipse", rng)
        shape = scale_shape(shape, 1.15, 0.72)
    else:
        shape = build_shape_from_type("tall_rectangle", rng)
        shape = scale_shape(shape, 0.85, 1.18)
    return [apply_anchor_layout(_sanitize_shape(shape), rng.choice(["center", "top", "bottom", "left", "right"]))]


def build_logic_conjunction_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    mode = scene_spec["conjunction_mode"]
    shapes = [
        build_shape_from_type(rng.choice(["circle", "ellipse", "rectangle", "triangle"]), rng),
        build_shape_from_type(rng.choice(["circle", "square", "ellipse", "pentagon"]), rng),
    ]
    if mode == "non_intersecting_upper_half":
        targets = [Point(0.34, 0.26), Point(0.68, 0.28)]
    else:
        targets = [Point(0.66, 0.36), Point(0.76, 0.68)]
    out = []
    for shape, target in zip(shapes, targets):
        center = bbox_center(shape.bbox)
        out.append(_sanitize_shape(translate_shape(shape, target.x - center.x, target.y - center.y)))
    return out


def build_reference_layout_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    target_name = scene_spec["reference_target"]
    anchor_map = {
        "top_middle": "top",
        "bottom_middle": "bottom",
        "center_left": "center_left",
        "center_right": "center_right",
    }
    shape = build_shape_from_type(rng.choice(["triangle", "circle", "square", "rectangle", "ellipse"]), rng)
    return [apply_anchor_layout(shape, anchor_map[target_name])]


def build_spatial_instruction_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    instruction_type = scene_spec["instruction_type"]
    if instruction_type == "parallel_lines":
        horizontal = rng.choice([True, False])
        if horizontal:
            y1 = rng.uniform(0.30, 0.40)
            y2 = y1 + rng.uniform(0.14, 0.22)
            x0 = rng.uniform(0.12, 0.24)
            x1 = rng.uniform(0.72, 0.88)
            shapes = [
                _line_shape([Point(x0, y1), Point(x1, y1)], "line", "a horizontal line"),
                _line_shape([Point(x0, y2), Point(x1, y2)], "line", "another parallel horizontal line"),
            ]
        else:
            x1 = rng.uniform(0.30, 0.40)
            x2 = x1 + rng.uniform(0.14, 0.22)
            y0 = rng.uniform(0.12, 0.24)
            y1 = rng.uniform(0.72, 0.88)
            shapes = [
                _line_shape([Point(x1, y0), Point(x1, y1)], "line", "a vertical line"),
                _line_shape([Point(x2, y0), Point(x2, y1)], "line", "another parallel vertical line"),
            ]
        return [_sanitize_shape(shape) for shape in shapes]

    if instruction_type == "perpendicular_lines":
        cx = rng.uniform(0.38, 0.62)
        cy = rng.uniform(0.38, 0.62)
        half_long = rng.uniform(0.18, 0.28)
        half_short = rng.uniform(0.12, 0.20)
        horizontal = _line_shape([Point(cx - half_long, cy), Point(cx + half_long, cy)], "line", "a horizontal line")
        vertical = _line_shape([Point(cx, cy - half_short), Point(cx, cy + half_short)], "line", "a vertical line")
        return [_sanitize_shape(horizontal), _sanitize_shape(vertical)]

    if instruction_type == "intersecting_lines":
        cx = rng.uniform(0.38, 0.62)
        cy = rng.uniform(0.38, 0.62)
        r1 = rng.uniform(0.18, 0.26)
        r2 = rng.uniform(0.16, 0.24)
        angle1 = rng.uniform(-math.pi / 6, math.pi / 6)
        angle2 = angle1 + rng.uniform(math.pi / 4, 3 * math.pi / 4)
        line1 = _line_shape(
            [Point(cx - r1 * math.cos(angle1), cy - r1 * math.sin(angle1)), Point(cx + r1 * math.cos(angle1), cy + r1 * math.sin(angle1))],
            "line",
            "a line through the center",
        )
        line2 = _line_shape(
            [Point(cx - r2 * math.cos(angle2), cy - r2 * math.sin(angle2)), Point(cx + r2 * math.cos(angle2), cy + r2 * math.sin(angle2))],
            "line",
            "another intersecting line",
        )
        return [_sanitize_shape(line1), _sanitize_shape(line2)]

    if instruction_type == "directed_line":
        start = _anchor_point(scene_spec["start_anchor"])
        end = _anchor_point(scene_spec["end_anchor"])
        jitter = lambda: rng.uniform(-0.03, 0.03)
        line = _line_shape(
            [Point(clamp(start.x + jitter()), clamp(start.y + jitter())), Point(clamp(end.x + jitter()), clamp(end.y + jitter()))],
            "line",
            "a directed line",
        )
        return [_sanitize_shape(line)]

    start = _anchor_point(scene_spec["start_anchor"])
    end = _anchor_point(scene_spec["end_anchor"])
    mid_x = (start.x + end.x) / 2
    mid_y = (start.y + end.y) / 2
    curve_style = scene_spec["curve_style"]
    if curve_style == "curving_upward":
        control = Point(mid_x, clamp(min(start.y, end.y) - rng.uniform(0.14, 0.24)))
    elif curve_style == "curving_downward":
        control = Point(mid_x, clamp(max(start.y, end.y) + rng.uniform(0.14, 0.24)))
    else:
        control = Point(clamp(mid_x + rng.uniform(-0.10, 0.10)), clamp(mid_y + rng.uniform(-0.18, 0.18)))

    points: List[Point] = []
    n = rng.randint(10, 16)
    for i in range(n):
        t = i / max(n - 1, 1)
        x = (1 - t) * (1 - t) * start.x + 2 * (1 - t) * t * control.x + t * t * end.x
        y = (1 - t) * (1 - t) * start.y + 2 * (1 - t) * t * control.y + t * t * end.y
        points.append(Point(clamp(x), clamp(y)))
    curve = _line_shape(points, "guided_curve", "a guided curve")
    return [_sanitize_shape(curve)]


def _rectangle_shape(w: float, h: float, cx: float, cy: float, angle: float, shape_type: str, prompt_name: str) -> ShapeSample:
    corners = [
        Point(cx - w / 2, cy - h / 2),
        Point(cx + w / 2, cy - h / 2),
        Point(cx + w / 2, cy + h / 2),
        Point(cx - w / 2, cy + h / 2),
    ]
    if angle != 0.0:
        corners = [rotate_point(p.x, p.y, cx, cy, angle) for p in corners]
    pos = describe_position(cx, cy)
    size = describe_size(max(w, h))
    prompt = f"一个{size}{prompt_name}，{pos}"
    return ShapeSample(shape_type, corners, prompt, polygon_bbox(corners), closed=True)


def sample_line(rng: random.Random) -> ShapeSample:
    x0 = rng.uniform(0.1, 0.9)
    y0 = rng.uniform(0.1, 0.9)
    length = rng.uniform(0.15, 0.35)
    angle = rng.uniform(0, 2 * math.pi)
    x1 = clamp(x0 + length * math.cos(angle))
    y1 = clamp(y0 + length * math.sin(angle))
    points = [Point(x0, y0), Point(x1, y1)]

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    prompt = f"一条{describe_size(length)}线段，{describe_position(cx, cy)}"
    return ShapeSample("line", points, prompt, polygon_bbox(points), closed=False)


def sample_polyline(rng: random.Random) -> ShapeSample:
    start = Point(rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.85))
    num_segments = rng.randint(2, 4)
    points = [start]
    heading = rng.uniform(-math.pi / 4, math.pi / 4)

    for _ in range(num_segments):
        heading += rng.uniform(-math.pi / 2, math.pi / 2)
        step = rng.uniform(0.08, 0.18)
        last = points[-1]
        new_x = clamp(last.x + step * math.cos(heading))
        new_y = clamp(last.y + step * math.sin(heading))
        points.append(Point(new_x, new_y))

    bbox = polygon_bbox(points)
    cx = (bbox["x_min"] + bbox["x_max"]) / 2
    cy = (bbox["y_min"] + bbox["y_max"]) / 2
    span = max(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"])
    prompt = f"一条{describe_size(span)}折线，{describe_position(cx, cy)}"
    return ShapeSample("polyline", points, prompt, bbox, closed=False)


def sample_irregular_quad(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    base_w = rng.uniform(0.18, 0.32)
    base_h = rng.uniform(0.14, 0.28)
    angle = rng.uniform(0, 2 * math.pi)

    raw = [
        Point(cx - base_w / 2 + rng.uniform(-0.03, 0.03), cy - base_h / 2 + rng.uniform(-0.03, 0.03)),
        Point(cx + base_w / 2 + rng.uniform(-0.03, 0.03), cy - base_h / 2 + rng.uniform(-0.03, 0.03)),
        Point(cx + base_w / 2 + rng.uniform(-0.03, 0.03), cy + base_h / 2 + rng.uniform(-0.03, 0.03)),
        Point(cx - base_w / 2 + rng.uniform(-0.03, 0.03), cy + base_h / 2 + rng.uniform(-0.03, 0.03)),
    ]
    points = [rotate_point(p.x, p.y, cx, cy, angle) for p in raw]

    bbox = polygon_bbox(points)
    span = max(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"])
    prompt = f"一个{describe_size(span)}不规则四边形，{describe_position(cx, cy)}"
    return ShapeSample("irregular_quad", points, prompt, bbox, closed=True)


def sample_trapezoid(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    top_w = rng.uniform(0.10, 0.22)
    bottom_w = rng.uniform(0.18, 0.32)
    h = rng.uniform(0.14, 0.24)
    pts = [
        Point(cx - top_w / 2, cy - h / 2),
        Point(cx + top_w / 2, cy - h / 2),
        Point(cx + bottom_w / 2, cy + h / 2),
        Point(cx - bottom_w / 2, cy + h / 2),
    ]
    angle = rng.uniform(-math.pi / 6, math.pi / 6)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(bottom_w, h))}梯形，{describe_position(cx, cy)}"
    return ShapeSample("trapezoid", pts, prompt, polygon_bbox(pts), closed=True)


def sample_rhombus(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    diag_w = rng.uniform(0.16, 0.28)
    diag_h = rng.uniform(0.12, 0.24)
    pts = [
        Point(cx, cy - diag_h / 2),
        Point(cx + diag_w / 2, cy),
        Point(cx, cy + diag_h / 2),
        Point(cx - diag_w / 2, cy),
    ]
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(diag_w, diag_h))}菱形，{describe_position(cx, cy)}"
    return ShapeSample("rhombus", pts, prompt, polygon_bbox(pts), closed=True)


def sample_parallelogram(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    w = rng.uniform(0.18, 0.30)
    h = rng.uniform(0.12, 0.22)
    shear = rng.uniform(0.04, 0.10)
    pts = [
        Point(cx - w / 2 + shear, cy - h / 2),
        Point(cx + w / 2 + shear, cy - h / 2),
        Point(cx + w / 2 - shear, cy + h / 2),
        Point(cx - w / 2 - shear, cy + h / 2),
    ]
    angle = rng.uniform(-math.pi / 6, math.pi / 6)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(w, h))}平行四边形，{describe_position(cx, cy)}"
    return ShapeSample("parallelogram", pts, prompt, polygon_bbox(pts), closed=True)


def sample_kite(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    top = rng.uniform(0.10, 0.18)
    bottom = rng.uniform(0.14, 0.24)
    side_w = rng.uniform(0.10, 0.18)
    pts = [
        Point(cx, cy - top),
        Point(cx + side_w, cy),
        Point(cx, cy + bottom),
        Point(cx - side_w, cy),
    ]
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(top + bottom, side_w * 2))}筝形，{describe_position(cx, cy)}"
    return ShapeSample("kite", pts, prompt, polygon_bbox(pts), closed=True)


def sample_square(rng: random.Random) -> ShapeSample:
    side = rng.uniform(0.14, 0.28)
    cx = rng.uniform(side / 2 + 0.06, 1 - side / 2 - 0.06)
    cy = rng.uniform(side / 2 + 0.06, 1 - side / 2 - 0.06)
    return _rectangle_shape(side, side, cx, cy, 0.0, "square", "正方形")


def sample_rotated_square(rng: random.Random) -> ShapeSample:
    side = rng.uniform(0.14, 0.26)
    cx = rng.uniform(side / 2 + 0.10, 1 - side / 2 - 0.10)
    cy = rng.uniform(side / 2 + 0.10, 1 - side / 2 - 0.10)
    angle = rng.uniform(math.pi / 8, math.pi / 3)
    return _rectangle_shape(side, side, cx, cy, angle, "rotated_square", "旋转正方形")


def sample_rectangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.15, 0.35)
    h = rng.uniform(0.12, 0.30)
    cx = rng.uniform(w / 2 + 0.05, 1 - w / 2 - 0.05)
    cy = rng.uniform(h / 2 + 0.05, 1 - h / 2 - 0.05)
    angle = rng.choice([0.0, 0.0, 0.0, math.pi / 12, -math.pi / 12])
    return _rectangle_shape(w, h, cx, cy, angle, "rectangle", "矩形")


def sample_wide_rectangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.24, 0.38)
    h = rng.uniform(0.10, 0.16)
    cx = rng.uniform(w / 2 + 0.06, 1 - w / 2 - 0.06)
    cy = rng.uniform(h / 2 + 0.06, 1 - h / 2 - 0.06)
    return _rectangle_shape(w, h, cx, cy, 0.0, "wide_rectangle", "宽矩形")


def sample_tall_rectangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.10, 0.16)
    h = rng.uniform(0.24, 0.38)
    cx = rng.uniform(w / 2 + 0.06, 1 - w / 2 - 0.06)
    cy = rng.uniform(h / 2 + 0.06, 1 - h / 2 - 0.06)
    return _rectangle_shape(w, h, cx, cy, 0.0, "tall_rectangle", "高矩形")


def sample_rotated_rectangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.18, 0.34)
    h = rng.uniform(0.12, 0.24)
    cx = rng.uniform(w / 2 + 0.12, 1 - w / 2 - 0.12)
    cy = rng.uniform(h / 2 + 0.12, 1 - h / 2 - 0.12)
    angle = rng.choice([math.pi / 6, -math.pi / 6, math.pi / 4, -math.pi / 4])
    return _rectangle_shape(w, h, cx, cy, angle, "rotated_rectangle", "旋转矩形")


def sample_triangle(rng: random.Random) -> ShapeSample:
    cx = rng.uniform(0.2, 0.8)
    cy = rng.uniform(0.2, 0.8)
    radius = rng.uniform(0.12, 0.22)
    base_angle = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(3):
        a = base_angle + i * 2 * math.pi / 3 + rng.uniform(-0.25, 0.25)
        r = radius * rng.uniform(0.8, 1.1)
        pts.append(Point(clamp(cx + r * math.cos(a)), clamp(cy + r * math.sin(a))))

    pos = describe_position(cx, cy)
    size = describe_size(radius * 2)
    prompt = f"一个{size}三角形，{pos}"
    return ShapeSample("triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_equilateral_triangle(rng: random.Random) -> ShapeSample:
    side = rng.uniform(0.18, 0.30)
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    radius = side / math.sqrt(3)
    phase = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(3):
        t = phase + i * 2 * math.pi / 3
        pts.append(Point(clamp(cx + radius * math.cos(t)), clamp(cy + radius * math.sin(t))))
    prompt = f"一个{describe_size(side)}等边三角形，{describe_position(cx, cy)}"
    return ShapeSample("equilateral_triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_isosceles_triangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.18, 0.32)
    h = rng.uniform(0.18, 0.30)
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    pts = [
        Point(cx, cy - h / 2),
        Point(cx + w / 2, cy + h / 2),
        Point(cx - w / 2, cy + h / 2),
    ]
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(w, h))}等腰三角形，{describe_position(cx, cy)}"
    return ShapeSample("isosceles_triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_right_triangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.16, 0.28)
    h = rng.uniform(0.16, 0.28)
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    pts = [
        Point(cx - w / 2, cy - h / 2),
        Point(cx + w / 2, cy - h / 2),
        Point(cx - w / 2, cy + h / 2),
    ]
    angle = rng.uniform(-math.pi / 3, math.pi / 3)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    prompt = f"一个{describe_size(max(w, h))}直角三角形，{describe_position(cx, cy)}"
    return ShapeSample("right_triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_acute_triangle(rng: random.Random) -> ShapeSample:
    """All three angles < 90°: ensures apex height > half base width."""
    w = rng.uniform(0.18, 0.30)
    h = rng.uniform(w * 0.6, w * 1.2)   # h > w/2 guarantees acute apex angle
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    offset = rng.uniform(-w * 0.15, w * 0.15)
    pts = [
        Point(cx - w / 2, cy + h / 2),
        Point(cx + w / 2, cy + h / 2),
        Point(cx + offset, cy - h / 2),
    ]
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    pts = [Point(clamp(p.x), clamp(p.y)) for p in pts]
    prompt = f"一个{describe_size(max(w, h))}锐角三角形，{describe_position(cx, cy)}"
    return ShapeSample("acute_triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_obtuse_triangle(rng: random.Random) -> ShapeSample:
    """One angle > 90°: either flat (obtuse apex) or offset apex (obtuse base angle)."""
    cx = rng.uniform(0.25, 0.75)
    cy = rng.uniform(0.25, 0.75)
    method = rng.choice(["flat", "offset"])
    if method == "flat":
        w = rng.uniform(0.22, 0.34)
        h = rng.uniform(w * 0.15, w * 0.38)   # h < w/2: apex angle obtuse
        offset = rng.uniform(-w * 0.1, w * 0.1)
    else:
        w = rng.uniform(0.18, 0.28)
        h = rng.uniform(w * 0.4, w * 0.9)
        # |offset| > w/2 puts apex past a base vertex → that base angle becomes obtuse
        offset = rng.uniform(w * 0.55, w * 0.80) * rng.choice([-1, 1])
    pts = [
        Point(cx - w / 2, cy + h / 2),
        Point(cx + w / 2, cy + h / 2),
        Point(cx + offset, cy - h / 2),
    ]
    angle = rng.uniform(-math.pi / 4, math.pi / 4)
    pts = [rotate_point(p.x, p.y, cx, cy, angle) for p in pts]
    pts = [Point(clamp(p.x), clamp(p.y)) for p in pts]
    prompt = f"一个{describe_size(max(w, h))}钝角三角形，{describe_position(cx, cy)}"
    return ShapeSample("obtuse_triangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_circle(rng: random.Random) -> ShapeSample:
    r = rng.uniform(0.08, 0.18)
    cx = rng.uniform(r + 0.05, 1 - r - 0.05)
    cy = rng.uniform(r + 0.05, 1 - r - 0.05)
    n = rng.randint(18, 32)
    pts = []
    phase = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        t = phase + 2 * math.pi * i / n
        pts.append(Point(cx + r * math.cos(t), cy + r * math.sin(t)))

    pos = describe_position(cx, cy)
    size = describe_size(r * 2)
    prompt = f"一个{size}圆形，{pos}"
    return ShapeSample("circle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_ellipse(rng: random.Random) -> ShapeSample:
    rx = rng.uniform(0.08, 0.18)
    ry = rng.uniform(0.05, 0.14)
    cx = rng.uniform(rx + 0.05, 1 - rx - 0.05)
    cy = rng.uniform(ry + 0.05, 1 - ry - 0.05)
    n = rng.randint(20, 36)
    angle = rng.uniform(0, 2 * math.pi)
    pts = []

    for i in range(n):
        t = 2 * math.pi * i / n
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)
        pts.append(rotate_point(x, y, cx, cy, angle))

    pos = describe_position(cx, cy)
    size = describe_size(max(rx, ry) * 2)
    prompt = f"一个{size}椭圆，{pos}"
    return ShapeSample("ellipse", pts, prompt, polygon_bbox(pts), closed=True)


def sample_wide_ellipse(rng: random.Random) -> ShapeSample:
    rx = rng.uniform(0.12, 0.20)
    ry = rng.uniform(0.04, 0.09)
    cx = rng.uniform(rx + 0.08, 1 - rx - 0.08)
    cy = rng.uniform(ry + 0.08, 1 - ry - 0.08)
    n = rng.randint(20, 36)
    angle = rng.uniform(-math.pi / 8, math.pi / 8)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)
        pts.append(rotate_point(x, y, cx, cy, angle))
    prompt = f"一个{describe_size(rx * 2)}宽椭圆，{describe_position(cx, cy)}"
    return ShapeSample("wide_ellipse", pts, prompt, polygon_bbox(pts), closed=True)


def sample_tall_ellipse(rng: random.Random) -> ShapeSample:
    rx = rng.uniform(0.04, 0.09)
    ry = rng.uniform(0.12, 0.20)
    cx = rng.uniform(rx + 0.08, 1 - rx - 0.08)
    cy = rng.uniform(ry + 0.08, 1 - ry - 0.08)
    n = rng.randint(20, 36)
    angle = rng.uniform(-math.pi / 8, math.pi / 8)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)
        pts.append(rotate_point(x, y, cx, cy, angle))
    prompt = f"一个{describe_size(ry * 2)}高椭圆，{describe_position(cx, cy)}"
    return ShapeSample("tall_ellipse", pts, prompt, polygon_bbox(pts), closed=True)


def sample_arc(rng: random.Random) -> ShapeSample:
    r = rng.uniform(0.10, 0.18)
    cx = rng.uniform(r + 0.05, 1 - r - 0.05)
    cy = rng.uniform(r + 0.05, 1 - r - 0.05)
    start = rng.uniform(0, 2 * math.pi)
    sweep = rng.uniform(math.pi / 4, math.pi)
    n = rng.randint(8, 16)
    pts = []

    for i in range(n):
        t = start + sweep * i / max(n - 1, 1)
        pts.append(Point(cx + r * math.cos(t), cy + r * math.sin(t)))

    pos = describe_position(cx, cy)
    prompt = f"一段{describe_size(r * 2)}弧线，{pos}"
    return ShapeSample("arc", pts, prompt, polygon_bbox(pts), closed=False)


def sample_notched_rectangle(rng: random.Random) -> ShapeSample:
    x0 = rng.uniform(0.15, 0.25)
    y0 = rng.uniform(0.15, 0.25)
    w = rng.uniform(0.28, 0.42)
    h = rng.uniform(0.24, 0.36)
    notch_w = rng.uniform(0.06, 0.12)
    notch_h = rng.uniform(0.06, 0.12)
    pts = [
        Point(x0, y0),
        Point(x0 + w, y0),
        Point(x0 + w, y0 + h),
        Point(x0 + w - notch_w, y0 + h),
        Point(x0 + w - notch_w, y0 + h - notch_h),
        Point(x0, y0 + h - notch_h),
    ]
    prompt = "一个带缺口的矩形"
    return ShapeSample("notched_rectangle", pts, prompt, polygon_bbox(pts), closed=True)


def sample_u_shape(rng: random.Random) -> ShapeSample:
    x0 = rng.uniform(0.18, 0.28)
    y0 = rng.uniform(0.18, 0.28)
    w = rng.uniform(0.28, 0.40)
    h = rng.uniform(0.28, 0.42)
    thickness = rng.uniform(0.06, 0.10)
    pts = [
        Point(x0, y0),
        Point(x0 + w, y0),
        Point(x0 + w, y0 + h),
        Point(x0 + w - thickness, y0 + h),
        Point(x0 + w - thickness, y0 + thickness),
        Point(x0 + thickness, y0 + thickness),
        Point(x0 + thickness, y0 + h),
        Point(x0, y0 + h),
    ]
    prompt = "一个U形轮廓"
    return ShapeSample("u_shape", pts, prompt, polygon_bbox(pts), closed=True)


def sample_c_shape(rng: random.Random) -> ShapeSample:
    x0 = rng.uniform(0.18, 0.28)
    y0 = rng.uniform(0.18, 0.28)
    w = rng.uniform(0.28, 0.40)
    h = rng.uniform(0.28, 0.42)
    thickness = rng.uniform(0.06, 0.10)
    pts = [
        Point(x0 + w, y0),
        Point(x0, y0),
        Point(x0, y0 + h),
        Point(x0 + w, y0 + h),
        Point(x0 + w, y0 + h - thickness),
        Point(x0 + thickness, y0 + h - thickness),
        Point(x0 + thickness, y0 + thickness),
        Point(x0 + w, y0 + thickness),
    ]
    prompt = "一个C形轮廓"
    return ShapeSample("c_shape", pts, prompt, polygon_bbox(pts), closed=True)


def _sample_function_points(fn, x_start: float, x_end: float, n: int) -> List[Point]:
    xs = [x_start + (x_end - x_start) * i / max(n - 1, 1) for i in range(n)]
    ys = [fn(x) for x in xs]
    y_min = min(ys)
    y_max = max(ys)
    y_span = max(y_max - y_min, 1e-6)

    points = []
    for x, y in zip(xs, ys):
        px = 0.1 + 0.8 * ((x - x_start) / max(x_end - x_start, 1e-6))
        py = 0.15 + 0.7 * ((y - y_min) / y_span)
        points.append(Point(clamp(px), clamp(py)))
    return points


def sample_linear_function(rng: random.Random) -> ShapeSample:
    slope = rng.uniform(-1.5, 1.5)
    intercept = rng.uniform(-0.4, 0.4)
    points = _sample_function_points(lambda x: slope * x + intercept, -1.0, 1.0, rng.randint(14, 22))
    descriptor = "上升" if slope > 0.2 else "下降" if slope < -0.2 else "水平"
    prompt = f"一条{descriptor}的线性函数图像"
    return ShapeSample("linear_function", points, prompt, polygon_bbox(points), closed=False)


def sample_parabola(rng: random.Random) -> ShapeSample:
    a = rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
    h = rng.uniform(-0.4, 0.4)
    k = rng.uniform(-0.4, 0.2)
    points = _sample_function_points(lambda x: a * (x - h) ** 2 + k, -1.0, 1.0, rng.randint(18, 28))
    opening = "向上" if a > 0 else "向下"
    prompt = f"一条开口{opening}的抛物线"
    return ShapeSample("parabola", points, prompt, polygon_bbox(points), closed=False)


def sample_sine_wave(rng: random.Random) -> ShapeSample:
    amp = rng.uniform(0.5, 1.2)
    freq = rng.uniform(0.8, 2.2)
    phase = rng.uniform(0, math.pi)
    points = _sample_function_points(lambda x: amp * math.sin(freq * math.pi * x + phase), -1.0, 1.0, rng.randint(24, 36))
    descriptor = "低频" if freq < 1.3 else "高频"
    prompt = f"一条{descriptor}正弦波"
    return ShapeSample("sine_wave", points, prompt, polygon_bbox(points), closed=False)


def sample_cosine_wave(rng: random.Random) -> ShapeSample:
    amp = rng.uniform(0.5, 1.2)
    freq = rng.uniform(0.8, 2.2)
    phase = rng.uniform(0, math.pi)
    points = _sample_function_points(lambda x: amp * math.cos(freq * math.pi * x + phase), -1.0, 1.0, rng.randint(24, 36))
    descriptor = "低频" if freq < 1.3 else "高频"
    prompt = f"一条{descriptor}余弦波"
    return ShapeSample("cosine_wave", points, prompt, polygon_bbox(points), closed=False)


def sample_cubic_function(rng: random.Random) -> ShapeSample:
    a = rng.uniform(-1.2, 1.2)
    b = rng.uniform(-0.8, 0.8)
    c = rng.uniform(-0.6, 0.6)
    d = rng.uniform(-0.3, 0.3)
    points = _sample_function_points(lambda x: a * x**3 + b * x**2 + c * x + d, -1.0, 1.0, rng.randint(18, 30))
    prompt = "一条三次函数图像"
    return ShapeSample("cubic_function", points, prompt, polygon_bbox(points), closed=False)


def sample_absolute_value_function(rng: random.Random) -> ShapeSample:
    scale = rng.uniform(0.6, 1.3)
    shift_x = rng.uniform(-0.4, 0.4)
    shift_y = rng.uniform(-0.3, 0.2)
    points = _sample_function_points(lambda x: scale * abs(x - shift_x) + shift_y, -1.0, 1.0, rng.randint(14, 22))
    prompt = "一条绝对值函数图像"
    return ShapeSample("absolute_value_function", points, prompt, polygon_bbox(points), closed=False)


def sample_regular_polygon(rng: random.Random) -> ShapeSample:
    n_sides = rng.randint(5, 8)
    r = rng.uniform(0.08, 0.16)
    cx = rng.uniform(r + 0.08, 1 - r - 0.08)
    cy = rng.uniform(r + 0.08, 1 - r - 0.08)
    phase = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n_sides):
        t = phase + 2 * math.pi * i / n_sides
        pts.append(Point(cx + r * math.cos(t), cy + r * math.sin(t)))

    prompt = f"一个{describe_size(r * 2)}正多边形，{describe_position(cx, cy)}"
    return ShapeSample("regular_polygon", pts, prompt, polygon_bbox(pts), closed=True)


def sample_pentagon(rng: random.Random) -> ShapeSample:
    return _sample_named_regular_polygon(rng, 5, "pentagon")


def sample_hexagon(rng: random.Random) -> ShapeSample:
    return _sample_named_regular_polygon(rng, 6, "hexagon")


def sample_octagon(rng: random.Random) -> ShapeSample:
    return _sample_named_regular_polygon(rng, 8, "octagon")


def _sample_named_regular_polygon(rng: random.Random, n_sides: int, name: str) -> ShapeSample:
    r = rng.uniform(0.08, 0.16)
    cx = rng.uniform(r + 0.08, 1 - r - 0.08)
    cy = rng.uniform(r + 0.08, 1 - r - 0.08)
    phase = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n_sides):
        t = phase + 2 * math.pi * i / n_sides
        pts.append(Point(cx + r * math.cos(t), cy + r * math.sin(t)))
    _name_zh = {"pentagon": "五边形", "hexagon": "六边形", "octagon": "八边形"}
    prompt = f"一个{describe_size(r * 2)}{_name_zh.get(name, name)}，{describe_position(cx, cy)}"
    return ShapeSample(name, pts, prompt, polygon_bbox(pts), closed=True)


def sample_open_rectangle(rng: random.Random) -> ShapeSample:
    w = rng.uniform(0.15, 0.30)
    h = rng.uniform(0.12, 0.24)
    cx = rng.uniform(w / 2 + 0.06, 1 - w / 2 - 0.06)
    cy = rng.uniform(h / 2 + 0.06, 1 - h / 2 - 0.06)
    corners = [
        Point(cx - w / 2, cy - h / 2),
        Point(cx + w / 2, cy - h / 2),
        Point(cx + w / 2, cy + h / 2),
        Point(cx - w / 2, cy + h / 2),
    ]
    start_idx = rng.randint(0, 3)
    ordered = corners[start_idx:] + corners[:start_idx]
    ordered = ordered[:3]
    prompt = f"一个{describe_size(max(w, h))}开口矩形，{describe_position(cx, cy)}"
    return ShapeSample("open_rectangle", ordered, prompt, polygon_bbox(corners), closed=False)


def sample_star(rng: random.Random) -> ShapeSample:
    outer = rng.uniform(0.10, 0.16)
    inner = outer * rng.uniform(0.4, 0.6)
    cx = rng.uniform(outer + 0.08, 1 - outer - 0.08)
    cy = rng.uniform(outer + 0.08, 1 - outer - 0.08)
    phase = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(10):
        r = outer if i % 2 == 0 else inner
        t = phase + math.pi * i / 5
        pts.append(Point(cx + r * math.cos(t), cy + r * math.sin(t)))

    prompt = f"一个{describe_size(outer * 2)}星形，{describe_position(cx, cy)}"
    return ShapeSample("star", pts, prompt, polygon_bbox(pts), closed=True)


def sample_room(rng: random.Random) -> ShapeSample:
    x0 = rng.uniform(0.08, 0.22)
    y0 = rng.uniform(0.08, 0.22)
    w = rng.uniform(0.35, 0.55)
    h = rng.uniform(0.28, 0.45)
    notch_w = rng.uniform(0.10, 0.18)
    notch_h = rng.uniform(0.10, 0.16)
    notch_side = rng.choice(["top_right", "bottom_left"])

    if notch_side == "top_right":
        pts = [
            Point(x0, y0),
            Point(x0 + w, y0),
            Point(x0 + w, y0 + h - notch_h),
            Point(x0 + w - notch_w, y0 + h - notch_h),
            Point(x0 + w - notch_w, y0 + h),
            Point(x0, y0 + h),
        ]
        desc = "一个右上角有缺口的L形房间"
    else:
        pts = [
            Point(x0 + notch_w, y0),
            Point(x0 + w, y0),
            Point(x0 + w, y0 + h),
            Point(x0, y0 + h),
            Point(x0, y0 + notch_h),
            Point(x0 + notch_w, y0 + notch_h),
        ]
        desc = "一个左下角有缺口的L形房间"

    return ShapeSample("room", pts, desc, polygon_bbox(pts), closed=True)


def build_shape_from_type(shape_type: str, rng: random.Random) -> ShapeSample:
    if shape_type == "line":
        shape = sample_line(rng)
    elif shape_type == "polyline":
        shape = sample_polyline(rng)
    elif shape_type == "irregular_quad":
        shape = sample_irregular_quad(rng)
    elif shape_type == "trapezoid":
        shape = sample_trapezoid(rng)
    elif shape_type == "rhombus":
        shape = sample_rhombus(rng)
    elif shape_type == "parallelogram":
        shape = sample_parallelogram(rng)
    elif shape_type == "kite":
        shape = sample_kite(rng)
    elif shape_type == "square":
        shape = sample_square(rng)
    elif shape_type == "rotated_square":
        shape = sample_rotated_square(rng)
    elif shape_type == "rectangle":
        shape = sample_rectangle(rng)
    elif shape_type == "wide_rectangle":
        shape = sample_wide_rectangle(rng)
    elif shape_type == "tall_rectangle":
        shape = sample_tall_rectangle(rng)
    elif shape_type == "rotated_rectangle":
        shape = sample_rotated_rectangle(rng)
    elif shape_type == "open_rectangle":
        shape = sample_open_rectangle(rng)
    elif shape_type == "triangle":
        shape = sample_triangle(rng)
    elif shape_type == "equilateral_triangle":
        shape = sample_equilateral_triangle(rng)
    elif shape_type == "isosceles_triangle":
        shape = sample_isosceles_triangle(rng)
    elif shape_type == "right_triangle":
        shape = sample_right_triangle(rng)
    elif shape_type == "acute_triangle":
        shape = sample_acute_triangle(rng)
    elif shape_type == "obtuse_triangle":
        shape = sample_obtuse_triangle(rng)
    elif shape_type == "regular_polygon":
        shape = sample_regular_polygon(rng)
    elif shape_type == "pentagon":
        shape = sample_pentagon(rng)
    elif shape_type == "hexagon":
        shape = sample_hexagon(rng)
    elif shape_type == "octagon":
        shape = sample_octagon(rng)
    elif shape_type == "circle":
        shape = sample_circle(rng)
    elif shape_type == "ellipse":
        shape = sample_ellipse(rng)
    elif shape_type == "wide_ellipse":
        shape = sample_wide_ellipse(rng)
    elif shape_type == "tall_ellipse":
        shape = sample_tall_ellipse(rng)
    elif shape_type == "arc":
        shape = sample_arc(rng)
    elif shape_type == "notched_rectangle":
        shape = sample_notched_rectangle(rng)
    elif shape_type == "u_shape":
        shape = sample_u_shape(rng)
    elif shape_type == "c_shape":
        shape = sample_c_shape(rng)
    elif shape_type == "linear_function":
        shape = sample_linear_function(rng)
    elif shape_type == "parabola":
        shape = sample_parabola(rng)
    elif shape_type == "sine_wave":
        shape = sample_sine_wave(rng)
    elif shape_type == "cosine_wave":
        shape = sample_cosine_wave(rng)
    elif shape_type == "cubic_function":
        shape = sample_cubic_function(rng)
    elif shape_type == "absolute_value_function":
        shape = sample_absolute_value_function(rng)
    elif shape_type == "star":
        shape = sample_star(rng)
    elif shape_type == "room":
        shape = sample_room(rng)
    else:
        raise ValueError(f"unsupported shape_type: {shape_type}")
    return _sanitize_shape(shape)


def apply_relation_layout(shapes: List[ShapeSample], relation_type: str, rng: random.Random) -> List[ShapeSample]:
    if len(shapes) != 2 or relation_type is None:
        return shapes

    shape_a, shape_b = shapes
    width_a = shape_a.bbox["x_max"] - shape_a.bbox["x_min"]
    height_a = shape_a.bbox["y_max"] - shape_a.bbox["y_min"]
    width_b = shape_b.bbox["x_max"] - shape_b.bbox["x_min"]
    height_b = shape_b.bbox["y_max"] - shape_b.bbox["y_min"]
    center_a = bbox_center(shape_a.bbox)
    center_b = bbox_center(shape_b.bbox)

    if relation_type == "left_of":
        target_a = Point(0.28, rng.uniform(0.25, 0.75))
        target_b = Point(0.72, rng.uniform(0.25, 0.75))
    elif relation_type == "right_of":
        target_a = Point(0.72, rng.uniform(0.25, 0.75))
        target_b = Point(0.28, rng.uniform(0.25, 0.75))
    elif relation_type == "above":
        target_a = Point(rng.uniform(0.25, 0.75), 0.28)
        target_b = Point(rng.uniform(0.25, 0.75), 0.72)
    elif relation_type == "below":
        target_a = Point(rng.uniform(0.25, 0.75), 0.72)
        target_b = Point(rng.uniform(0.25, 0.75), 0.28)
    elif relation_type == "inside":
        shape_b = scale_shape(shape_b, rng.uniform(1.05, 1.25), rng.uniform(1.05, 1.25))
        shape_a = scale_shape(shape_a, rng.uniform(0.45, 0.70), rng.uniform(0.45, 0.70))
        width_a = shape_a.bbox["x_max"] - shape_a.bbox["x_min"]
        height_a = shape_a.bbox["y_max"] - shape_a.bbox["y_min"]
        width_b = shape_b.bbox["x_max"] - shape_b.bbox["x_min"]
        height_b = shape_b.bbox["y_max"] - shape_b.bbox["y_min"]
        if width_a >= width_b * 0.85 or height_a >= height_b * 0.85:
            shrink_x = min(0.60 * width_b / max(width_a, 1e-6), 0.55)
            shrink_y = min(0.60 * height_b / max(height_a, 1e-6), 0.55)
            shape_a = scale_shape(shape_a, shrink_x, shrink_y)
            width_a = shape_a.bbox["x_max"] - shape_a.bbox["x_min"]
            height_a = shape_a.bbox["y_max"] - shape_a.bbox["y_min"]
        center_a = bbox_center(shape_a.bbox)
        center_b = bbox_center(shape_b.bbox)
        target_b = Point(0.50, 0.50)
        slack_x = max((width_b - width_a) / 6, 0.005)
        slack_y = max((height_b - height_a) / 6, 0.005)
        target_a = Point(
            target_b.x + rng.uniform(-slack_x, slack_x),
            target_b.y + rng.uniform(-slack_y, slack_y),
        )
    elif relation_type == "adjacent":
        gap = rng.uniform(0.01, 0.03)
        if rng.choice([True, False]):
            target_a = Point(0.45, rng.uniform(0.35, 0.65))
            target_b = Point(target_a.x + (width_a + width_b) / 2 + gap, target_a.y)
        else:
            target_a = Point(rng.uniform(0.35, 0.65), 0.45)
            target_b = Point(target_a.x, target_a.y + (height_a + height_b) / 2 + gap)
    elif relation_type == "overlap":
        target_b = Point(rng.uniform(0.40, 0.60), rng.uniform(0.40, 0.60))
        target_a = Point(target_b.x + rng.uniform(-0.06, 0.06), target_b.y + rng.uniform(-0.06, 0.06))
    else:
        # separate
        horizontal = rng.choice([True, False])
        if horizontal:
            gap = rng.uniform(0.12, 0.20)
            target_a = Point(0.30, rng.uniform(0.25, 0.75))
            target_b = Point(target_a.x + (width_a + width_b) / 2 + gap, rng.uniform(0.25, 0.75))
        else:
            gap = rng.uniform(0.12, 0.20)
            target_a = Point(rng.uniform(0.25, 0.75), 0.30)
            target_b = Point(rng.uniform(0.25, 0.75), target_a.y + (height_a + height_b) / 2 + gap)

    shape_a = translate_shape(shape_a, target_a.x - center_a.x, target_a.y - center_a.y)
    shape_b = translate_shape(shape_b, target_b.x - center_b.x, target_b.y - center_b.y)
    return [shape_a, shape_b]


def build_geometry(scene_spec: Dict[str, object], rng: random.Random) -> List[ShapeSample]:
    if scene_spec["scene_type"] == "spatial_instruction":
        return build_spatial_instruction_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "contrast_pair":
        return build_contrast_pair_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "equation_function":
        return build_equation_function_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "triple_composition":
        return build_triple_composition_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "compound_relation":
        return build_compound_relation_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "constraint_curve":
        return build_constraint_curve_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "counting_group":
        return build_counting_group_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "alignment_group":
        return build_alignment_group_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "edge_contact":
        return build_edge_contact_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "relative_size_pair":
        return build_relative_size_pair_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "same_diff_pair":
        return build_same_diff_pair_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "through_points_curve":
        return build_through_points_curve_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "region_occupancy":
        return build_region_occupancy_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "step_instruction":
        return build_step_instruction_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "negation_constraint":
        return build_negation_constraint_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "style_precision":
        return build_style_precision_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "logic_conjunction":
        return build_logic_conjunction_geometry(scene_spec, rng)
    if scene_spec["scene_type"] == "reference_layout":
        return build_reference_layout_geometry(scene_spec, rng)
    shape_types = [rng.choice(scene_spec["allowed_shapes"]) for _ in range(scene_spec["num_shapes"])]
    shapes = [build_shape_from_type(shape_type, rng) for shape_type in shape_types]
    if len(shapes) == 1 and scene_spec.get("anchor_position"):
        return [apply_anchor_layout(shapes[0], scene_spec.get("anchor_position"))]
    return apply_relation_layout(shapes, scene_spec.get("relation_type"), rng)
