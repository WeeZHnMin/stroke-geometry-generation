import math
from typing import List

from .schema import ShapeSample


RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
    "separate": "separate from",
    "inside": "inside",
    "adjacent": "adjacent to",
    "overlap": "overlapping",
}


def _article_for(text: str) -> str:
    return "an" if text[0].lower() in "aeiou" else "a"


def _shape_display_name(shape_type: str) -> str:
    names = {
        "line": "line segment",
        "polyline": "polyline",
        "triangle": "triangle",
        "equilateral_triangle": "equilateral triangle",
        "isosceles_triangle": "isosceles triangle",
        "right_triangle": "right triangle",
        "square": "square",
        "rotated_square": "rotated square",
        "rectangle": "rectangle",
        "wide_rectangle": "wide rectangle",
        "tall_rectangle": "tall rectangle",
        "rotated_rectangle": "rotated rectangle",
        "open_rectangle": "open rectangle",
        "regular_polygon": "regular polygon",
        "pentagon": "pentagon",
        "hexagon": "hexagon",
        "octagon": "octagon",
        "irregular_quad": "irregular quadrilateral",
        "trapezoid": "trapezoid",
        "rhombus": "rhombus",
        "parallelogram": "parallelogram",
        "kite": "kite shape",
        "circle": "circle",
        "ellipse": "ellipse",
        "wide_ellipse": "wide ellipse",
        "tall_ellipse": "tall ellipse",
        "arc": "arc",
        "star": "star shape",
        "notched_rectangle": "notched rectangle",
        "u_shape": "U-shaped outline",
        "c_shape": "C-shaped outline",
        "linear_function": "linear function graph",
        "parabola": "parabola",
        "sine_wave": "sine wave",
        "cosine_wave": "cosine wave",
        "cubic_function": "cubic function graph",
        "absolute_value_function": "absolute value function",
        "guided_curve": "guided curve",
        "room": "room outline",
    }
    return names.get(shape_type, shape_type)


def _position_from_bbox(shape: ShapeSample) -> str:
    cx = (shape.bbox["x_min"] + shape.bbox["x_max"]) / 2
    cy = (shape.bbox["y_min"] + shape.bbox["y_max"]) / 2
    horiz = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
    vert = "top" if cy < 0.35 else "bottom" if cy > 0.65 else "middle"
    if horiz == "center" and vert == "middle":
        return "near the center"
    if horiz == "center":
        return f"near the {vert}"
    if vert == "middle":
        return f"on the {horiz}"
    return f"in the {vert} {horiz}"


ANCHOR_POSITION_TEXT = {
    "top_left": "in the top left",
    "top_right": "in the top right",
    "bottom_left": "in the bottom left",
    "bottom_right": "in the bottom right",
    "center": "near the center",
    "top": "near the top edge",
    "bottom": "near the bottom edge",
    "left": "on the left side",
    "right": "on the right side",
    "upper_left": "near the upper left area",
    "upper_right": "near the upper right area",
    "lower_left": "near the lower left area",
    "lower_right": "near the lower right area",
    "center_left": "near the center left",
    "center_right": "near the center right",
}


def _size_from_bbox(shape: ShapeSample) -> str:
    span = max(shape.bbox["x_max"] - shape.bbox["x_min"], shape.bbox["y_max"] - shape.bbox["y_min"])
    if span < 0.12:
        return "tiny"
    if span < 0.18:
        return "small"
    if span < 0.3:
        return "medium"
    return "large"


def _relation_fragment(shape: ShapeSample) -> str:
    phrase = _detailed_shape_phrase(shape)
    return f"{_article_for(phrase)} {phrase}"


def _line_orientation(shape: ShapeSample) -> str:
    if len(shape.points) < 2:
        return "short"
    p0 = shape.points[0]
    p1 = shape.points[-1]
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    if abs(dy) < abs(dx) * 0.35:
        return "horizontal"
    if abs(dx) < abs(dy) * 0.35:
        return "vertical"
    return "rising diagonal" if dx * dy > 0 else "falling diagonal"


def _triangle_direction(shape: ShapeSample) -> str:
    cx = sum(p.x for p in shape.points) / len(shape.points)
    cy = sum(p.y for p in shape.points) / len(shape.points)
    apex = max(shape.points, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
    dx = apex.x - cx
    dy = apex.y - cy
    if abs(dx) > abs(dy):
        return "pointing right" if dx > 0 else "pointing left"
    return "pointing down" if dy > 0 else "pointing up"


def _arc_direction(shape: ShapeSample) -> str:
    if len(shape.points) < 3:
        return "curved"
    start = shape.points[0]
    mid = shape.points[len(shape.points) // 2]
    end = shape.points[-1]
    baseline_y = (start.y + end.y) / 2
    if mid.y < baseline_y - 0.02:
        return "upward-opening"
    if mid.y > baseline_y + 0.02:
        return "downward-opening"
    return "side-opening"


def _detailed_shape_phrase(shape: ShapeSample) -> str:
    size = _size_from_bbox(shape)
    t = shape.shape_type
    if t == "line":
        return f"{size} {_line_orientation(shape)} line segment"
    if t == "polyline":
        return f"{size} bent polyline"
    if t in {"triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle"}:
        prefix = {
            "triangle": "triangle",
            "equilateral_triangle": "equilateral triangle",
            "isosceles_triangle": "isosceles triangle",
            "right_triangle": "right triangle",
        }[t]
        return f"{size} {prefix} {_triangle_direction(shape)}"
    if t in {"square", "rotated_square"}:
        return f"{size} {_shape_display_name(t)}"
    if t in {"rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle"}:
        return f"{size} {_shape_display_name(t)}"
    if t in {"circle", "ellipse", "wide_ellipse", "tall_ellipse"}:
        return f"{size} {_shape_display_name(t)}"
    if t == "arc":
        return f"{size} {_arc_direction(shape)} arc"
    if t in {"regular_polygon", "pentagon", "hexagon", "octagon", "trapezoid", "rhombus", "parallelogram", "kite"}:
        return f"{size} {_shape_display_name(t)}"
    return f"{size} {_shape_display_name(t)}"


def make_prompt(
    shapes: List[ShapeSample],
    relation_type: str | None = None,
    anchor_position: str | None = None,
    scene_spec: dict | None = None,
) -> str:
    if scene_spec and scene_spec.get("scene_type") == "spatial_instruction":
        instruction_type = scene_spec.get("instruction_type")
        if instruction_type == "parallel_lines":
            orientation = "horizontal" if abs(shapes[0].points[0].y - shapes[0].points[1].y) < abs(shapes[0].points[0].x - shapes[0].points[1].x) else "vertical"
            return f"draw two parallel {orientation} line segments"
        if instruction_type == "perpendicular_lines":
            return "draw one horizontal line and one vertical line crossing each other"
        if instruction_type == "intersecting_lines":
            return "draw two intersecting line segments"
        if instruction_type == "directed_line":
            return f"draw a line from the {scene_spec.get('start_anchor').replace('_', ' ')} to the {scene_spec.get('end_anchor').replace('_', ' ')}"
        if instruction_type == "guided_curve":
            curve_style = scene_spec.get("curve_style", "").replace("_", " ")
            return f"draw a {curve_style} curve from the {scene_spec.get('start_anchor').replace('_', ' ')} to the {scene_spec.get('end_anchor').replace('_', ' ')}"
    if scene_spec and scene_spec.get("scene_type") == "contrast_pair":
        family = scene_spec.get("contrast_family")
        family_text = {
            "rectangle": "rectangles",
            "triangle": "triangles",
            "square": "squares",
            "ellipse": "ellipses",
        }[family]
        return f"draw two different {family_text}"
    if scene_spec and scene_spec.get("scene_type") == "equation_function":
        spec = scene_spec.get("equation_spec", {})
        if spec.get("family") == "linear":
            m = spec["m"]
            b = spec["b"]
            sign = "+" if b >= 0 else "-"
            return f"draw the graph of y = {m}x {sign} {abs(b)}"
        if spec.get("family") == "quadratic":
            if spec.get("standard"):
                return "draw the standard quadratic function"
            a, b, c = spec["a"], spec["b"], spec["c"]
            parts = [f"{a}x^2"]
            if b != 0:
                parts.append(f"+ {b}x" if b > 0 else f"- {abs(b)}x")
            if c != 0:
                parts.append(f"+ {c}" if c > 0 else f"- {abs(c)}")
            return "draw the graph of y = " + " ".join(parts)
        a, b, c, d = spec["a"], spec["b"], spec["c"], spec["d"]
        parts = [f"{a}x^3"]
        if b != 0:
            parts.append(f"+ {b}x^2" if b > 0 else f"- {abs(b)}x^2")
        if c != 0:
            parts.append(f"+ {c}x" if c > 0 else f"- {abs(c)}x")
        if d != 0:
            parts.append(f"+ {d}" if d > 0 else f"- {abs(d)}")
        return "draw the graph of y = " + " ".join(parts)
    if scene_spec and scene_spec.get("scene_type") == "triple_composition":
        family = scene_spec.get("triple_family")
        if family == "rect_tri_square":
            return "draw a rectangle, a triangle, and a square"
        if family == "circles_and_triangle":
            return "draw two rounded shapes and one triangle"
        return "draw three different shapes"
    if scene_spec and scene_spec.get("scene_type") == "compound_relation":
        return f"draw one shape {scene_spec.get('compound_relation', '').replace('_', ' ')} another shape"
    if scene_spec and scene_spec.get("scene_type") == "constraint_curve":
        constraint = scene_spec.get("constraint_type")
        if constraint == "through_center":
            return "draw a curve passing through the center"
        if constraint == "touch_top_edge":
            return "draw a curve touching the top edge"
        return "draw a curve touching the bottom edge"
    if scene_spec and scene_spec.get("scene_type") == "counting_group":
        count = scene_spec.get("num_shapes")
        family = scene_spec.get("count_family")
        mode = scene_spec.get("alignment_mode")
        if mode == "column":
            return f"draw {count} {family} arranged vertically"
        if mode == "grid":
            return f"draw {count} {family} in a grid"
        return f"draw {count} {family} arranged horizontally"
    if scene_spec and scene_spec.get("scene_type") == "alignment_group":
        mode = scene_spec.get("alignment_mode")
        count = scene_spec.get("num_shapes")
        if mode == "aligned_horizontally":
            return f"draw {count} shapes aligned horizontally"
        if mode == "aligned_vertically":
            return f"draw {count} shapes aligned vertically"
        return f"draw {count} shapes symmetric about the center"
    if scene_spec and scene_spec.get("scene_type") == "edge_contact":
        mode = scene_spec.get("edge_mode", "").replace("_", " ")
        return f"draw a shape {mode}"
    if scene_spec and scene_spec.get("scene_type") == "relative_size_pair":
        relation = "larger than" if scene_spec.get("size_relation") == "larger_than" else "smaller than"
        return f"draw one shape {relation} the other"
    if scene_spec and scene_spec.get("scene_type") == "same_diff_pair":
        if scene_spec.get("sameness_mode") == "same_shape":
            return "draw two shapes of the same kind"
        return "draw two different shapes"
    if scene_spec and scene_spec.get("scene_type") == "through_points_curve":
        pc = scene_spec.get("point_constraint", "").replace("_", " ")
        return f"draw a curve passing through the {pc.replace('through ', '')}"
    if scene_spec and scene_spec.get("scene_type") == "region_occupancy":
        count = scene_spec.get("num_shapes")
        family = scene_spec.get("count_family", "").replace("_", " ")
        region = scene_spec.get("occupancy_region", "").replace("_", " ")
        return f"draw {count} {family} in the {region}"
    if scene_spec and scene_spec.get("scene_type") == "step_instruction":
        plan = scene_spec.get("step_plan")
        if plan == "circle_then_rectangle_right":
            return "first draw a circle, then draw a rectangle to its right"
        if plan == "square_then_triangle_below":
            return "first draw a square, then draw a triangle below it"
        return "first draw an ellipse, then draw a line above it"
    if scene_spec and scene_spec.get("scene_type") == "negation_constraint":
        mode = scene_spec.get("negation_mode")
        if mode == "two_non_intersecting_circles":
            return "draw two circles that do not intersect"
        if mode == "two_shapes_not_touching_boundary":
            return "draw two shapes that do not touch the boundary"
        return "draw two shapes that do not overlap"
    if scene_spec and scene_spec.get("scene_type") == "style_precision":
        target = scene_spec.get("style_target")
        if target == "nearly_square_rectangle":
            return "draw a nearly square rectangle"
        if target == "very_flat_ellipse":
            return "draw a very flat ellipse"
        return "draw a very tall rectangle"
    if scene_spec and scene_spec.get("scene_type") == "logic_conjunction":
        mode = scene_spec.get("conjunction_mode")
        if mode == "non_intersecting_upper_half":
            return "draw two shapes that do not intersect and stay in the upper half"
        return "draw two shapes that stay separate in the right half"
    if scene_spec and scene_spec.get("scene_type") == "reference_layout":
        target = scene_spec.get("reference_target", "").replace("_", " ")
        return f"draw a shape centered at the {target}"

    if scene_spec and scene_spec.get("scene_type") == "spatial_attribute":
        phrase = _detailed_shape_phrase(shapes[0])
        position = ANCHOR_POSITION_TEXT.get(anchor_position, _position_from_bbox(shapes[0]))
        return f"draw {_article_for(phrase)} {phrase} {position}"

    if len(shapes) == 1:
        size = _size_from_bbox(shapes[0])
        name = _shape_display_name(shapes[0].shape_type)
        position = _position_from_bbox(shapes[0])
        if anchor_position:
            return f"draw {_article_for(size)} {size} {name} {position}"
        return f"draw {_article_for(size)} {size} {name} {position}"

    if len(shapes) == 2:
        if relation_type in RELATION_TEXT:
            return f"draw {_relation_fragment(shapes[0])} {RELATION_TEXT[relation_type]} {_relation_fragment(shapes[1])}"
        left_shape, right_shape = sorted(shapes, key=lambda s: (s.bbox["x_min"] + s.bbox["x_max"]) / 2)
        return f"draw {left_shape.prompt_fragment} and {right_shape.prompt_fragment}"

    fragments = [s.prompt_fragment for s in shapes]
    return "draw " + ", ".join(fragments[:-1]) + f", and {fragments[-1]}"
