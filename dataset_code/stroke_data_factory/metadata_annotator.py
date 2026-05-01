from typing import Any, Dict, List

from .schema import ShapeSample, StrokeStep


def build_capability_tags(scene_type: str, shapes: List[ShapeSample]) -> List[str]:
    tags = set()
    if any(shape.closed for shape in shapes):
        tags.add("requires_closure")
    if any(not shape.closed for shape in shapes):
        tags.add("requires_open_contour")
    shape_types = {shape.shape_type for shape in shapes}
    if any(shape_type in {"triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle", "square", "rotated_square", "rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle", "open_rectangle", "regular_polygon", "pentagon", "hexagon", "octagon", "irregular_quad", "trapezoid", "rhombus", "parallelogram", "kite", "star", "notched_rectangle", "u_shape", "c_shape", "room", "absolute_value_function"} for shape_type in shape_types):
        tags.add("requires_corner_turning")
    if any(shape_type in {"circle", "ellipse", "wide_ellipse", "tall_ellipse", "arc", "sine_wave", "cosine_wave", "parabola", "cubic_function"} for shape_type in shape_types):
        tags.add("requires_curvature")
    if any(shape_type in {"linear_function", "parabola", "sine_wave", "cosine_wave", "cubic_function", "absolute_value_function"} for shape_type in shape_types):
        tags.add("requires_function_curve")
    if len(shapes) > 1:
        tags.add("requires_multi_shape")
        tags.add("requires_left_right_relation")
    if scene_type == "room_only":
        tags.add("requires_structured_outline")
    return sorted(tags)


def annotate_metadata(scene_spec: Dict[str, Any], shapes: List[ShapeSample], strokes: List[StrokeStep]) -> Dict[str, Any]:
    relation_type = scene_spec.get("relation_type")
    anchor_position = scene_spec.get("anchor_position")
    tags = set(build_capability_tags(scene_spec["scene_type"], shapes))
    if relation_type in {"above", "below"}:
        tags.add("requires_vertical_relation")
    if relation_type == "separate":
        tags.add("requires_separation")
    if relation_type == "inside":
        tags.add("requires_containment")
    if relation_type == "adjacent":
        tags.add("requires_adjacency_reasoning")
    if relation_type == "overlap":
        tags.add("requires_overlap_reasoning")
    if anchor_position is not None:
        tags.add("requires_coordinate_grounding")
    instruction_type = scene_spec.get("instruction_type")
    if instruction_type is not None:
        tags.add("requires_spatial_instruction_following")
        if instruction_type in {"parallel_lines", "perpendicular_lines", "intersecting_lines", "directed_line"}:
            tags.add("requires_line_geometry")
        if instruction_type == "guided_curve":
            tags.add("requires_guided_curve")
    if scene_spec.get("scene_type") == "spatial_attribute":
        tags.add("requires_shape_attribute_grounding")
        tags.add("requires_fine_grained_positioning")
    if scene_spec.get("scene_type") == "contrast_pair":
        tags.add("requires_difference_reasoning")
        tags.add("requires_same_family_comparison")
    if scene_spec.get("scene_type") == "equation_function":
        tags.add("requires_equation_grounding")
        tags.add("requires_symbolic_to_geometry_mapping")
    if scene_spec.get("scene_type") == "triple_composition":
        tags.add("requires_three_shape_composition")
    if scene_spec.get("scene_type") == "compound_relation":
        tags.add("requires_compound_relation_reasoning")
    if scene_spec.get("scene_type") == "constraint_curve":
        tags.add("requires_path_constraint_following")
    if scene_spec.get("scene_type") == "counting_group":
        tags.add("requires_counting")
        tags.add("requires_group_arrangement")
    if scene_spec.get("scene_type") == "alignment_group":
        tags.add("requires_alignment_reasoning")
    if scene_spec.get("scene_type") == "edge_contact":
        tags.add("requires_boundary_contact_reasoning")
    if scene_spec.get("scene_type") == "relative_size_pair":
        tags.add("requires_relative_size_reasoning")
    if scene_spec.get("scene_type") == "same_diff_pair":
        tags.add("requires_same_different_reasoning")
    if scene_spec.get("scene_type") == "through_points_curve":
        tags.add("requires_waypoint_reasoning")
    if scene_spec.get("scene_type") == "region_occupancy":
        tags.add("requires_region_occupancy_reasoning")
    if scene_spec.get("scene_type") == "step_instruction":
        tags.add("requires_step_by_step_instruction_following")
    if scene_spec.get("scene_type") == "negation_constraint":
        tags.add("requires_negation_constraint_reasoning")
    if scene_spec.get("scene_type") == "style_precision":
        tags.add("requires_style_precision_grounding")
    if scene_spec.get("scene_type") == "logic_conjunction":
        tags.add("requires_logical_conjunction_reasoning")
    if scene_spec.get("scene_type") == "reference_layout":
        tags.add("requires_reference_point_grounding")
    return {
        "difficulty": scene_spec["difficulty"],
        "capability_tags": sorted(tags),
        "num_shapes": len(shapes),
        "sequence_length": len(strokes),
        "scene_type": scene_spec["scene_type"],
        "relation_type": relation_type,
        "anchor_position": anchor_position,
        "instruction_type": instruction_type,
        "contrast_family": scene_spec.get("contrast_family"),
        "equation_spec": scene_spec.get("equation_spec"),
        "triple_family": scene_spec.get("triple_family"),
        "compound_relation": scene_spec.get("compound_relation"),
        "constraint_type": scene_spec.get("constraint_type"),
        "count_family": scene_spec.get("count_family"),
        "alignment_mode": scene_spec.get("alignment_mode"),
        "edge_mode": scene_spec.get("edge_mode"),
        "size_relation": scene_spec.get("size_relation"),
        "sameness_mode": scene_spec.get("sameness_mode"),
        "point_constraint": scene_spec.get("point_constraint"),
        "occupancy_region": scene_spec.get("occupancy_region"),
        "step_plan": scene_spec.get("step_plan"),
        "negation_mode": scene_spec.get("negation_mode"),
        "style_target": scene_spec.get("style_target"),
        "conjunction_mode": scene_spec.get("conjunction_mode"),
        "reference_target": scene_spec.get("reference_target"),
        "shape_types": [shape.shape_type for shape in shapes],
    }
