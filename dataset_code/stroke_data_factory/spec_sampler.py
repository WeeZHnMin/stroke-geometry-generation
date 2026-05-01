import random
from typing import Any, Dict, Optional


def difficulty_from_scene_type(scene_type: str) -> str:
    if scene_type in {"single_basic", "spatial_attribute"}:
        return "easy"
    if scene_type in {"double_basic", "spatial_instruction", "contrast_pair", "equation_function"}:
        return "medium"
    if scene_type in {"triple_composition", "compound_relation", "constraint_curve", "counting_group", "alignment_group", "edge_contact", "relative_size_pair", "same_diff_pair", "through_points_curve", "region_occupancy", "step_instruction", "negation_constraint", "style_precision", "logic_conjunction", "reference_layout"}:
        return "hard"
    return "hard"


RECIPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced_v1": {
        "scene_type_weights": {
            "single_basic": 0.55,
            "double_basic": 0.30,
            "room_only": 0.15,
        }
    },
    "level1_basic": {
        "scene_type_weights": {
            "single_basic": 0.85,
            "double_basic": 0.15,
            "room_only": 0.00,
        }
    },
    "level2_mixed": {
        "scene_type_weights": {
            "single_basic": 0.55,
            "double_basic": 0.35,
            "room_only": 0.10,
        }
    },
    "room_focus": {
        "scene_type_weights": {
            "single_basic": 0.15,
            "double_basic": 0.15,
            "room_only": 0.70,
        }
    },
    "coordinate_focus": {
        "scene_type_weights": {
            "single_basic": 0.65,
            "double_basic": 0.35,
            "room_only": 0.00,
        }
    },
    "spatial_instruction_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 1.00,
        }
    },
    "spatial_attribute_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 1.00,
        }
    },
    "spatial_relation_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 1.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
        }
    },
    "contrast_pair_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "contrast_pair": 1.00,
        }
    },
    "equation_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 1.00,
        }
    },
    "advanced_composition_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.40,
            "compound_relation": 0.35,
            "constraint_curve": 0.25,
        }
    },
    "triple_composition_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 1.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
        }
    },
    "compound_relation_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 1.00,
            "constraint_curve": 0.00,
        }
    },
    "constraint_curve_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 1.00,
        }
    },
    "counting_group_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 1.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
        }
    },
    "alignment_group_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 1.00,
            "edge_contact": 0.00,
        }
    },
    "edge_contact_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 1.00,
        }
    },
    "relative_size_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 1.00,
            "same_diff_pair": 0.00,
            "through_points_curve": 0.00,
            "region_occupancy": 0.00,
        }
    },
    "same_diff_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 0.00,
            "same_diff_pair": 1.00,
            "through_points_curve": 0.00,
            "region_occupancy": 0.00,
        }
    },
    "through_points_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 0.00,
            "same_diff_pair": 0.00,
            "through_points_curve": 1.00,
            "region_occupancy": 0.00,
        }
    },
    "region_occupancy_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 0.00,
            "same_diff_pair": 0.00,
            "through_points_curve": 0.00,
            "region_occupancy": 1.00,
        }
    },
    "step_instruction_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 0.00,
            "same_diff_pair": 0.00,
            "through_points_curve": 0.00,
            "region_occupancy": 0.00,
            "step_instruction": 1.00,
            "negation_constraint": 0.00,
        }
    },
    "negation_constraint_focus": {
        "scene_type_weights": {
            "single_basic": 0.00,
            "double_basic": 0.00,
            "room_only": 0.00,
            "spatial_instruction": 0.00,
            "spatial_attribute": 0.00,
            "spatial_relation": 0.00,
            "contrast_pair": 0.00,
            "equation_function": 0.00,
            "triple_composition": 0.00,
            "compound_relation": 0.00,
            "constraint_curve": 0.00,
            "counting_group": 0.00,
            "alignment_group": 0.00,
            "edge_contact": 0.00,
            "relative_size_pair": 0.00,
            "same_diff_pair": 0.00,
            "through_points_curve": 0.00,
            "region_occupancy": 0.00,
            "step_instruction": 0.00,
            "negation_constraint": 1.00,
        }
    },
    "style_precision_focus": {
        "scene_type_weights": {
            "single_basic": 0.00, "double_basic": 0.00, "room_only": 0.00,
            "spatial_instruction": 0.00, "spatial_attribute": 0.00, "spatial_relation": 0.00,
            "contrast_pair": 0.00, "equation_function": 0.00, "triple_composition": 0.00,
            "compound_relation": 0.00, "constraint_curve": 0.00, "counting_group": 0.00,
            "alignment_group": 0.00, "edge_contact": 0.00, "relative_size_pair": 0.00,
            "same_diff_pair": 0.00, "through_points_curve": 0.00, "region_occupancy": 0.00,
            "step_instruction": 0.00, "negation_constraint": 0.00, "style_precision": 1.00,
            "logic_conjunction": 0.00, "reference_layout": 0.00,
        }
    },
    "logic_conjunction_focus": {
        "scene_type_weights": {
            "single_basic": 0.00, "double_basic": 0.00, "room_only": 0.00,
            "spatial_instruction": 0.00, "spatial_attribute": 0.00, "spatial_relation": 0.00,
            "contrast_pair": 0.00, "equation_function": 0.00, "triple_composition": 0.00,
            "compound_relation": 0.00, "constraint_curve": 0.00, "counting_group": 0.00,
            "alignment_group": 0.00, "edge_contact": 0.00, "relative_size_pair": 0.00,
            "same_diff_pair": 0.00, "through_points_curve": 0.00, "region_occupancy": 0.00,
            "step_instruction": 0.00, "negation_constraint": 0.00, "style_precision": 0.00,
            "logic_conjunction": 1.00, "reference_layout": 0.00,
        }
    },
    "reference_layout_focus": {
        "scene_type_weights": {
            "single_basic": 0.00, "double_basic": 0.00, "room_only": 0.00,
            "spatial_instruction": 0.00, "spatial_attribute": 0.00, "spatial_relation": 0.00,
            "contrast_pair": 0.00, "equation_function": 0.00, "triple_composition": 0.00,
            "compound_relation": 0.00, "constraint_curve": 0.00, "counting_group": 0.00,
            "alignment_group": 0.00, "edge_contact": 0.00, "relative_size_pair": 0.00,
            "same_diff_pair": 0.00, "through_points_curve": 0.00, "region_occupancy": 0.00,
            "step_instruction": 0.00, "negation_constraint": 0.00, "style_precision": 0.00,
            "logic_conjunction": 0.00, "reference_layout": 1.00,
        }
    },
}


def get_recipe(recipe_name: str) -> Dict[str, Any]:
    if recipe_name not in RECIPE_PRESETS:
        valid = ", ".join(sorted(RECIPE_PRESETS))
        raise ValueError(f"unknown recipe '{recipe_name}', valid recipes: {valid}")
    return RECIPE_PRESETS[recipe_name]


def _build_scene_type_weights(recipe: Dict[str, Any], difficulty: Optional[str]) -> Dict[str, float]:
    weights = dict(recipe["scene_type_weights"])
    if difficulty is None:
        return weights

    filtered: Dict[str, float] = {}
    for scene_type, weight in weights.items():
        if difficulty_from_scene_type(scene_type) == difficulty:
            filtered[scene_type] = weight

    if not filtered:
        raise ValueError(f"recipe does not contain any scene type for difficulty '{difficulty}'")
    return filtered


def sample_scene_spec(
    rng: random.Random,
    recipe_name: str = "balanced_v1",
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    recipe = get_recipe(recipe_name)
    scene_type_weights = _build_scene_type_weights(recipe, difficulty)
    scene_types = list(scene_type_weights.keys())
    weights = list(scene_type_weights.values())

    scene_type = rng.choices(scene_types, weights=weights, k=1)[0]

    if scene_type == "single_basic":
        num_shapes = 1
        allowed_shapes = [
            "line", "polyline", "triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle",
            "square", "rotated_square", "rectangle", "wide_rectangle",
            "tall_rectangle", "rotated_rectangle", "open_rectangle", "regular_polygon",
            "pentagon", "hexagon", "octagon", "irregular_quad", "trapezoid", "rhombus", "parallelogram", "kite",
            "circle", "ellipse", "wide_ellipse", "tall_ellipse", "arc", "star",
            "notched_rectangle", "u_shape", "c_shape",
            "linear_function", "parabola", "sine_wave", "cosine_wave", "cubic_function", "absolute_value_function",
        ]
        relation_type = None
        anchor_position = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right", "center", "top", "bottom", "left", "right"]) if recipe_name == "coordinate_focus" else None
    elif scene_type == "double_basic":
        num_shapes = 2
        allowed_shapes = [
            "line", "polyline", "triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle",
            "square", "rotated_square", "rectangle", "wide_rectangle",
            "tall_rectangle", "rotated_rectangle", "open_rectangle", "regular_polygon",
            "pentagon", "hexagon", "octagon", "irregular_quad", "trapezoid", "rhombus", "parallelogram", "kite",
            "circle", "ellipse", "wide_ellipse", "tall_ellipse", "arc", "star",
            "notched_rectangle", "u_shape", "c_shape",
            "linear_function", "parabola", "sine_wave", "cosine_wave", "cubic_function", "absolute_value_function",
        ]
        relation_type = rng.choice(["left_of", "right_of", "above", "below", "separate", "inside", "adjacent", "overlap"])
        anchor_position = None
    else:
        if scene_type == "spatial_instruction":
            instruction_type = rng.choice([
                "parallel_lines",
                "perpendicular_lines",
                "intersecting_lines",
                "directed_line",
                "guided_curve",
            ])
            if instruction_type in {"parallel_lines", "perpendicular_lines", "intersecting_lines"}:
                num_shapes = 2
            else:
                num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            start_anchor = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right", "left", "right", "top", "bottom"])
            end_anchor = rng.choice([a for a in ["top_left", "top_right", "bottom_left", "bottom_right", "left", "right", "top", "bottom"] if a != start_anchor])
            curve_style = rng.choice(["curving_upward", "curving_downward", "s_bend"])
        elif scene_type == "spatial_attribute":
            num_shapes = 1
            allowed_shapes = [
                "line", "polyline",
                "triangle", "equilateral_triangle", "isosceles_triangle", "right_triangle",
                "square", "rotated_square", "rectangle", "wide_rectangle", "tall_rectangle", "rotated_rectangle",
                "circle", "ellipse", "wide_ellipse", "tall_ellipse",
                "arc", "regular_polygon", "pentagon", "hexagon", "octagon",
                "trapezoid", "rhombus", "parallelogram", "kite",
            ]
            relation_type = None
            anchor_position = rng.choice([
                "top_left", "top_right", "bottom_left", "bottom_right",
                "center", "top", "bottom", "left", "right",
                "upper_left", "upper_right", "lower_left", "lower_right",
                "center_left", "center_right",
            ])
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
        elif scene_type == "contrast_pair":
            num_shapes = 2
            allowed_shapes = []
            relation_type = "separate"
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = rng.choice(["rectangle", "triangle", "square", "ellipse"])
        elif scene_type == "equation_function":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            function_family = rng.choice(["linear", "quadratic", "cubic"])
            if function_family == "linear":
                m = rng.choice([-3, -2, -1, 1, 2, 3])
                b = rng.randint(-5, 5)
                equation_spec = {"family": "linear", "m": m, "b": b}
            elif function_family == "quadratic":
                standard = rng.random() < 0.35
                if standard:
                    equation_spec = {"family": "quadratic", "a": 1, "b": 0, "c": 0, "standard": True}
                else:
                    a = rng.choice([-2, -1, 1, 2])
                    b = rng.randint(-4, 4)
                    c = rng.randint(-5, 5)
                    equation_spec = {"family": "quadratic", "a": a, "b": b, "c": c, "standard": False}
            else:
                a = rng.choice([-2, -1, 1, 2])
                b = rng.randint(-3, 3)
                c = rng.randint(-3, 3)
                d = rng.randint(-4, 4)
                equation_spec = {"family": "cubic", "a": a, "b": b, "c": c, "d": d}
            contrast_family = None
        elif scene_type == "triple_composition":
            num_shapes = 3
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = rng.choice(["mixed_basic", "rect_tri_square", "circles_and_triangle"])
            compound_relation = None
            constraint_type = None
        elif scene_type == "compound_relation":
            num_shapes = 2
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = rng.choice(["above_left_of", "above_right_of", "below_left_of", "below_right_of"])
            constraint_type = None
        elif scene_type == "constraint_curve":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = rng.choice(["left", "top_left", "bottom_left"])
            end_anchor = rng.choice(["right", "top_right", "bottom_right"])
            curve_style = rng.choice(["curving_upward", "curving_downward", "s_bend"])
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = rng.choice(["through_center", "touch_top_edge", "touch_bottom_edge"])
        elif scene_type == "counting_group":
            num_shapes = rng.choice([3, 4])
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = rng.choice(["circles", "squares", "rectangles"])
            alignment_mode = rng.choice(["row", "column", "grid"])
            edge_mode = None
        elif scene_type == "alignment_group":
            num_shapes = rng.choice([2, 3, 4])
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = rng.choice(["aligned_horizontally", "aligned_vertically", "symmetric_about_center"])
            edge_mode = None
        elif scene_type == "edge_contact":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = rng.choice(["touch_left_edge", "touch_right_edge", "touch_top_edge", "touch_bottom_edge", "touch_top_right_corner"])
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
        elif scene_type == "relative_size_pair":
            num_shapes = 2
            allowed_shapes = []
            relation_type = rng.choice(["left_of", "right_of", "above", "below", "separate"])
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = rng.choice(["larger_than", "smaller_than"])
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
        elif scene_type == "same_diff_pair":
            num_shapes = 2
            allowed_shapes = []
            relation_type = rng.choice(["separate", "left_of", "right_of"])
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = rng.choice(["same_shape", "different_shapes"])
            point_constraint = None
            occupancy_region = None
        elif scene_type == "through_points_curve":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = rng.choice(["left", "top_left", "bottom_left"])
            end_anchor = rng.choice(["right", "top_right", "bottom_right"])
            curve_style = rng.choice(["curving_upward", "curving_downward", "s_bend"])
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = rng.choice(["through_center", "through_top_right", "through_bottom_left"])
            occupancy_region = None
        elif scene_type == "region_occupancy":
            num_shapes = rng.choice([3, 4])
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = rng.choice(["circles", "squares", "mixed_shapes"])
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = rng.choice(["upper_half", "lower_half", "left_half", "right_half"])
            step_plan = None
            negation_mode = None
        elif scene_type == "step_instruction":
            num_shapes = 2
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = rng.choice(["circle_then_rectangle_right", "square_then_triangle_below", "ellipse_then_line_above"])
            negation_mode = None
        elif scene_type == "negation_constraint":
            num_shapes = 2
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = None
            negation_mode = rng.choice(["two_non_intersecting_circles", "two_shapes_not_touching_boundary", "two_shapes_not_overlapping"])
            style_target = None
            conjunction_mode = None
            reference_target = None
        elif scene_type == "style_precision":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = None
            negation_mode = None
            style_target = rng.choice(["nearly_square_rectangle", "very_flat_ellipse", "very_tall_rectangle"])
            conjunction_mode = None
            reference_target = None
        elif scene_type == "logic_conjunction":
            num_shapes = 2
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = None
            negation_mode = None
            style_target = None
            conjunction_mode = rng.choice(["non_intersecting_upper_half", "separate_right_half"])
            reference_target = None
        elif scene_type == "reference_layout":
            num_shapes = 1
            allowed_shapes = []
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = None
            negation_mode = None
            style_target = None
            conjunction_mode = None
            reference_target = rng.choice(["top_middle", "bottom_middle", "center_left", "center_right"])
        else:
            num_shapes = 1
            allowed_shapes = ["room"]
            relation_type = None
            anchor_position = None
            instruction_type = None
            start_anchor = None
            end_anchor = None
            curve_style = None
            contrast_family = None
            equation_spec = None
            triple_family = None
            compound_relation = None
            constraint_type = None
            count_family = None
            alignment_mode = None
            edge_mode = None
            size_relation = None
            sameness_mode = None
            point_constraint = None
            occupancy_region = None
            step_plan = None
            negation_mode = None
            style_target = None
            conjunction_mode = None
            reference_target = None

    spec = {
        "scene_type": scene_type,
        "difficulty": difficulty_from_scene_type(scene_type),
        "num_shapes": num_shapes,
        "allowed_shapes": allowed_shapes,
        "relation_type": relation_type,
        "anchor_position": anchor_position,
        "recipe_name": recipe_name,
    }
    if scene_type == "spatial_instruction":
        spec.update(
            {
                "instruction_type": instruction_type,
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "curve_style": curve_style,
            }
        )
    if scene_type == "contrast_pair":
        spec["contrast_family"] = contrast_family
    if scene_type == "equation_function":
        spec["equation_spec"] = equation_spec
    if scene_type == "triple_composition":
        spec["triple_family"] = triple_family
    if scene_type == "compound_relation":
        spec["compound_relation"] = compound_relation
    if scene_type == "constraint_curve":
        spec.update(
            {
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "curve_style": curve_style,
                "constraint_type": constraint_type,
            }
        )
    if scene_type == "counting_group":
        spec["count_family"] = count_family
        spec["alignment_mode"] = alignment_mode
    if scene_type == "alignment_group":
        spec["alignment_mode"] = alignment_mode
    if scene_type == "edge_contact":
        spec["edge_mode"] = edge_mode
    if scene_type == "relative_size_pair":
        spec["size_relation"] = size_relation
    if scene_type == "same_diff_pair":
        spec["sameness_mode"] = sameness_mode
    if scene_type == "through_points_curve":
        spec.update(
            {
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "curve_style": curve_style,
                "point_constraint": point_constraint,
            }
        )
    if scene_type == "region_occupancy":
        spec["count_family"] = count_family
        spec["occupancy_region"] = occupancy_region
    if scene_type == "step_instruction":
        spec["step_plan"] = step_plan
    if scene_type == "negation_constraint":
        spec["negation_mode"] = negation_mode
    if scene_type == "style_precision":
        spec["style_target"] = style_target
    if scene_type == "logic_conjunction":
        spec["conjunction_mode"] = conjunction_mode
    if scene_type == "reference_layout":
        spec["reference_target"] = reference_target
    return spec
