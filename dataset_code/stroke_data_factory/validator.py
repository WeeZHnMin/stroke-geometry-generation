import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


EPS = 1e-6


@dataclass
class ValidationIssue:
    sample_index: int
    code: str
    message: str


def _bbox_from_points(points: List[Dict[str, float]]) -> Dict[str, float]:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def _bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    return ((bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2)


def _bbox_overlap(b1: Dict[str, float], b2: Dict[str, float]) -> Tuple[float, float]:
    x_overlap = max(0.0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    y_overlap = max(0.0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    return x_overlap, y_overlap


def _bbox_contains(outer: Dict[str, float], inner: Dict[str, float], margin: float = 0.02) -> bool:
    return (
        inner["x_min"] >= outer["x_min"] - margin
        and inner["y_min"] >= outer["y_min"] - margin
        and inner["x_max"] <= outer["x_max"] + margin
        and inner["y_max"] <= outer["y_max"] + margin
    )


def _validate_relation(relation_type: str | None, shapes: List[Dict[str, Any]]) -> str | None:
    if relation_type is None or len(shapes) != 2:
        return None

    bbox_a = shapes[0]["bbox"]
    bbox_b = shapes[1]["bbox"]
    cx_a, cy_a = _bbox_center(bbox_a)
    cx_b, cy_b = _bbox_center(bbox_b)
    x_overlap, y_overlap = _bbox_overlap(bbox_a, bbox_b)

    if relation_type == "left_of" and not (cx_a < cx_b):
        return "left_of relation violated"
    if relation_type == "right_of" and not (cx_a > cx_b):
        return "right_of relation violated"
    if relation_type == "above" and not (cy_a < cy_b):
        return "above relation violated"
    if relation_type == "below" and not (cy_a > cy_b):
        return "below relation violated"
    if relation_type == "separate" and (x_overlap > 0.02 and y_overlap > 0.02):
        return "separate relation violated"
    if relation_type == "inside" and not _bbox_contains(bbox_b, bbox_a):
        return "inside relation violated"
    if relation_type == "overlap" and not (x_overlap > 0.01 and y_overlap > 0.01):
        return "overlap relation violated"
    if relation_type == "adjacent":
        near_x = abs(bbox_a["x_max"] - bbox_b["x_min"]) < 0.08 or abs(bbox_b["x_max"] - bbox_a["x_min"]) < 0.08
        near_y = abs(bbox_a["y_max"] - bbox_b["y_min"]) < 0.08 or abs(bbox_b["y_max"] - bbox_a["y_min"]) < 0.08
        if not (near_x or near_y):
            return "adjacent relation violated"
    return None


def validate_sample(sample: Dict[str, Any], sample_index: int) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    shapes = sample.get("shapes", [])
    strokes = sample.get("strokes", [])
    metadata = sample.get("metadata", {})
    scene_spec = sample.get("scene_spec", {})

    if metadata.get("num_shapes") != len(shapes):
        issues.append(ValidationIssue(sample_index, "num_shapes_mismatch", "metadata.num_shapes does not match shapes length"))

    if metadata.get("sequence_length") != len(strokes):
        issues.append(ValidationIssue(sample_index, "sequence_length_mismatch", "metadata.sequence_length does not match strokes length"))

    if not strokes:
        issues.append(ValidationIssue(sample_index, "empty_strokes", "stroke sequence is empty"))
        return issues

    if strokes[-1]["pen_state"] != "end_all":
        issues.append(ValidationIssue(sample_index, "missing_end_all", "last stroke is not end_all"))

    end_shape_count = sum(1 for s in strokes if s["pen_state"] == "end_shape")
    expected_end_shape = max(len(shapes) - 1, 0)
    if end_shape_count != expected_end_shape:
        issues.append(ValidationIssue(sample_index, "end_shape_count_mismatch", f"expected {expected_end_shape} end_shape, got {end_shape_count}"))

    valid_pen = {"move", "draw", "end_shape", "end_all"}
    for idx, stroke in enumerate(strokes):
        if stroke["pen_state"] not in valid_pen:
            issues.append(ValidationIssue(sample_index, "invalid_pen_state", f"invalid pen_state at stroke {idx}: {stroke['pen_state']}"))

    for shape_idx, shape in enumerate(shapes):
        points = shape["points"]
        bbox = shape["bbox"]
        derived = _bbox_from_points(points)

        for k in ["x_min", "y_min", "x_max", "y_max"]:
            if abs(derived[k] - bbox[k]) > 1e-4:
                issues.append(ValidationIssue(sample_index, "bbox_mismatch", f"shape {shape_idx} bbox mismatch on {k}"))
                break

        for p_idx, point in enumerate(points):
            if not (0.0 - EPS <= point["x"] <= 1.0 + EPS and 0.0 - EPS <= point["y"] <= 1.0 + EPS):
                issues.append(ValidationIssue(sample_index, "point_out_of_bounds", f"shape {shape_idx} point {p_idx} outside [0,1]"))
                break

        if shape["closed"] and len(points) < 3:
            issues.append(ValidationIssue(sample_index, "closed_shape_too_short", f"shape {shape_idx} marked closed but has <3 points"))
        if not shape["closed"] and len(points) < 2:
            issues.append(ValidationIssue(sample_index, "open_shape_too_short", f"shape {shape_idx} marked open but has <2 points"))

    relation_issue = _validate_relation(scene_spec.get("relation_type"), shapes)
    if relation_issue:
        issues.append(ValidationIssue(sample_index, "relation_violation", relation_issue))

    return issues


def validate_jsonl(path: Path, max_samples: int | None = None) -> Dict[str, Any]:
    issues: List[ValidationIssue] = []
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            total += 1
            sample = json.loads(line)
            issues.extend(validate_sample(sample, idx))

    counts = Counter(issue.code for issue in issues)
    return {
        "path": str(path),
        "validated_samples": total,
        "issue_count": len(issues),
        "issue_summary": dict(counts),
        "issues": issues,
    }
