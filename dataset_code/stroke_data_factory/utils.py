import math
from typing import Dict, List

from .schema import Point, ShapeSample


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def polygon_bbox(points: List[Point]) -> Dict[str, float]:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def rotate_point(px: float, py: float, cx: float, cy: float, angle: float) -> Point:
    dx = px - cx
    dy = py - cy
    ca = math.cos(angle)
    sa = math.sin(angle)
    return Point(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)


def bbox_center(bbox: Dict[str, float]) -> Point:
    return Point((bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2)


def translate_points(points: List[Point], tx: float, ty: float) -> List[Point]:
    return [Point(clamp(p.x + tx), clamp(p.y + ty)) for p in points]


def translate_shape(shape: ShapeSample, tx: float, ty: float) -> ShapeSample:
    points = translate_points(shape.points, tx, ty)
    bbox = polygon_bbox(points)
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=bbox,
        closed=shape.closed,
    )


def scale_shape(shape: ShapeSample, scale_x: float, scale_y: float) -> ShapeSample:
    center = bbox_center(shape.bbox)
    points = []
    for p in shape.points:
        dx = p.x - center.x
        dy = p.y - center.y
        points.append(Point(clamp(center.x + dx * scale_x), clamp(center.y + dy * scale_y)))
    bbox = polygon_bbox(points)
    return ShapeSample(
        shape_type=shape.shape_type,
        points=points,
        prompt_fragment=shape.prompt_fragment,
        bbox=bbox,
        closed=shape.closed,
    )


def resample_dense(points: List[Point], closed: bool, step: float = 0.015) -> List[Point]:
    """Interpolate sparse vertices into dense evenly-spaced points for smooth strokes."""
    if len(points) < 2:
        return points

    if closed:
        segments = list(zip(points, points[1:] + [points[0]]))
    else:
        segments = list(zip(points[:-1], points[1:]))

    result: List[Point] = []
    for a, b in segments:
        dist = math.hypot(b.x - a.x, b.y - a.y)
        n = max(1, int(dist / step))
        for i in range(n):
            t = i / n
            result.append(Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))

    return result
