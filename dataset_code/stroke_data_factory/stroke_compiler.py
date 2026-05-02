from typing import List, Tuple

from .constants import DRAW, END_ALL, END_SHAPE, MOVE
from .schema import Point, ShapeSample, StrokeStep
from .utils import resample_dense


def shape_to_strokes(points: List[Point], start_from: Point, closed: bool = True, dense: bool = True) -> Tuple[List[StrokeStep], Point]:
    """Convert shape points to stroke steps. If dense=True, interpolates sparse vertices."""
    if dense:
        points = resample_dense(points, closed)

    steps: List[StrokeStep] = []
    if closed and (abs(points[0].x - points[-1].x) > 1e-9 or abs(points[0].y - points[-1].y) > 1e-9):
        work = points + [points[0]]
    else:
        work = points

    prev = start_from
    for i, p in enumerate(work):
        dx = p.x - prev.x
        dy = p.y - prev.y
        if i == 0:
            pen_state = MOVE
        elif i < len(work) - 1:
            pen_state = DRAW
        else:
            pen_state = END_SHAPE
        steps.append(StrokeStep(dx=dx, dy=dy, pen_state=pen_state))
        prev = p
    return steps, prev


def compile_strokes(shapes: List[ShapeSample]) -> List[StrokeStep]:
    all_steps: List[StrokeStep] = []
    cursor = Point(0.0, 0.0)
    for idx, shape in enumerate(shapes):
        shape_steps, cursor = shape_to_strokes(shape.points, start_from=cursor, closed=shape.closed)
        if idx == len(shapes) - 1:
            shape_steps[-1].pen_state = END_ALL
        all_steps.extend(shape_steps)
    return all_steps
