import math
from typing import List, Tuple

from .constants import DRAW, END_ALL, END_SHAPE, MOVE
from .schema import Point, ShapeSample, StrokeStep
from .utils import resample_dense


def append_move_steps(
    steps: List[StrokeStep],
    start: Point,
    target: Point,
    dense_step: float,
) -> Point:
    """Move the pen-up cursor to target using small dx/dy increments."""
    dx = target.x - start.x
    dy = target.y - start.y
    dist = math.hypot(dx, dy)
    n = max(1, math.ceil(dist / dense_step))
    prev = start
    for i in range(1, n + 1):
        t = i / n
        p = Point(start.x + dx * t, start.y + dy * t)
        steps.append(StrokeStep(dx=p.x - prev.x, dy=p.y - prev.y, pen_state=MOVE))
        prev = p
    return prev


def shape_to_strokes(
    points: List[Point],
    start_from: Point,
    closed: bool = True,
    dense: bool = True,
    dense_step: float = 0.015,
) -> Tuple[List[StrokeStep], Point]:
    """Convert shape points to stroke steps. If dense=True, interpolates sparse vertices."""
    if dense:
        points = resample_dense(points, closed, step=dense_step)

    steps: List[StrokeStep] = []
    if closed and (abs(points[0].x - points[-1].x) > 1e-9 or abs(points[0].y - points[-1].y) > 1e-9):
        work = points + [points[0]]
    else:
        work = points

    prev = append_move_steps(steps, start_from, work[0], dense_step)
    for i, p in enumerate(work[1:], start=1):
        dx = p.x - prev.x
        dy = p.y - prev.y
        if i < len(work) - 1:
            pen_state = DRAW
        else:
            pen_state = END_SHAPE
        steps.append(StrokeStep(dx=dx, dy=dy, pen_state=pen_state))
        prev = p
    return steps, prev


def compile_strokes(shapes: List[ShapeSample], dense_step: float = 0.015) -> List[StrokeStep]:
    all_steps: List[StrokeStep] = []
    cursor = Point(0.0, 0.0)
    for idx, shape in enumerate(shapes):
        shape_steps, cursor = shape_to_strokes(
            shape.points,
            start_from=cursor,
            closed=shape.closed,
            dense_step=dense_step,
        )
        if idx == len(shapes) - 1:
            shape_steps[-1].pen_state = END_ALL
        all_steps.extend(shape_steps)
    return all_steps
