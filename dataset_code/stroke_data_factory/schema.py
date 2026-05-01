from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Point:
    x: float
    y: float


@dataclass
class StrokeStep:
    dx: float
    dy: float
    pen_state: str


@dataclass
class ShapeSample:
    shape_type: str
    points: List[Point]
    prompt_fragment: str
    bbox: Dict[str, float]
    closed: bool
