"""Synthesize parameterised geometric shapes for HeteroKV pretraining smoke tests.

Each emitted sample is one line of JSONL:
    {"prompt": "<shape_name>", "strokes": [{"dx", "dy", "pen_state"}, ...]}

Shapes vary in position, size, rotation, subdivision count, and (for multi)
composition — exercising 2D RoPE, distance bias, length head, and the
move/draw pen split simultaneously.

All absolute canvas points stay inside the tokenizer's [-0.5, 0.5] window
(actually inside [-0.45, 0.45] with a safety margin), and per-step
|dx|+|dy| stays comfortably above one bin (~0.002 with bins=500).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# ---------- canvas / tokenizer constraints ----------
COORD_LO, COORD_HI = -0.5, 0.5
SAFE_MARGIN = 0.45            # absolute points must stay within [-SAFE_MARGIN, SAFE_MARGIN]
MIN_STEP_NORM = 0.005         # |dx|+|dy| floor — well above the 0.002 bin width at bins=500
MAX_STEPS_TOTAL = 170         # max stroke steps per sample (decoder uses 3*L positions)
MIN_STEPS_TOTAL = 4

# ---------- helpers ----------


def _rotate(x: float, y: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


def _interpolate(a: tuple[float, float], b: tuple[float, float], n_steps: int) -> list[tuple[float, float]]:
    """Subdivide segment a→b into n_steps equal pieces, returning the n_steps points
    *after* a (i.e. ending at b)."""
    return [
        (a[0] + (k / n_steps) * (b[0] - a[0]), a[1] + (k / n_steps) * (b[1] - a[1]))
        for k in range(1, n_steps + 1)
    ]


def points_to_strokes(points_with_pen: list[dict]) -> list[dict]:
    """Convert absolute-position waypoints to (dx, dy, pen_state) deltas.

    `points_with_pen` is a list of {"x", "y", "pen_state"} dicts; the deltas
    are computed relative to the previous absolute position (origin = (0, 0)
    before the first point)."""
    strokes: list[dict] = []
    px, py = 0.0, 0.0
    for p in points_with_pen:
        strokes.append({"dx": p["x"] - px, "dy": p["y"] - py, "pen_state": p["pen_state"]})
        px, py = p["x"], p["y"]
    return strokes


def _validate_strokes(strokes: list[dict]) -> bool:
    """Reject sequences that violate canvas / step-size guarantees."""
    if not (MIN_STEPS_TOTAL <= len(strokes) <= MAX_STEPS_TOTAL):
        return False
    x, y = 0.0, 0.0
    for step in strokes:
        if abs(step["dx"]) > 0.5 or abs(step["dy"]) > 0.5:
            return False
        # Step too tiny → encoded as bin 0 + bin 0 → no signal.
        if abs(step["dx"]) + abs(step["dy"]) < MIN_STEP_NORM and step["pen_state"] != "move":
            return False
        x += step["dx"]
        y += step["dy"]
        if not (COORD_LO <= x <= COORD_HI and COORD_LO <= y <= COORD_HI):
            return False
    return True


# ---------- shape generators (return list[dict] of absolute waypoints) ----------


def _circle(rng: random.Random) -> list[dict]:
    cx = rng.uniform(-0.25, 0.25)
    cy = rng.uniform(-0.25, 0.25)
    r_max = SAFE_MARGIN - max(abs(cx), abs(cy))
    r = rng.uniform(0.08, min(0.22, r_max))
    n = rng.choice([16, 24, 32, 48, 64])
    rot = rng.uniform(0.0, 2 * math.pi)
    points: list[dict] = []
    for k in range(n + 1):  # +1 closes the circle
        ang = rot + 2 * math.pi * k / n
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pen = "move" if k == 0 else "draw"
        points.append({"x": x, "y": y, "pen_state": pen})
    return points


def _rect(rng: random.Random) -> list[dict]:
    cx = rng.uniform(-0.2, 0.2)
    cy = rng.uniform(-0.2, 0.2)
    half_max_w = SAFE_MARGIN - abs(cx)
    half_max_h = SAFE_MARGIN - abs(cy)
    w = rng.uniform(0.1, min(0.4, 2 * half_max_w))
    h = rng.uniform(0.1, min(0.4, 2 * half_max_h))
    rot = rng.uniform(0.0, 2 * math.pi)
    spe = rng.choice([2, 3, 4, 6, 8])
    half_w, half_h = w / 2, h / 2
    corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners = [tuple(map(lambda v: v + 0, _rotate(x, y, rot))) for x, y in corners_local]
    corners = [(cx + x, cy + y) for x, y in corners]

    points: list[dict] = [{"x": corners[0][0], "y": corners[0][1], "pen_state": "move"}]
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        for x, y in _interpolate(a, b, spe):
            points.append({"x": x, "y": y, "pen_state": "draw"})
    return points


def _triangle(rng: random.Random) -> list[dict]:
    cx = rng.uniform(-0.25, 0.25)
    cy = rng.uniform(-0.25, 0.25)
    r_max = SAFE_MARGIN - max(abs(cx), abs(cy))
    r = rng.uniform(0.1, min(0.22, r_max))
    rot = rng.uniform(0.0, 2 * math.pi)
    spe = rng.choice([2, 3, 4, 6])
    verts = [
        (cx + r * math.cos(rot + 2 * math.pi * k / 3), cy + r * math.sin(rot + 2 * math.pi * k / 3))
        for k in range(3)
    ]
    points: list[dict] = [{"x": verts[0][0], "y": verts[0][1], "pen_state": "move"}]
    for i in range(3):
        a, b = verts[i], verts[(i + 1) % 3]
        for x, y in _interpolate(a, b, spe):
            points.append({"x": x, "y": y, "pen_state": "draw"})
    return points


def _polyline(rng: random.Random) -> list[dict]:
    n_segs = rng.randint(4, 24)
    x = rng.uniform(-0.3, 0.3)
    y = rng.uniform(-0.3, 0.3)
    points: list[dict] = [{"x": x, "y": y, "pen_state": "move"}]
    for _ in range(n_segs):
        for _attempt in range(20):
            ang = rng.uniform(0.0, 2 * math.pi)
            step = rng.uniform(0.03, 0.10)
            nx, ny = x + step * math.cos(ang), y + step * math.sin(ang)
            if -SAFE_MARGIN <= nx <= SAFE_MARGIN and -SAFE_MARGIN <= ny <= SAFE_MARGIN:
                x, y = nx, ny
                points.append({"x": x, "y": y, "pen_state": "draw"})
                break
    return points


def _spiral(rng: random.Random) -> list[dict]:
    cx = rng.uniform(-0.2, 0.2)
    cy = rng.uniform(-0.2, 0.2)
    margin = SAFE_MARGIN - max(abs(cx), abs(cy))
    r0 = rng.uniform(0.02, 0.08)
    r1 = rng.uniform(0.12, min(0.22, max(margin, 0.13)))
    if r1 <= r0:
        r1 = r0 + 0.05
    n_turns = rng.choice([1, 2, 3])
    n_steps = rng.choice([24, 36, 48, 64])
    rot = rng.uniform(0.0, 2 * math.pi)
    direction = rng.choice([-1, 1])
    points: list[dict] = []
    for k in range(n_steps + 1):
        t = k / n_steps
        r = r0 + (r1 - r0) * t
        ang = rot + direction * 2 * math.pi * n_turns * t
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pen = "move" if k == 0 else "draw"
        points.append({"x": x, "y": y, "pen_state": pen})
    return points


def _multi(rng: random.Random) -> list[dict]:
    """2~3 simple shapes laid down with a `move` between them. Composition
    forces the model to recover from a `move` jump (pen-state matters)."""
    k = rng.choice([2, 3])
    shape_pool = [_circle, _rect, _triangle]
    out: list[dict] = []
    used_steps = 0
    for shape_idx in range(k):
        gen = rng.choice(shape_pool)
        pts = gen(rng)
        if used_steps + len(pts) > MAX_STEPS_TOTAL - 2:
            break
        if shape_idx == 0:
            out.extend(pts)
        else:
            # First point of subsequent shape stays a `move` (pen up + jump).
            out.extend([{**pts[0], "pen_state": "move"}, *pts[1:]])
        used_steps += len(pts)
    if not out:
        return _circle(rng)
    return out


GENERATORS: list[tuple[str, callable, float]] = [
    ("circle",   _circle,   0.20),
    ("rect",     _rect,     0.20),
    ("triangle", _triangle, 0.15),
    ("polyline", _polyline, 0.20),
    ("spiral",   _spiral,   0.15),
    ("multi",    _multi,    0.10),
]


# ---------- main loop ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=10,
        help="Per-sample retry budget when a generated shape fails validation",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [n for n, _, _ in GENERATORS]
    fns = [f for _, f, _ in GENERATORS]
    weights = [w for _, _, w in GENERATORS]

    counts = {n: 0 for n in names}
    rejected = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        while written < args.num_samples:
            idx = rng.choices(range(len(fns)), weights=weights)[0]
            name, gen = names[idx], fns[idx]
            ok_strokes = None
            for _ in range(args.max_attempts_per_sample):
                try:
                    points = gen(rng)
                except Exception:
                    rejected += 1
                    continue
                strokes = points_to_strokes(points)
                if _validate_strokes(strokes):
                    ok_strokes = strokes
                    break
                rejected += 1
            if ok_strokes is None:
                continue
            f.write(json.dumps({"prompt": name, "strokes": ok_strokes}, ensure_ascii=False) + "\n")
            counts[name] += 1
            written += 1
            if written % 2000 == 0:
                print(f"written={written}/{args.num_samples} rejected={rejected}", flush=True)

    print()
    print(f"output: {out_path}")
    print(f"total samples: {written}")
    print(f"total rejected (validation): {rejected}")
    print("per-shape counts:")
    for n in names:
        print(f"  {n:10s} {counts[n]}")


if __name__ == "__main__":
    main()
