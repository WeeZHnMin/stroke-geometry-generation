from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "generated_data" / "rotation_points"


def rotate_point(x: float, y: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y, s * x + c * y


def sample_point_square(rng: random.Random, coord_min: float, coord_max: float) -> tuple[float, float]:
    return rng.uniform(coord_min, coord_max), rng.uniform(coord_min, coord_max)


def sample_point_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(rng.random())
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(phi), r * math.sin(phi)


def build_sample(
    rng: random.Random,
    *,
    point_mode: str,
    coord_min: float,
    coord_max: float,
) -> dict:
    if point_mode == "square":
        x, y = sample_point_square(rng, coord_min, coord_max)
    elif point_mode == "disk":
        radius = max(abs(coord_min), abs(coord_max))
        x, y = sample_point_disk(rng, radius)
    else:
        raise ValueError(f"unsupported point_mode: {point_mode}")

    theta = rng.uniform(0.0, 2.0 * math.pi)
    c = math.cos(theta)
    s = math.sin(theta)
    xr, yr = rotate_point(x, y, theta)
    return {
        "input": {
            "x": x,
            "y": y,
            "theta": theta,
        },
        "rotation": {
            "cos_theta": c,
            "sin_theta": s,
            "matrix": [[c, -s], [s, c]],
        },
        "target": {
            "x": xr,
            "y": yr,
        },
        "metadata": {
            "formula": "x_rot = cos(theta)*x - sin(theta)*y; y_rot = sin(theta)*x + cos(theta)*y",
            "theta_range": [0.0, 2.0 * math.pi],
            "point_mode": point_mode,
            "coord_range": [coord_min, coord_max],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point rotation pairs: ([x,y], theta) -> rotated [x,y].")
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-mode", type=str, default="square", choices=["square", "disk"])
    parser.add_argument("--coord-min", type=float, default=-1.0)
    parser.add_argument("--coord-max", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=20_000)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.coord_max <= args.coord_min:
        raise ValueError("--coord-max must be greater than --coord-min")

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"rotation_points_{args.num_samples}_{timestamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(1, args.num_samples + 1):
            sample = build_sample(
                rng,
                point_mode=args.point_mode,
                coord_min=args.coord_min,
                coord_max=args.coord_max,
            )
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(f"written={idx}/{args.num_samples}", flush=True)

    print(f"wrote {args.num_samples} samples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
