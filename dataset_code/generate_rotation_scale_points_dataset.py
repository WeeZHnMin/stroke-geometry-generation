from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "generated_data" / "rotation_scale_points"


def rotate_point(x: float, y: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y, s * x + c * y


def sample_point(rng: random.Random, coord_min: float, coord_max: float) -> tuple[float, float]:
    return rng.uniform(coord_min, coord_max), rng.uniform(coord_min, coord_max)


def build_sample(
    rng: random.Random,
    *,
    coord_min: float,
    coord_max: float,
    scale_min: float,
    scale_max: float,
) -> dict:
    x, y = sample_point(rng, coord_min, coord_max)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    log_scale = rng.uniform(math.log(scale_min), math.log(scale_max))
    scale = math.exp(log_scale)
    xr, yr = rotate_point(x, y, theta)
    xr *= scale
    yr *= scale
    c = math.cos(theta)
    s = math.sin(theta)
    return {
        "input": {
            "x": x,
            "y": y,
            "theta": theta,
            "scale": scale,
            "log_scale": log_scale,
        },
        "transform": {
            "rotation_matrix": [[c, -s], [s, c]],
            "scale": scale,
            "matrix": [[scale * c, -scale * s], [scale * s, scale * c]],
        },
        "target": {
            "x": xr,
            "y": yr,
        },
        "metadata": {
            "formula": "target = scale * R(theta) @ [x, y]",
            "theta_range": [0.0, 2.0 * math.pi],
            "scale_range": [scale_min, scale_max],
            "scale_sampling": "log_uniform",
            "coord_range": [coord_min, coord_max],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point rotation+scale pairs.")
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord-min", type=float, default=-1.0)
    parser.add_argument("--coord-max", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--progress-every", type=int, default=20_000)
    args = parser.parse_args()

    if args.coord_max <= args.coord_min:
        raise ValueError("--coord-max must be greater than --coord-min")
    if args.scale_min <= 0 or args.scale_max <= args.scale_min:
        raise ValueError("Require 0 < --scale-min < --scale-max")

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"rotation_scale_points_{args.num_samples}_{timestamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(1, args.num_samples + 1):
            sample = build_sample(
                rng,
                coord_min=args.coord_min,
                coord_max=args.coord_max,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
            )
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(f"written={idx}/{args.num_samples}", flush=True)

    print(f"wrote {args.num_samples} samples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
