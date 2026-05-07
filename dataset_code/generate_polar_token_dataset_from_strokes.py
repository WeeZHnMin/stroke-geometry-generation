from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stroke_baseline.dataset import read_jsonl
from stroke_baseline.polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig


def build_distance_buckets(distance_step: float, distance_max: float) -> tuple[float, ...]:
    if distance_step <= 0.0:
        raise ValueError("distance_step must be positive")
    if distance_max < distance_step:
        raise ValueError("distance_max must be >= distance_step")
    count = int(round(distance_max / distance_step))
    return tuple(round((idx + 1) * distance_step, 10) for idx in range(count))


def quantize_theta_id(dx: float, dy: float, theta_bins: int) -> int:
    theta = math.atan2(dy, dx) % (2.0 * math.pi)
    return int(round(theta / (2.0 * math.pi) * theta_bins)) % theta_bins


def quantize_distance_id(distance: float, distance_step: float, max_distance_id: int) -> int:
    distance_id = int(round(distance / distance_step)) - 1
    return max(0, min(max_distance_id, distance_id))


def angle_error_deg(theta_a: float, theta_b: float) -> float:
    delta = (theta_a - theta_b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta) * 180.0 / math.pi


def convert_sample(sample: dict, tokenizer: PolarActionTokenizer, distance_step: float) -> dict:
    actions: list[dict] = []
    tokens: list[int] = []
    decoded_strokes: list[dict] = []

    dx_abs_err_sum = 0.0
    dy_abs_err_sum = 0.0
    dist_abs_err_sum = 0.0
    theta_abs_err_sum = 0.0

    max_distance_id = len(tokenizer.distance_buckets) - 1
    for step in sample["strokes"]:
        dx = float(step["dx"])
        dy = float(step["dy"])
        pen_state = str(step["pen_state"])

        distance = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) % (2.0 * math.pi)
        distance_id = quantize_distance_id(distance, distance_step, max_distance_id)
        theta_id = quantize_theta_id(dx, dy, tokenizer.theta_bins)

        token = tokenizer.encode_action(distance_id, theta_id, pen_state)
        decoded = tokenizer.token_to_stroke(token)
        decoded_distance = math.hypot(decoded["dx"], decoded["dy"])
        decoded_theta = math.atan2(decoded["dy"], decoded["dx"]) % (2.0 * math.pi)

        dx_abs_err_sum += abs(decoded["dx"] - dx)
        dy_abs_err_sum += abs(decoded["dy"] - dy)
        dist_abs_err_sum += abs(decoded_distance - distance)
        theta_abs_err_sum += angle_error_deg(decoded_theta, theta)

        tokens.append(token)
        decoded_strokes.append(decoded)
        actions.append(
            {
                "distance_id": distance_id,
                "distance": tokenizer.distance_buckets[distance_id],
                "theta_id": theta_id,
                "theta": 2.0 * math.pi * theta_id / tokenizer.theta_bins,
                "pen_state": pen_state,
                "decoded_dx": decoded["dx"],
                "decoded_dy": decoded["dy"],
                "source_dx": dx,
                "source_dy": dy,
            }
        )

    num_steps = max(len(actions), 1)
    metadata = dict(sample.get("metadata", {}))
    metadata.update(
        {
            "polar_distance_step": distance_step,
            "distance_buckets": list(tokenizer.distance_buckets),
            "theta_bins": tokenizer.theta_bins,
            "num_polar_actions": len(tokens),
            "polar_dx_mae": dx_abs_err_sum / num_steps,
            "polar_dy_mae": dy_abs_err_sum / num_steps,
            "polar_distance_mae": dist_abs_err_sum / num_steps,
            "polar_theta_mae_deg": theta_abs_err_sum / num_steps,
        }
    )

    scene_spec = dict(sample.get("scene_spec", {}))
    scene_spec.update(
        {
            "distance_buckets": list(tokenizer.distance_buckets),
            "theta_bins": tokenizer.theta_bins,
            "polar_distance_step": distance_step,
            "polar_step_format": ["distance_id", "theta_id", "pen_state"],
        }
    )

    return {
        **sample,
        "scene_spec": scene_spec,
        "polar_actions": actions,
        "action_tokens": tokens,
        "decoded_polar_strokes": decoded_strokes,
        "metadata": metadata,
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_report(samples: list[dict]) -> None:
    lengths = [len(sample["action_tokens"]) for sample in samples]
    dx_mae = [float(sample["metadata"]["polar_dx_mae"]) for sample in samples]
    dy_mae = [float(sample["metadata"]["polar_dy_mae"]) for sample in samples]
    dist_mae = [float(sample["metadata"]["polar_distance_mae"]) for sample in samples]
    theta_mae = [float(sample["metadata"]["polar_theta_mae_deg"]) for sample in samples]
    print(f"samples={len(samples)}")
    print(f"action_len min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"dx_mae avg={sum(dx_mae)/len(dx_mae):.6f} max={max(dx_mae):.6f}")
    print(f"dy_mae avg={sum(dy_mae)/len(dy_mae):.6f} max={max(dy_mae):.6f}")
    print(f"distance_mae avg={sum(dist_mae)/len(dist_mae):.6f} max={max(dist_mae):.6f}")
    print(f"theta_mae_deg avg={sum(theta_mae)/len(theta_mae):.6f} max={max(theta_mae):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert stroke JSONL into polar single-token action dataset.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--distance-step", type=float, default=0.01)
    parser.add_argument("--distance-max", type=float, default=0.71)
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    distance_buckets = build_distance_buckets(args.distance_step, args.distance_max)
    tokenizer = PolarActionTokenizer(
        PolarActionTokenizerConfig(distance_buckets=distance_buckets, theta_bins=args.theta_bins)
    )
    raw_samples = read_jsonl(args.input, limit=args.limit)
    samples = [convert_sample(sample, tokenizer, args.distance_step) for sample in raw_samples]
    save_jsonl(samples, Path(args.output))
    print_report(samples)
    print(f"saved: {args.output}")
    print(f"distance_buckets={len(distance_buckets)} theta_bins={args.theta_bins} vocab={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
