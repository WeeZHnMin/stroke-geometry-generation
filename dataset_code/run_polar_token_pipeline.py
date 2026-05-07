from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_code"))

from dataset_code.generate_quantized_grid_dataset import DEFAULT_SHAPES, sample_scene
from dataset_code.generate_polar_token_dataset_from_strokes import build_distance_buckets, convert_sample, save_jsonl
from stroke_baseline.dataset import read_jsonl
from stroke_baseline.polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_distance_string(distance_step: float, distance_max: float) -> str:
    buckets = build_distance_buckets(distance_step, distance_max)
    return ",".join(f"{value:g}" for value in buckets)


def generate_unique_raw_samples(
    *,
    num_samples: int,
    seed: int,
    shapes: list[str],
    canvas_size: float,
    dense_step: float,
    max_attempts_per_sample: int,
) -> list[dict]:
    rng = random.Random(seed)
    seen: set[str] = set()
    samples: list[dict] = []
    attempts = 0
    while len(samples) < num_samples:
        attempts += 1
        if attempts > num_samples * max_attempts_per_sample:
            raise RuntimeError(f"too many duplicate raw samples while building {num_samples} unique examples")
        sample = sample_scene(rng, shapes, canvas_size=canvas_size, dense_step=dense_step)
        signature = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        samples.append(sample)
    return samples


def run_python_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot pipeline: generate unique raw data, convert to polar tokens, optionally train and sample.")
    parser.add_argument("--num-samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-size", type=float, default=6.0)
    parser.add_argument("--dense-step", type=float, default=0.08)
    parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--max-attempts-per-sample", type=int, default=50)

    parser.add_argument("--raw-output", type=str, default="generated_data/raw/raw_unique_15000.jsonl")
    parser.add_argument("--polar-output", type=str, default="generated_data/polar_tokens/polar_tokens_unique_15000.jsonl")

    parser.add_argument("--distance-step", type=float, default=0.01)
    parser.add_argument("--distance-max", type=float, default=0.71)
    parser.add_argument("--theta-bins", type=int, default=360)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-output-dir", type=str, default="runs/polar_tokens_unique_15000")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-action-len", type=int, default=192)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--sample-json", type=str, default="viz_output/polar_tokens_pipeline_sample.json")
    parser.add_argument("--sample-png", type=str, default="viz_output/polar_tokens_pipeline_sample.png")
    parser.add_argument("--sample-max-steps", type=int, default=192)
    args = parser.parse_args()

    shapes = parse_list(args.shapes)
    unique_raw = generate_unique_raw_samples(
        num_samples=args.num_samples,
        seed=args.seed,
        shapes=shapes,
        canvas_size=args.canvas_size,
        dense_step=args.dense_step,
        max_attempts_per_sample=args.max_attempts_per_sample,
    )
    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(unique_raw, raw_output)

    distance_buckets = build_distance_buckets(args.distance_step, args.distance_max)
    tokenizer = PolarActionTokenizer(
        PolarActionTokenizerConfig(distance_buckets=distance_buckets, theta_bins=args.theta_bins)
    )
    polar_samples = [convert_sample(sample, tokenizer, args.distance_step) for sample in unique_raw]
    polar_output = Path(args.polar_output)
    save_jsonl(polar_samples, polar_output)

    print(f"raw_samples={len(unique_raw)} raw_output={raw_output}")
    print(f"polar_samples={len(polar_samples)} polar_output={polar_output}")
    print(f"distance_buckets={len(distance_buckets)} theta_bins={args.theta_bins} vocab={tokenizer.vocab_size}")

    if args.train:
        distances = make_distance_string(args.distance_step, args.distance_max)
        train_args = [
            "--data",
            str(polar_output),
            "--output-dir",
            args.train_output_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-action-len",
            str(args.max_action_len),
            "--max-text-len",
            str(args.max_text_len),
            "--d-model",
            str(args.d_model),
            "--n-heads",
            str(args.n_heads),
            "--decoder-layers",
            str(args.decoder_layers),
            "--dropout",
            str(args.dropout),
            "--distances",
            distances,
            "--theta-bins",
            str(args.theta_bins),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--val-ratio",
            str(args.val_ratio),
            "--log-every",
            str(args.log_every),
        ]
        run_python_module("stroke_baseline.train_polar_tokens", train_args)

        if args.sample:
            sample_prompt = args.sample_prompt or unique_raw[0]["prompt"]
            sample_args = [
                "--checkpoint",
                str(Path(args.train_output_dir) / "checkpoint.pt"),
                "--prompt",
                sample_prompt,
                "--max-steps",
                str(args.sample_max_steps),
                "--png",
                args.sample_png,
                "--json",
                args.sample_json,
            ]
            run_python_module("stroke_baseline.sample_polar_tokens", sample_args)


if __name__ == "__main__":
    main()
