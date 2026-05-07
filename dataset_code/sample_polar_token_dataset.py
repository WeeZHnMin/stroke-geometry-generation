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
from stroke_baseline.visualize import save_strokes_png


def build_tokenizer_from_sample(sample: dict) -> PolarActionTokenizer:
    metadata = sample.get("metadata", {})
    distance_buckets = tuple(float(v) for v in metadata["distance_buckets"])
    theta_bins = int(metadata["theta_bins"])
    return PolarActionTokenizer(PolarActionTokenizerConfig(distance_buckets=distance_buckets, theta_bins=theta_bins))


def stroke_mae(source: list[dict], decoded: list[dict]) -> dict[str, float]:
    usable = min(len(source), len(decoded))
    if usable == 0:
        return {"dx_mae": 0.0, "dy_mae": 0.0}
    dx_err = 0.0
    dy_err = 0.0
    for idx in range(usable):
        dx_err += abs(float(source[idx]["dx"]) - float(decoded[idx]["dx"]))
        dy_err += abs(float(source[idx]["dy"]) - float(decoded[idx]["dy"]))
    return {
        "dx_mae": dx_err / usable,
        "dy_mae": dy_err / usable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and visualize a sample from a polar-token dataset.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--json", type=str, default="viz_output/polar_dataset_sample.json")
    parser.add_argument("--png", type=str, default="viz_output/polar_dataset_sample.png")
    args = parser.parse_args()

    samples = read_jsonl(args.data)
    sample = samples[args.sample_index]
    tokenizer = build_tokenizer_from_sample(sample)
    tokens = [int(token) for token in sample["action_tokens"]]
    decoded_strokes = tokenizer.decode_tokens(tokens)
    errors = stroke_mae(sample["strokes"], decoded_strokes)

    payload = {
        "prompt": sample["prompt"],
        "sample_index": args.sample_index,
        "num_tokens": len(tokens),
        "tokens": tokens,
        "source_strokes": sample["strokes"],
        "decoded_strokes": decoded_strokes,
        "metrics": errors,
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_strokes_png(decoded_strokes, args.png, title=sample["prompt"])

    print(f"prompt={sample['prompt']}")
    print(f"num_tokens={len(tokens)}")
    print(f"dx_mae={errors['dx_mae']:.6f} dy_mae={errors['dy_mae']:.6f}")
    print(f"saved_json={args.json}")
    print(f"saved_png={args.png}")


if __name__ == "__main__":
    main()
