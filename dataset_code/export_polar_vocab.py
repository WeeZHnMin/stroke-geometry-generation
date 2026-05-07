from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stroke_baseline.dataset import read_jsonl
from stroke_baseline.polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig


def build_tokenizer_from_sample(sample: dict) -> PolarActionTokenizer:
    metadata = sample.get("metadata", {})
    distance_buckets = tuple(float(v) for v in metadata["distance_buckets"])
    theta_bins = int(metadata["theta_bins"])
    return PolarActionTokenizer(PolarActionTokenizerConfig(distance_buckets=distance_buckets, theta_bins=theta_bins))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the observed polar token vocabulary from a dataset.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    samples = read_jsonl(args.data)
    if not samples:
        raise ValueError(f"No samples found in {args.data}")

    tokenizer = build_tokenizer_from_sample(samples[0])
    vocab: dict[int, dict] = {}
    counts: dict[int, int] = {}

    for sample in samples:
        for token in sample["action_tokens"]:
            token_id = int(token)
            counts[token_id] = counts.get(token_id, 0) + 1
            if token_id not in vocab:
                action = tokenizer.decode_action(token_id)
                vocab[token_id] = {
                    "token_id": token_id,
                    "distance_id": int(action["distance_id"]),
                    "distance": float(action["distance"]),
                    "theta_id": int(action["theta_id"]),
                    "theta": float(action["theta"]),
                    "pen_state": str(action["pen_state"]),
                }

    ordered = []
    for token_id in sorted(vocab):
        item = dict(vocab[token_id])
        item["count"] = counts[token_id]
        ordered.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_path": str(Path(args.data)),
        "num_samples": len(samples),
        "observed_vocab_size": len(ordered),
        "theoretical_action_vocab_size": tokenizer.action_vocab_size,
        "full_vocab_size_with_special_tokens": tokenizer.vocab_size,
        "distance_buckets": list(tokenizer.distance_buckets),
        "theta_bins": tokenizer.theta_bins,
        "tokens": ordered,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"num_samples={len(samples)}")
    print(f"observed_vocab_size={len(ordered)}")
    print(f"theoretical_action_vocab_size={tokenizer.action_vocab_size}")
    print(f"full_vocab_size_with_special_tokens={tokenizer.vocab_size}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
