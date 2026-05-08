from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the observed compact Cartesian token vocabulary from a stroke dataset.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    samples = read_jsonl(args.data)
    if not samples:
        raise ValueError(f"No samples found in {args.data}")

    tokenizer = StrokeActionTokenizer(ActionTokenizerConfig())
    vocab: dict[int, dict] = {}
    counts: dict[int, int] = {}

    for sample in samples:
        canvas_size = float(sample.get("metadata", {}).get("canvas_size", 1.0))
        actions = tokenizer.strokes_to_cartesian_actions(sample["strokes"], canvas_size=canvas_size)
        for action in actions:
            raw_token_id = tokenizer.encode_action(action["x"], action["y"], action["pen_state"])
            counts[raw_token_id] = counts.get(raw_token_id, 0) + 1
            if raw_token_id not in vocab:
                x, y, pen_state = tokenizer.decode_action(raw_token_id)
                vocab[raw_token_id] = {
                    "raw_token_id": raw_token_id,
                    "x": float(x),
                    "y": float(y),
                    "pen_state": int(pen_state),
                }

    ordered = []
    for raw_token_id in sorted(vocab):
        item = dict(vocab[raw_token_id])
        item["count"] = counts[raw_token_id]
        ordered.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_path": str(Path(args.data)),
        "num_samples": len(samples),
        "observed_vocab_size": len(ordered),
        "theoretical_action_vocab_size": tokenizer.pad_id,
        "full_vocab_size_with_special_tokens": tokenizer.vocab_size,
        "tokens": ordered,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"num_samples={len(samples)}")
    print(f"observed_vocab_size={len(ordered)}")
    print(f"theoretical_action_vocab_size={tokenizer.pad_id}")
    print(f"full_vocab_size_with_special_tokens={tokenizer.vocab_size}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
