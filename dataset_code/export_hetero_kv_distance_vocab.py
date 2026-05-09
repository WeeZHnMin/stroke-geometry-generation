from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.dataset import PEN_TO_ID, read_jsonl


def _normalize_pen_state(pen_state: str) -> str:
    return "end_all" if pen_state == "end" else pen_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the observed dx/dy/pen vocabulary from a dataset.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--min-value", type=float, default=-0.5)
    parser.add_argument("--max-value", type=float, default=0.5)
    parser.add_argument("--draw-min-value", type=float, default=-0.5)
    parser.add_argument("--draw-max-value", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = StrokeActionTokenizer(
        ActionTokenizerConfig(
            bins=args.bins,
            min_value=args.min_value,
            max_value=args.max_value,
            draw_min_value=args.draw_min_value,
            draw_max_value=args.draw_max_value,
        )
    )
    samples = read_jsonl(args.data)
    counter: Counter[int] = Counter()
    dx_counter: Counter[int] = Counter()
    dy_counter: Counter[int] = Counter()
    pen_counter: Counter[int] = Counter()

    for sample in samples:
        for step in sample.get("strokes", []):
            tokens = tokenizer.encode_step(float(step["dx"]), float(step["dy"]), _normalize_pen_state(str(step["pen_state"])))
            dx_id, dy_id, pen_id = tokens
            counter[dx_id] += 1
            counter[dy_id] += 1
            counter[pen_id] += 1
            dx_counter[dx_id] += 1
            dy_counter[dy_id] += 1
            pen_counter[pen_id] += 1

    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    payload = {
        "data_path": args.data,
        "num_samples": len(samples),
        "observed_vocab_size": len(ordered),
        "theoretical_dx_vocab_size": tokenizer.bins,
        "theoretical_dy_vocab_size": tokenizer.bins,
        "theoretical_pen_vocab_size": len(PEN_TO_ID),
        "full_vocab_size_with_special_tokens": tokenizer.vocab_size,
        "special_tokens": {
            "start_id": tokenizer.start_id,
            "pad_id": tokenizer.pad_id,
        },
        "tokens": [
            {
                "token_id": token_id,
                "kind": "dx" if token_id < tokenizer.dy_offset else "dy" if token_id < tokenizer.pen_offset else "pen",
                "count": count,
            }
            for token_id, count in ordered
        ],
        "dx_observed": len(dx_counter),
        "dy_observed": len(dy_counter),
        "pen_observed": len(pen_counter),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"observed_vocab_size={payload['observed_vocab_size']}")
    print(f"dx_observed={payload['dx_observed']} dy_observed={payload['dy_observed']} pen_observed={payload['pen_observed']}")
    print(f"full_vocab_size_with_special_tokens={payload['full_vocab_size_with_special_tokens']}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
