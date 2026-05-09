from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from itertools import islice
from pathlib import Path

from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.dataset import PEN_TO_ID


_WORKER_TOKENIZER: StrokeActionTokenizer | None = None
_WORKER_MAX_ACTION_LEN = 384


def _normalize_pen_state(pen_state: str) -> str:
    return "end_all" if pen_state == "end" else pen_state


def _build_coords_from_strokes(strokes: list[dict], max_action_len: int) -> list[list[float]]:
    coords = [[0.0, 0.0] for _ in range(max_action_len)]
    x = 0.0
    y = 0.0
    pos = 0
    for step in strokes:
        if pos >= max_action_len:
            break
        dx = float(step["dx"])
        dy = float(step["dy"])
        for _ in range(3):
            if pos >= max_action_len:
                break
            coords[pos] = [x, y]
            pos += 1
        x += dx
        y += dy
    return coords


def _init_worker(cfg: dict, max_action_len: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_ACTION_LEN
    _WORKER_TOKENIZER = StrokeActionTokenizer(ActionTokenizerConfig(**cfg))
    _WORKER_MAX_ACTION_LEN = max_action_len


def _materialize_sample(sample: dict) -> dict:
    assert _WORKER_TOKENIZER is not None
    tokenizer = _WORKER_TOKENIZER
    max_action_len = _WORKER_MAX_ACTION_LEN

    strokes = []
    for step in sample.get("strokes", []):
        strokes.append(
            {
                "dx": float(step["dx"]),
                "dy": float(step["dy"]),
                "pen_state": _normalize_pen_state(str(step["pen_state"])),
            }
        )

    tokens = tokenizer.encode_strokes(strokes)
    seq_len = min(len(tokens), max_action_len)
    actual = tokens[:seq_len]

    decoder_input = [tokenizer.pad_id] * max_action_len
    target = [-100] * max_action_len
    target_mask = [False] * max_action_len

    decoder_input[0] = tokenizer.start_id
    if seq_len > 1:
        decoder_input[1:seq_len] = actual[:-1]
    target[:seq_len] = actual
    for i in range(seq_len):
        target_mask[i] = True

    return {
        **sample,
        "strokes": strokes,
        "coords": _build_coords_from_strokes(strokes, max_action_len),
        "decoder_input_ids": decoder_input,
        "target_ids": target,
        "target_mask": target_mask,
        "length": int(seq_len),
        "metadata": {
            **sample.get("metadata", {}),
            "dx_vocab_size": tokenizer.bins,
            "dy_vocab_size": tokenizer.bins,
            "pen_vocab_size": len(PEN_TO_ID),
            "max_action_len": max_action_len,
            "output_format": "dxdy_pen_with_coords",
        },
    }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _batched(iterable, batch_size: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dx/dy/pen training-ready dataset with absolute coords.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-action-len", type=int, default=384)
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--min-value", type=float, default=-0.5)
    parser.add_argument("--max-value", type=float, default=0.5)
    parser.add_argument("--draw-min-value", type=float, default=-0.5)
    parser.add_argument("--draw-max-value", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--write-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = _count_lines(data_path)
    cfg = {
        "bins": args.bins,
        "min_value": args.min_value,
        "max_value": args.max_value,
        "draw_min_value": args.draw_min_value,
        "draw_max_value": args.draw_max_value,
    }

    processed = 0
    with output_path.open("w", encoding="utf-8") as f, mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(cfg, args.max_action_len),
    ) as pool:
        for batch in _batched(_iter_jsonl(data_path), args.write_batch_size):
            for item in pool.imap(_materialize_sample, batch, chunksize=args.chunk_size):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                processed += 1
                if processed % args.progress_every == 0 or processed == total:
                    print(f"processed={processed}/{total}", flush=True)

    print(f"samples={processed}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
