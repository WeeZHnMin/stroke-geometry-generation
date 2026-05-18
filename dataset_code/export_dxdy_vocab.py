"""Export a compact DxDy vocabulary from one or more JSONL stroke datasets.

The compact vocab contains only the (dx, dy) token IDs that actually appear
in the data, remapped to a contiguous range [0, N-1].  The resulting JSON
file can be loaded by CompactDxDyTokenizer.load() and reused across training
runs without re-scanning the data each time.

Usage:
    python dataset_code/export_dxdy_vocab.py \
        --data generated_data/bulk/stage1_foundation_shapes_v3_mixed_*.jsonl \
        --output generated_data/vocab/dxdy_vocab.json

    # multiple files:
    python dataset_code/export_dxdy_vocab.py \
        --data file1.jsonl file2.jsonl \
        --output generated_data/vocab/dxdy_vocab.json

    # adjust binning:
    python dataset_code/export_dxdy_vocab.py \
        --data ... --output ... --dx-bins 100 --dy-bins 100 --log-scale 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stroke_baseline.action_tokenizer import (
    CompactDxDyTokenizer,
    DxDyPairTokenizer,
    DxDyPairTokenizerConfig,
)

_KEEP_PEN = frozenset({"draw", "move"})


def scan_files(
    paths: list[Path],
    base: DxDyPairTokenizer,
    limit: int | None = None,
) -> tuple[dict[int, int], int]:
    """Return (raw_token_id -> count, total_samples)."""
    counts: dict[int, int] = {}
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if limit is not None and total >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for s in obj.get("strokes", []):
                    if s.get("pen_state") in _KEEP_PEN:
                        raw = base.encode_step(float(s["dx"]), float(s["dy"]))
                        counts[raw] = counts.get(raw, 0) + 1
                total += 1
        if limit is not None and total >= limit:
            break
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact DxDy vocabulary from JSONL stroke data."
    )
    parser.add_argument("--data", nargs="+", required=True,
        help="One or more JSONL dataset files (glob patterns resolved by shell).")
    parser.add_argument("--output", type=str, required=True,
        help="Output JSON file path.")
    parser.add_argument("--dx-bins", type=int, default=100)
    parser.add_argument("--dy-bins", type=int, default=100)
    parser.add_argument("--log-scale", type=float, default=10.0,
        help="log1p compression factor C (0 = uniform binning).")
    parser.add_argument("--limit", type=int, default=None,
        help="Max total samples to scan across all files.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.data]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"[warn] not found: {m}", file=sys.stderr)
        paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("No valid data files found.")

    base_cfg = DxDyPairTokenizerConfig(
        dx_bins=args.dx_bins,
        dy_bins=args.dy_bins,
        log_scale=args.log_scale,
    )
    base = DxDyPairTokenizer(base_cfg)

    print(f"Scanning {len(paths)} file(s)...", flush=True)
    counts, total_samples = scan_files(paths, base, limit=args.limit)

    tok = CompactDxDyTokenizer.from_raw_ids(base, list(counts.keys()))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)

    # enrich saved JSON with count info
    data = json.loads(output_path.read_text(encoding="utf-8"))
    compact_counts = {
        str(tok._raw_to_compact[raw]): cnt
        for raw, cnt in counts.items()
    }
    data["meta"] = {
        "total_samples": total_samples,
        "total_steps": sum(counts.values()),
        "theoretical_action_vocab": base.action_vocab_size,
        "observed_action_vocab": tok.action_vocab_size,
        "dx_bins": args.dx_bins,
        "dy_bins": args.dy_bins,
        "log_scale": args.log_scale,
        "data_files": [str(p) for p in paths],
    }
    data["compact_counts"] = compact_counts
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total_samples    = {total_samples:,}")
    print(f"total_steps      = {sum(counts.values()):,}")
    print(f"theoretical_vocab= {base.action_vocab_size:,}  (dx_bins * dy_bins)")
    print(f"observed_vocab   = {tok.action_vocab_size:,}")
    print(f"vocab_size (w/ special tokens) = {tok.vocab_size:,}")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
