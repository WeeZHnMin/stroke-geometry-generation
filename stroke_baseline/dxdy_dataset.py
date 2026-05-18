"""Dataset for DxDyDecoder pretraining.

Each JSONL sample contains a "strokes" list with dx/dy/pen_state fields.
We encode each (draw/move) step as a single paired token and build a
next-token-prediction sequence:

    input:   [BOS, t0, t1, t2, ..., t_{N-1}]
    target:  [t0,  t1, t2, ..., t_{N-1}, PAD]   (PAD positions masked with -100)
"""

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .action_tokenizer import (
    CompactDxDyTokenizer,
    DxDyPairTokenizer,
    DxDyPairTokenizerConfig,
)

_KEEP_PEN = frozenset({"draw", "move"})


def _extract_steps(strokes: list[dict]) -> list[tuple[float, float]]:
    return [
        (float(s["dx"]), float(s["dy"]))
        for s in strokes
        if s.get("pen_state") in _KEEP_PEN
    ]


class DxDyJsonlDataset(Dataset):
    """Loads a JSONL stroke dataset and tokenises each sequence.

    On first construction the full file is scanned to build the compact
    vocabulary from observed (dx, dy) pairs.  Pass a pre-built tokenizer
    via `tokenizer` to skip this scan (e.g. for the validation split that
    shares the same vocab as training).
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        tokenizer: "DxDyPairTokenizer | CompactDxDyTokenizer | None" = None,
        base_cfg: DxDyPairTokenizerConfig | None = None,
        max_seq_len: int = 256,
        limit: int | None = None,
        min_steps: int = 2,
    ):
        self.max_seq_len = max_seq_len
        self.min_steps = min_steps

        raw_samples: list[list[tuple[float, float]]] = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                steps = _extract_steps(obj.get("strokes", []))
                if len(steps) >= min_steps:
                    raw_samples.append(steps)

        if tokenizer is None:
            base = DxDyPairTokenizer(base_cfg or DxDyPairTokenizerConfig())
            observed: list[int] = []
            for steps in raw_samples:
                for dx, dy in steps:
                    observed.append(base.encode_step(dx, dy))
            tokenizer = CompactDxDyTokenizer.from_raw_ids(base, observed)

        self.tokenizer = tokenizer
        self.samples: list[torch.Tensor] = []
        for steps in raw_samples:
            ids = [tokenizer.encode_step(dx, dy) for dx, dy in steps]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        tok = self.tokenizer
        bos = tok.bos_id
        pad = tok.pad_id

        # truncate to max_seq_len action tokens (before adding BOS)
        ids = ids[: self.max_seq_len]

        # input:  [BOS, t0, t1, ..., t_{N-1}]         length = N+1
        # target: [t0,  t1, ..., t_{N-1}, -100]       -100 at the end (no next token)
        input_ids = torch.cat([torch.tensor([bos], dtype=torch.long), ids])
        target_ids = torch.cat([ids, torch.tensor([-100], dtype=torch.long)])

        # pad to max_seq_len + 1
        total = self.max_seq_len + 1
        pad_len = total - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "target_ids": target_ids}


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch]),
    }
