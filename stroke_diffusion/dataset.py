from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from stroke_baseline.dataset import read_jsonl


STEP_DIM = 5


def build_absolute_positions(steps: torch.Tensor) -> torch.Tensor:
    """Convert [T, 5] step sequence into absolute xy positions [T, 2]."""
    return torch.cumsum(steps[:, :2], dim=0)


class QuantizedStrokeDiffusionDataset(Dataset):
    """Dataset for continuous stroke diffusion training.

    Expects JSONL samples containing:
    - prompt
    - continuous_sequence: [{dx, dy, pen_move, pen_draw, pen_end_all}, ...]
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_seq_len: int = 192,
        limit: int | None = None,
    ) -> None:
        self.raw_samples = read_jsonl(path, limit=limit)
        if not self.raw_samples:
            raise ValueError(f"No samples found in {path}")
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
        seq = raw.get("continuous_sequence")
        if not seq:
            raise ValueError("Sample is missing continuous_sequence.")
        seq_len = min(len(seq), self.max_seq_len)

        steps = torch.zeros(self.max_seq_len, STEP_DIM, dtype=torch.float32)
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        for i, step in enumerate(seq[:seq_len]):
            steps[i, 0] = float(step["dx"])
            steps[i, 1] = float(step["dy"])
            steps[i, 2] = float(step["pen_move"])
            steps[i, 3] = float(step["pen_draw"])
            steps[i, 4] = float(step["pen_end_all"])
            seq_mask[i] = True

        target_abs = torch.zeros(self.max_seq_len, 2, dtype=torch.float32)
        target_abs[:seq_len] = build_absolute_positions(steps[:seq_len])
        start_pos = torch.zeros(1, 2, dtype=torch.float32)

        return {
            "prompt": str(raw["prompt"]),
            "steps": steps,
            "target_abs": target_abs,
            "start_pos": start_pos,
            "seq_mask": seq_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }

