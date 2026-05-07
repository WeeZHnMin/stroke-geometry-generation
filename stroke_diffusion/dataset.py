from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from stroke_baseline.dataset import read_jsonl


STEP_DIM = 3
PEN_STATE_TO_ID = {
    "move": 0,
    "draw": 1,
    "end_all": 2,
}


def parse_pen_state(step: dict) -> int:
    pen_state = step.get("pen_state")
    if isinstance(pen_state, str):
        if pen_state not in PEN_STATE_TO_ID:
            raise ValueError(f"Unsupported pen_state={pen_state!r}")
        return int(PEN_STATE_TO_ID[pen_state])

    for key, pen_id in (("pen_move", PEN_STATE_TO_ID["move"]), ("pen_draw", PEN_STATE_TO_ID["draw"]), ("pen_end_all", PEN_STATE_TO_ID["end_all"])):
        if float(step.get(key, 0.0)) >= 0.5:
            return pen_id

    if "pen_id" in step:
        return int(step["pen_id"])
    raise ValueError("Step is missing pen_state/pen_id label.")


class QuantizedStrokeDiffusionDataset(Dataset):
    """Dataset for continuous stroke diffusion training.

    Expects JSONL samples containing:
    - prompt
    - continuous_sequence: [{x, y, pen_state|pen_id}, ...]
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
        pen_ids = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        for i, step in enumerate(seq[:seq_len]):
            steps[i, 0] = float(step["x"])
            steps[i, 1] = float(step["y"])
            pen_id = parse_pen_state(step)
            steps[i, 2] = float(pen_id)
            pen_ids[i] = pen_id
            seq_mask[i] = True

        return {
            "prompt": str(raw["prompt"]),
            "steps": steps,
            "pen_ids": pen_ids,
            "seq_mask": seq_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
