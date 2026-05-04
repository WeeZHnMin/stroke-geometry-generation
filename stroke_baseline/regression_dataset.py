import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import read_jsonl

REGRESSION_PEN_TO_ID = {
    "draw": 0,
    "move": 1,
    "end": 2,
    "end_shape": 2,
    "end_all": 2,
}
REGRESSION_ID_TO_PEN = {0: "draw", 1: "move", 2: "end"}
NUM_REGRESSION_PEN_STATES = 3


def _accumulate_transitions(
    sample: dict,
    draw_only: bool,
    max_delta: float,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    x = 0.0
    y = 0.0
    coords: list[list[float]] = []
    deltas: list[list[float]] = []
    pens: list[int] = []
    for step in sample.get("strokes", []):
        dx = float(step["dx"])
        dy = float(step["dy"])
        pen = str(step.get("pen_state", "draw"))
        use_step = (not draw_only) or pen != "move"
        chunks = max(1, int(torch.ceil(torch.tensor(max(abs(dx), abs(dy)) / max_delta)).item()))
        chunk_dx = dx / chunks
        chunk_dy = dy / chunks
        for chunk_idx in range(chunks):
            chunk_pen = pen
            if pen in {"end", "end_shape", "end_all"} and chunk_idx < chunks - 1:
                chunk_pen = "draw"
            chunk_use_step = (not draw_only) or chunk_pen != "move"
            if chunk_use_step:
                if abs(chunk_dx) > max_delta + 1e-6 or abs(chunk_dy) > max_delta + 1e-6:
                    raise ValueError(f"delta out of range after split: dx={chunk_dx} dy={chunk_dy}")
                coords.append([x, y])
                deltas.append([chunk_dx, chunk_dy])
                pens.append(REGRESSION_PEN_TO_ID.get(chunk_pen, 0))
            x += chunk_dx
            y += chunk_dy
    return coords, deltas, pens


class StrokeRegressionJsonlDataset(Dataset):
    """JSONL dataset for continuous dx/dy regression.

    Decoder input is the true historical absolute coordinate (x, y) plus
    normalized step index. Target is the next continuous dx/dy.
    """

    def __init__(
        self,
        path: str | Path,
        max_len: int = 96,
        canvas_size: float = 8.0,
        max_delta: float = 0.5,
        draw_only: bool = True,
        limit: int | None = None,
        skip_invalid: bool = True,
    ) -> None:
        raw_samples = read_jsonl(path, limit=limit)
        self.samples: list[dict] = []
        self.max_len = int(max_len)
        self.canvas_size = float(canvas_size)
        self.max_delta = float(max_delta)
        self.draw_only = bool(draw_only)

        for raw in raw_samples:
            try:
                coords, deltas, pens = _accumulate_transitions(raw, self.draw_only, self.max_delta)
            except ValueError:
                if skip_invalid:
                    continue
                raise
            if not coords:
                continue
            self.samples.append({"prompt": str(raw["prompt"]), "coords": coords, "deltas": deltas, "pens": pens})
        if not self.samples:
            raise ValueError(f"No valid regression samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        coords = sample["coords"][: self.max_len]
        deltas = sample["deltas"][: self.max_len]
        pens = sample["pens"][: self.max_len]
        seq_len = len(coords)

        decoder_coords = torch.zeros(self.max_len, 3, dtype=torch.float32)
        target_dxdy = torch.zeros(self.max_len, 2, dtype=torch.float32)
        target_pen = torch.full((self.max_len,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.max_len, dtype=torch.bool)

        for i, ((x, y), (dx, dy), pen) in enumerate(zip(coords, deltas, pens)):
            decoder_coords[i, 0] = x / self.canvas_size * 2.0 - 1.0
            decoder_coords[i, 1] = y / self.canvas_size * 2.0 - 1.0
            decoder_coords[i, 2] = i / max(self.max_len - 1, 1)
            target_dxdy[i, 0] = dx
            target_dxdy[i, 1] = dy
            target_pen[i] = pen
            target_mask[i] = True

        return {
            "prompt": sample["prompt"],
            "decoder_coords": decoder_coords,
            "target_dxdy": target_dxdy,
            "target_pen": target_pen,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
