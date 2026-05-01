import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


PEN_TO_ID = {
    "draw": 0,
    "move": 1,
    "end_shape": 2,
    "end_all": 3,
}
ID_TO_PEN = {idx: name for name, idx in PEN_TO_ID.items()}
START_PEN_ID = 4
PAD_PEN_ID = 5
NUM_INPUT_PEN_STATES = 6
NUM_TARGET_PEN_STATES = len(PEN_TO_ID)


@dataclass
class StrokeSample:
    prompt: str
    dxdy: list[list[float]]
    pen: list[int]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_stroke_samples(path: str | Path, limit: int | None = None) -> list[StrokeSample]:
    raw_samples = read_jsonl(path, limit=limit)
    samples: list[StrokeSample] = []
    for raw in raw_samples:
        strokes = raw.get("strokes", [])
        dxdy = [[float(step["dx"]), float(step["dy"])] for step in strokes]
        pen = [PEN_TO_ID[step["pen_state"]] for step in strokes]
        samples.append(StrokeSample(prompt=str(raw["prompt"]), dxdy=dxdy, pen=pen))
    return samples


class StrokeJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: CharTokenizer | None = None,
        max_text_len: int = 96,
        max_stroke_len: int = 128,
        limit: int | None = None,
    ) -> None:
        self.samples = load_stroke_samples(path, limit=limit)
        if not self.samples:
            raise ValueError(f"No samples found in {path}")
        self.tokenizer = tokenizer or CharTokenizer.build(sample.prompt for sample in self.samples)
        self.max_text_len = max_text_len
        self.max_stroke_len = max_stroke_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        seq_len = min(len(sample.dxdy), self.max_stroke_len)
        dxdy = torch.zeros(self.max_stroke_len, 2, dtype=torch.float32)
        pen = torch.full((self.max_stroke_len,), PAD_PEN_ID, dtype=torch.long)
        target_dxdy = torch.zeros(self.max_stroke_len, 2, dtype=torch.float32)
        target_pen = torch.full((self.max_stroke_len,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.max_stroke_len, dtype=torch.bool)

        actual_dxdy = torch.tensor(sample.dxdy[:seq_len], dtype=torch.float32)
        actual_pen = torch.tensor(sample.pen[:seq_len], dtype=torch.long)

        # Decoder input is shifted right: start token, then previous ground-truth strokes.
        dxdy[0] = torch.zeros(2)
        pen[0] = START_PEN_ID
        if seq_len > 1:
            dxdy[1:seq_len] = actual_dxdy[:-1]
            pen[1:seq_len] = actual_pen[:-1]

        target_dxdy[:seq_len] = actual_dxdy
        target_pen[:seq_len] = actual_pen
        target_mask[:seq_len] = True

        text_ids = torch.tensor(self.tokenizer.encode(sample.prompt, self.max_text_len), dtype=torch.long)
        text_mask = text_ids != self.tokenizer.pad_id

        return {
            "prompt": sample.prompt,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "decoder_dxdy": dxdy,
            "decoder_pen": pen,
            "target_dxdy": target_dxdy,
            "target_pen": target_pen,
            "target_mask": target_mask,
        }


def prompts_from_jsonl(path: str | Path, limit: int | None = None) -> Iterable[str]:
    for raw in read_jsonl(path, limit=limit):
        yield str(raw["prompt"])

