from __future__ import annotations

import torch
from torch.utils.data import Dataset

from stroke_baseline.action_dataset import _decoder_input_with_end_padding, _effective_action_len
from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.dataset import read_jsonl


GEOMETRY_STATE_DIM = 15


def _geometry_row(
    *,
    x: float,
    y: float,
    pending_dx: float,
    pending_dy: float,
    start_x: float,
    start_y: float,
    bbox_min_x: float,
    bbox_min_y: float,
    bbox_max_x: float,
    bbox_max_y: float,
    progress: float,
) -> list[float]:
    partial_x = x + pending_dx
    partial_y = y + pending_dy
    return [
        x,
        y,
        pending_dx,
        pending_dy,
        partial_x,
        partial_y,
        start_x,
        start_y,
        partial_x - start_x,
        partial_y - start_y,
        bbox_min_x,
        bbox_min_y,
        bbox_max_x,
        bbox_max_y,
        progress,
    ]


def build_geometry_states(strokes: list[dict], max_action_len: int) -> torch.Tensor:
    """Build continuous geometry state aligned to action-token prediction positions.

    For each stroke step, the three positions predict dx, dy, and pen. The state
    intentionally mirrors inference-time availability:
    - dx position: current point only
    - dy position: current point plus known/generated dx
    - pen position: current point plus known/generated dx and dy
    """
    states = torch.zeros(max_action_len, GEOMETRY_STATE_DIM, dtype=torch.float32)
    if not strokes:
        return states

    x = y = 0.0
    start_x = start_y = 0.0
    bbox_min_x = bbox_max_x = x
    bbox_min_y = bbox_max_y = y
    token_pos = 0
    total_positions = max(1, min(len(strokes) * 3, max_action_len))

    for step_idx, step in enumerate(strokes):
        if token_pos >= max_action_len:
            break
        dx = float(step["dx"])
        dy = float(step["dy"])

        for phase, (pending_dx, pending_dy) in enumerate(((0.0, 0.0), (dx, 0.0), (dx, dy))):
            if token_pos >= max_action_len:
                break
            states[token_pos] = torch.tensor(
                _geometry_row(
                    x=x,
                    y=y,
                    pending_dx=pending_dx,
                    pending_dy=pending_dy,
                    start_x=start_x,
                    start_y=start_y,
                    bbox_min_x=bbox_min_x,
                    bbox_min_y=bbox_min_y,
                    bbox_max_x=bbox_max_x,
                    bbox_max_y=bbox_max_y,
                    progress=token_pos / total_positions,
                ),
                dtype=torch.float32,
            )
            token_pos += 1

        x += dx
        y += dy
        bbox_min_x = min(bbox_min_x, x)
        bbox_min_y = min(bbox_min_y, y)
        bbox_max_x = max(bbox_max_x, x)
        bbox_max_y = max(bbox_max_y, y)
        if step.get("pen_state") in {"end_shape", "end_all"}:
            start_x = x
            start_y = y

    return states


class GeometryActionTokenJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        action_tokenizer: StrokeActionTokenizer | None = None,
        max_action_len: int = 384,
        limit: int | None = None,
    ) -> None:
        self.raw_samples = read_jsonl(path, limit=limit)
        if not self.raw_samples:
            raise ValueError(f"No samples found in {path}")
        self.action_tokenizer = action_tokenizer or StrokeActionTokenizer(ActionTokenizerConfig())
        self.max_action_len = _effective_action_len(max_action_len)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
        strokes = raw["strokes"]
        tokens = self.action_tokenizer.encode_strokes(strokes)
        seq_len = min(len(tokens), self.max_action_len)
        actual = torch.tensor(tokens[:seq_len], dtype=torch.long)

        decoder_input = _decoder_input_with_end_padding(self.action_tokenizer, self.max_action_len)
        target = torch.full((self.max_action_len,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.max_action_len, dtype=torch.bool)

        decoder_input[0] = self.action_tokenizer.start_id
        if seq_len > 1:
            decoder_input[1:seq_len] = actual[:-1]

        target[:seq_len] = actual
        target_mask[:seq_len] = True

        return {
            "prompt": str(raw["prompt"]),
            "decoder_input_ids": decoder_input,
            "geometry_states": build_geometry_states(strokes, self.max_action_len),
            "target_ids": target,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
