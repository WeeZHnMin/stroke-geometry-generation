import warnings

import torch
from torch.utils.data import Dataset

from .action_tokenizer import ActionTokenizerConfig, CompactActionTokenMapper, StrokeActionTokenizer
from .dataset import read_jsonl


def _effective_action_len(max_action_len: int) -> int:
    if max_action_len <= 0:
        raise ValueError("max_action_len must be positive")
    return max_action_len


def _count_range_issues(raw_samples: list[dict], action_tokenizer: StrokeActionTokenizer) -> tuple[int, int]:
    total = 0
    out_of_range = 0
    lo = action_tokenizer.cfg.min_coord
    hi = action_tokenizer.cfg.max_coord
    for raw in raw_samples:
        canvas_size = float(raw.get("metadata", {}).get("canvas_size", 1.0))
        for step in action_tokenizer.strokes_to_cartesian_actions(raw["strokes"], canvas_size=canvas_size):
            total += 2
            if not (lo <= float(step["x"]) <= hi):
                out_of_range += 1
            if not (lo <= float(step["y"]) <= hi):
                out_of_range += 1
    return out_of_range, total


def _warn_if_tokenizer_clamps(raw_samples: list[dict], action_tokenizer: StrokeActionTokenizer) -> None:
    out_of_range, total = _count_range_issues(raw_samples, action_tokenizer)
    if out_of_range:
        warnings.warn(
            f"{out_of_range}/{total} coordinate values are outside the tokenizer range and will be clamped.",
            stacklevel=2,
        )


class ActionTokenJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        action_tokenizer: StrokeActionTokenizer | None = None,
        compact_mapper: CompactActionTokenMapper | None = None,
        max_action_len: int = 384,
        limit: int | None = None,
    ) -> None:
        self.raw_samples = read_jsonl(path, limit=limit)
        if not self.raw_samples:
            raise ValueError(f"No samples found in {path}")
        self.action_tokenizer = action_tokenizer or StrokeActionTokenizer(ActionTokenizerConfig())
        self.compact_mapper = compact_mapper
        self.max_action_len = _effective_action_len(max_action_len)
        _warn_if_tokenizer_clamps(self.raw_samples, self.action_tokenizer)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
        strokes = raw["strokes"]
        canvas_size = float(raw.get("metadata", {}).get("canvas_size", 1.0))
        actions = self.action_tokenizer.strokes_to_cartesian_actions(strokes, canvas_size=canvas_size)
        raw_tokens = self.action_tokenizer.encode_sequence(actions)
        tokens = [self.compact_mapper.encode(token) for token in raw_tokens] if self.compact_mapper is not None else raw_tokens
        seq_len = min(len(tokens), self.max_action_len)

        actual_coords = torch.tensor(
            [[float(step["x"]), float(step["y"])] for step in actions[:seq_len]],
            dtype=torch.float32,
        )
        actual_pen = torch.tensor(
            [self.action_tokenizer.normalize_pen_state(step["pen_state"]) for step in actions[:seq_len]],
            dtype=torch.long,
        )
        actual_tokens = torch.tensor(tokens[:seq_len], dtype=torch.long)

        decoder_coords = torch.zeros(self.max_action_len, 2, dtype=torch.float32)
        decoder_pen_states = torch.full(
            (self.max_action_len,),
            self.action_tokenizer.pad_input_pen_id,
            dtype=torch.long,
        )
        target = torch.full((self.max_action_len,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.max_action_len, dtype=torch.bool)

        decoder_pen_states[0] = self.action_tokenizer.start_input_pen_id
        if seq_len > 1:
            decoder_coords[1:seq_len] = actual_coords[:-1]
            decoder_pen_states[1:seq_len] = actual_pen[:-1]

        target[:seq_len] = actual_tokens
        target_mask[:seq_len] = True

        return {
            "prompt": str(raw["prompt"]),
            "decoder_coords": decoder_coords,
            "decoder_pen_states": decoder_pen_states,
            "target_ids": target,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


class TwoStageActionTokenJsonlDataset(ActionTokenJsonlDataset):
    pass
