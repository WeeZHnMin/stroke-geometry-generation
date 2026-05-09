import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer


def _effective_action_len(max_action_len: int) -> int:
    effective = max_action_len - (max_action_len % 3)
    if effective != max_action_len:
        warnings.warn(
            f"max_action_len={max_action_len} is not divisible by 3; "
            f"using {effective} so dx/dy/pen triples are not truncated.",
            stacklevel=2,
        )
    if effective <= 0:
        raise ValueError("max_action_len must leave room for at least one dx/dy/pen triple")
    return effective


def _count_range_issues_from_sample(
    raw: dict,
    action_tokenizer: StrokeActionTokenizer,
    *,
    two_stage: bool,
) -> tuple[int, int]:
    total = 0
    out_of_range = 0
    if two_stage:
        lo = action_tokenizer.cfg.draw_min_value
        hi = action_tokenizer.cfg.draw_max_value
        strokes = raw["strokes"][1:]
    else:
        lo = action_tokenizer.cfg.min_value
        hi = action_tokenizer.cfg.max_value
        strokes = raw["strokes"]

    for step in strokes:
        total += 2
        dx = float(step["dx"])
        dy = float(step["dy"])
        if not (lo <= dx <= hi):
            out_of_range += 1
        if not (lo <= dy <= hi):
            out_of_range += 1
    return out_of_range, total


def _decoder_input_with_end_padding(action_tokenizer: StrokeActionTokenizer, max_action_len: int) -> torch.Tensor:
    pad_step = action_tokenizer.end_padding_step()
    return torch.tensor([pad_step[pos % 3] for pos in range(max_action_len)], dtype=torch.long)


def _build_coords_from_strokes(strokes: list[dict], max_action_len: int) -> torch.Tensor:
    """Each stroke step occupies 3 consecutive sequence positions, all sharing
    the same canvas position (the position right BEFORE the action is applied).
    After the 3 positions are filled the running (x, y) is advanced by (dx, dy)."""
    coords = torch.zeros(max_action_len, 2, dtype=torch.float32)
    if not strokes:
        return coords
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
            coords[pos, 0] = x
            coords[pos, 1] = y
            pos += 1
        x += dx
        y += dy
    return coords


def _build_remaining_log(seq_len: int, max_action_len: int) -> torch.Tensor:
    """Per-position regression target: log(remaining_tokens) for valid positions.

    At position p ∈ [0, seq_len), value = log(seq_len - p). Positions beyond
    seq_len are zero-filled and masked out via target_mask in the loss.
    """
    out = torch.zeros(max_action_len, dtype=torch.float32)
    if seq_len > 0:
        positions = torch.arange(seq_len, dtype=torch.float32)
        out[:seq_len] = torch.log((seq_len - positions).clamp_min(1.0))
    return out


def _tensor_from_list(values: list, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


class _IndexedJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        action_tokenizer: StrokeActionTokenizer | None,
        max_action_len: int,
        limit: int | None,
        *,
        two_stage: bool,
        check_ranges: bool = False,
        progress_every: int = 10000,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(path)
        self.action_tokenizer = action_tokenizer or StrokeActionTokenizer(ActionTokenizerConfig())
        self.max_action_len = _effective_action_len(max_action_len)
        self.two_stage = two_stage
        self.offsets: list[int] = []

        total = 0
        out_of_range = 0
        with self.path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self.offsets.append(offset)
                if check_ranges:
                    raw = json.loads(line)
                    cur_out, cur_total = _count_range_issues_from_sample(raw, self.action_tokenizer, two_stage=two_stage)
                    out_of_range += cur_out
                    total += cur_total
                if progress_every > 0 and len(self.offsets) % progress_every == 0:
                    print(f"indexed_samples={len(self.offsets)} path={self.path.name}", flush=True)
                if limit is not None and len(self.offsets) >= limit:
                    break

        if not self.offsets:
            raise ValueError(f"No samples found in {path}")

        if check_ranges and out_of_range:
            mode = "two-stage draw" if two_stage else "single-stage"
            warnings.warn(
                f"{out_of_range}/{total} {mode} dx/dy values are outside the tokenizer range and will be clamped. "
                "This can make low token loss incompatible with accurate geometry.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_raw(self, idx: int) -> dict:
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)


class ActionTokenJsonlDataset(_IndexedJsonlDataset):
    def __init__(
        self,
        path: str,
        action_tokenizer: StrokeActionTokenizer | None = None,
        max_action_len: int = 384,
        limit: int | None = None,
        check_ranges: bool = False,
        progress_every: int = 10000,
    ) -> None:
        super().__init__(
            path,
            action_tokenizer,
            max_action_len,
            limit,
            two_stage=False,
            check_ranges=check_ranges,
            progress_every=progress_every,
        )

    def __getitem__(self, idx: int) -> dict:
        raw = self._read_raw(idx)
        if "decoder_input_ids" in raw and "target_ids" in raw and "target_mask" in raw:
            decoder_input = _tensor_from_list(raw["decoder_input_ids"], dtype=torch.long)
            target = _tensor_from_list(raw["target_ids"], dtype=torch.long)
            target_mask = _tensor_from_list(raw["target_mask"], dtype=torch.bool)
            seq_len = int(raw.get("length", int(target_mask.sum().item())))
        else:
            tokens = self.action_tokenizer.encode_strokes(raw["strokes"])
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

        if "coords" in raw:
            coords = _tensor_from_list(raw["coords"], dtype=torch.float32)
            if coords.dim() == 2 and coords.size(0) < self.max_action_len:
                padded = torch.zeros(self.max_action_len, 2, dtype=torch.float32)
                padded[: coords.size(0)] = coords
                coords = padded
        else:
            coords = _build_coords_from_strokes(raw["strokes"], self.max_action_len)

        return {
            "prompt": str(raw["prompt"]),
            "coords": coords,
            "decoder_input_ids": decoder_input,
            "target_ids": target,
            "target_mask": target_mask,
            "remaining_log": _build_remaining_log(seq_len, self.max_action_len),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


class TwoStageActionTokenJsonlDataset(_IndexedJsonlDataset):
    def __init__(
        self,
        path: str,
        action_tokenizer: StrokeActionTokenizer | None = None,
        max_action_len: int = 384,
        limit: int | None = None,
        check_ranges: bool = False,
        progress_every: int = 10000,
    ) -> None:
        super().__init__(
            path,
            action_tokenizer,
            max_action_len,
            limit,
            two_stage=True,
            check_ranges=check_ranges,
            progress_every=progress_every,
        )

    def __getitem__(self, idx: int) -> dict:
        raw = self._read_raw(idx)
        strokes = raw["strokes"]
        move_step = strokes[0]
        start_position = torch.tensor([move_step["dx"], move_step["dy"]], dtype=torch.float32)
        assert move_step["pen_state"] == "move", f"Expected first step to be move, got {move_step['pen_state']}"

        draw_strokes = strokes[1:]
        if "decoder_input_ids" in raw and "target_ids" in raw and "target_mask" in raw:
            decoder_input = _tensor_from_list(raw["decoder_input_ids"], dtype=torch.long)
            target = _tensor_from_list(raw["target_ids"], dtype=torch.long)
            target_mask = _tensor_from_list(raw["target_mask"], dtype=torch.bool)
            seq_len = int(raw.get("length", int(target_mask.sum().item())))
        else:
            tokens = self.action_tokenizer.encode_draw_strokes(draw_strokes)
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

        if "coords" in raw:
            coords = _tensor_from_list(raw["coords"], dtype=torch.float32)
            if coords.dim() == 2 and coords.size(0) < self.max_action_len:
                padded = torch.zeros(self.max_action_len, 2, dtype=torch.float32)
                padded[: coords.size(0)] = coords
                coords = padded
        else:
            coords = _build_coords_from_strokes(raw["strokes"], self.max_action_len)

        return {
            "prompt": str(raw["prompt"]),
            "start_position": start_position,
            "coords": coords,
            "decoder_input_ids": decoder_input,
            "target_ids": target,
            "target_mask": target_mask,
            "remaining_log": _build_remaining_log(seq_len, self.max_action_len),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
