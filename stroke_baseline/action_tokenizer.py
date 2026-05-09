from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


# ---------- Per-dimension (dx / dy / pen) tokenizer ----------
#
# Each stroke step is expanded into 3 sequence positions:
#   [dx_token, dy_token, pen_token, dx_token, dy_token, pen_token, ...]
#
# Pen state vocabulary is local to this tokenizer (only "draw" / "move" — end-type
# states are dropped during encoding).
#
# Token id layout in a single shared vocabulary:
#   [0,        bins)            -> dx bin tokens   (dx_offset = 0)
#   [bins,     2*bins)          -> dy bin tokens   (dy_offset = bins)
#   [2*bins,   2*bins + P)      -> pen tokens      (pen_offset = 2*bins, P = 2)
#   2*bins + P                  -> start_id
#   2*bins + P + 1              -> pad_id
#   vocab_size = 2*bins + P + 2


ACTION_PEN_TO_ID: dict[str, int] = {"draw": 0, "move": 1}
ACTION_ID_TO_PEN: dict[int, str] = {idx: name for name, idx in ACTION_PEN_TO_ID.items()}

_DROPPED_PEN_STATES = frozenset({"end", "end_shape", "end_all", "close"})


def _normalize_pen_state(pen_state: int | str) -> str | None:
    """Return canonical pen-state name, or None if this step should be dropped."""
    if isinstance(pen_state, int):
        return ACTION_ID_TO_PEN.get(int(pen_state))
    name = str(pen_state)
    if name in _DROPPED_PEN_STATES:
        return None
    if name not in ACTION_PEN_TO_ID:
        raise ValueError(f"Unsupported pen_state: {pen_state}")
    return name


@dataclass
class PerDimActionTokenizerConfig:
    bins: int = 500
    min_value: float = -0.5
    max_value: float = 0.5
    draw_min_value: float = -0.5
    draw_max_value: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class PerDimActionTokenizer:
    """Per-dimension tokenizer: emits 3 tokens (dx_bin, dy_bin, pen) per stroke step."""

    def __init__(self, cfg: PerDimActionTokenizerConfig | None = None):
        self.cfg = cfg or PerDimActionTokenizerConfig()
        if self.cfg.bins <= 0:
            raise ValueError("bins must be positive")
        if self.cfg.max_value <= self.cfg.min_value:
            raise ValueError("max_value must be greater than min_value")
        if self.cfg.draw_max_value <= self.cfg.draw_min_value:
            raise ValueError("draw_max_value must be greater than draw_min_value")

        self.bins = int(self.cfg.bins)
        self.num_pen_states = len(ACTION_PEN_TO_ID)

        self.dx_offset = 0
        self.dy_offset = self.bins
        self.pen_offset = 2 * self.bins
        self.start_id = 2 * self.bins + self.num_pen_states
        self.pad_id = self.start_id + 1
        self.vocab_size = self.pad_id + 1

    # --- bin <-> value ---

    def _value_to_bin(self, value: float, lo: float, hi: float) -> int:
        v = float(value)
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        width = (hi - lo) / self.bins
        idx = int((v - lo) / width)
        if idx < 0:
            idx = 0
        elif idx >= self.bins:
            idx = self.bins - 1
        return idx

    def _bin_to_value(self, bin_idx: int, lo: float, hi: float) -> float:
        idx = int(bin_idx)
        if idx < 0:
            idx = 0
        elif idx >= self.bins:
            idx = self.bins - 1
        width = (hi - lo) / self.bins
        return lo + (idx + 0.5) * width

    # --- per-step encode / decode ---

    def encode_step(self, dx: float, dy: float, pen_state: int | str) -> list[int] | None:
        """Encode one stroke step. Returns None if the step's pen_state is end-type
        (end / end_shape / end_all / close) and should be dropped."""
        return self._encode_step(dx, dy, pen_state, self.cfg.min_value, self.cfg.max_value)

    def encode_draw_step(self, dx: float, dy: float, pen_state: int | str) -> list[int] | None:
        return self._encode_step(dx, dy, pen_state, self.cfg.draw_min_value, self.cfg.draw_max_value)

    def _encode_step(self, dx: float, dy: float, pen_state: int | str, lo: float, hi: float) -> list[int] | None:
        pen_name = _normalize_pen_state(pen_state)
        if pen_name is None:
            return None
        dx_bin = self._value_to_bin(dx, lo, hi)
        dy_bin = self._value_to_bin(dy, lo, hi)
        pen_id = ACTION_PEN_TO_ID[pen_name]
        return [
            self.dx_offset + dx_bin,
            self.dy_offset + dy_bin,
            self.pen_offset + pen_id,
        ]

    def decode_step(self, dx_token: int, dy_token: int, pen_token: int) -> dict:
        return self._decode_step(dx_token, dy_token, pen_token, self.cfg.min_value, self.cfg.max_value)

    def decode_draw_step(self, dx_token: int, dy_token: int, pen_token: int) -> dict:
        return self._decode_step(dx_token, dy_token, pen_token, self.cfg.draw_min_value, self.cfg.draw_max_value)

    def _decode_step(self, dx_token: int, dy_token: int, pen_token: int, lo: float, hi: float) -> dict:
        dx_bin = int(dx_token) - self.dx_offset
        dy_bin = int(dy_token) - self.dy_offset
        pen_id = int(pen_token) - self.pen_offset
        if pen_id < 0 or pen_id >= self.num_pen_states:
            pen_id = max(0, min(self.num_pen_states - 1, pen_id))
        return {
            "dx": self._bin_to_value(dx_bin, lo, hi),
            "dy": self._bin_to_value(dy_bin, lo, hi),
            "pen_state": ACTION_ID_TO_PEN[pen_id],
        }

    # --- sequence helpers ---

    def encode_strokes(self, strokes: Iterable[dict]) -> list[int]:
        tokens: list[int] = []
        for step in strokes:
            triple = self.encode_step(float(step["dx"]), float(step["dy"]), step["pen_state"])
            if triple is None:
                continue
            tokens.extend(triple)
        return tokens

    def encode_draw_strokes(self, strokes: Iterable[dict]) -> list[int]:
        tokens: list[int] = []
        for step in strokes:
            triple = self.encode_draw_step(float(step["dx"]), float(step["dy"]), step["pen_state"])
            if triple is None:
                continue
            tokens.extend(triple)
        return tokens

    def decode_tokens(self, tokens: Iterable[int] | torch.Tensor) -> list[dict]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        tokens = [int(t) for t in tokens]
        n = len(tokens) - (len(tokens) % 3)
        strokes: list[dict] = []
        for i in range(0, n, 3):
            dx_tok, dy_tok, pen_tok = tokens[i], tokens[i + 1], tokens[i + 2]
            if not self._is_dx_token(dx_tok) or not self._is_dy_token(dy_tok) or not self._is_pen_token(pen_tok):
                # Hitting pad_id (or any out-of-range token) terminates decoding.
                break
            strokes.append(self.decode_step(dx_tok, dy_tok, pen_tok))
        return strokes

    def _is_dx_token(self, token: int) -> bool:
        return self.dx_offset <= token < self.dx_offset + self.bins

    def _is_dy_token(self, token: int) -> bool:
        return self.dy_offset <= token < self.dy_offset + self.bins

    def _is_pen_token(self, token: int) -> bool:
        return self.pen_offset <= token < self.pen_offset + self.num_pen_states

    def end_padding_step(self) -> tuple[int, int, int]:
        """3 pad tokens to fill decoder_input positions past the real sequence length.
        These are masked out via target_mask in the loss."""
        return (self.pad_id, self.pad_id, self.pad_id)


# ---------- Cartesian (single-token-per-step) tokenizer ----------
#
# Kept for backward compatibility with `dataset_code/export_cartesian_vocab.py`
# which uses a single-token-per-action vocabulary
# (pen * grid + bin_x * grid + bin_y).

GRID_SIZE = 101
GRID_CELLS = GRID_SIZE * GRID_SIZE
NUM_PEN_STATES_CARTESIAN = 3
CARTESIAN_PAD_TOKEN_ID = GRID_CELLS * NUM_PEN_STATES_CARTESIAN
CARTESIAN_VOCAB_SIZE = CARTESIAN_PAD_TOKEN_ID + 1

_CARTESIAN_PEN_TO_ID = {"move": 0, "draw": 1, "end": 2, "close": 2, "end_shape": 2, "end_all": 2}
_CARTESIAN_ID_TO_PEN = {0: "move", 1: "draw", 2: "end_all"}


@dataclass
class CartesianTokenizerConfig:
    vocab_size: int = CARTESIAN_VOCAB_SIZE
    pad_id: int = CARTESIAN_PAD_TOKEN_ID
    min_coord: float = -0.5
    max_coord: float = 0.5
    coord_step: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


class CartesianActionTokenizer:
    def __init__(self, cfg: CartesianTokenizerConfig | None = None):
        self.cfg = cfg or CartesianTokenizerConfig()
        self.vocab_size = self.cfg.vocab_size
        self.pad_id = self.cfg.pad_id
        self.grid_size = GRID_SIZE
        self.grid_cells = GRID_CELLS
        self.num_pen_states = NUM_PEN_STATES_CARTESIAN

    def _clamp_coord(self, value: float) -> float:
        return max(self.cfg.min_coord, min(self.cfg.max_coord, float(value)))

    def normalize_absolute_coord(self, value: float, canvas_size: float) -> float:
        canvas_size = max(float(canvas_size), 1e-6)
        centered = (float(value) / canvas_size) - 0.5
        return self._clamp_coord(centered)

    def _value_to_bin(self, value: float) -> int:
        clamped = self._clamp_coord(value)
        return int(round(clamped * 100.0)) + 50

    def _bin_to_value(self, bin_idx: int) -> float:
        return (int(bin_idx) - 50) / 100.0

    def _normalize_pen(self, pen_state: int | str) -> int:
        if isinstance(pen_state, str):
            if pen_state not in _CARTESIAN_PEN_TO_ID:
                raise ValueError(f"Unsupported pen_state: {pen_state}")
            return _CARTESIAN_PEN_TO_ID[pen_state]
        pen_id = int(pen_state)
        if pen_id < 0 or pen_id >= self.num_pen_states:
            raise ValueError(f"pen_state must be in [0, {self.num_pen_states - 1}], got {pen_state}")
        return pen_id

    def encode_action(self, x: float, y: float, pen_state: int | str) -> int:
        pen_id = self._normalize_pen(pen_state)
        bin_x = self._value_to_bin(x)
        bin_y = self._value_to_bin(y)
        return pen_id * self.grid_cells + bin_x * self.grid_size + bin_y

    def decode_action(self, token_id: int) -> tuple[float, float, int]:
        token_id = int(token_id)
        if token_id == self.pad_id:
            raise ValueError("pad_id does not decode to a Cartesian action.")
        if token_id < 0 or token_id >= self.pad_id:
            raise ValueError(f"token_id must be in [0, {self.pad_id - 1}], got {token_id}")
        pen_state, offset = divmod(token_id, self.grid_cells)
        bin_x, bin_y = divmod(offset, self.grid_size)
        return self._bin_to_value(bin_x), self._bin_to_value(bin_y), pen_state

    def strokes_to_cartesian_actions(
        self,
        strokes: Iterable[dict],
        *,
        canvas_size: float = 1.0,
    ) -> list[dict]:
        x = 0.0
        y = 0.0
        actions: list[dict] = []
        for step in strokes:
            x += float(step["dx"])
            y += float(step["dy"])
            actions.append(
                {
                    "x": self.normalize_absolute_coord(x, canvas_size),
                    "y": self.normalize_absolute_coord(y, canvas_size),
                    "pen_state": self._normalize_pen(step["pen_state"]),
                }
            )
        return actions


class CompactActionTokenMapper:
    def __init__(self, raw_token_ids: Iterable[int]):
        unique = sorted({int(token_id) for token_id in raw_token_ids})
        if not unique:
            raise ValueError("No Cartesian action tokens found for compact vocabulary.")
        self.raw_to_compact = {raw: idx for idx, raw in enumerate(unique)}
        self.compact_to_raw = {idx: raw for idx, raw in enumerate(unique)}
        self.action_vocab_size = len(unique)
        self.pad_id = self.action_vocab_size
        self.vocab_size = self.action_vocab_size + 1

    @classmethod
    def from_vocab_file(cls, path: str | Path) -> "CompactActionTokenMapper":
        payload = torch.load(path) if str(path).endswith(".pt") else None
        if payload is not None:
            return cls(payload["raw_token_ids"])
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([int(item["raw_token_id"]) for item in data["tokens"]])

    def encode(self, raw_token_id: int) -> int:
        try:
            return self.raw_to_compact[int(raw_token_id)]
        except KeyError as exc:
            raise KeyError(f"raw token id {raw_token_id} is not in the observed vocabulary") from exc

    def decode(self, compact_token_id: int) -> int:
        compact_token_id = int(compact_token_id)
        if compact_token_id < 0 or compact_token_id >= self.action_vocab_size:
            raise ValueError(f"not a compact action token: {compact_token_id}")
        return self.compact_to_raw[compact_token_id]


# Public aliases used across the codebase.
ActionTokenizerConfig = PerDimActionTokenizerConfig
StrokeActionTokenizer = PerDimActionTokenizer
