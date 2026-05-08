from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


GRID_SIZE = 101
GRID_CELLS = GRID_SIZE * GRID_SIZE
NUM_PEN_STATES = 3
PAD_TOKEN_ID = GRID_CELLS * NUM_PEN_STATES
VOCAB_SIZE = PAD_TOKEN_ID + 1

MOVE_PEN_ID = 0
DRAW_PEN_ID = 1
END_PEN_ID = 2

START_INPUT_PEN_ID = 3
PAD_INPUT_PEN_ID = 4
NUM_INPUT_PEN_STATES = 5

RAW_PEN_TO_ID = {
    "move": MOVE_PEN_ID,
    "draw": DRAW_PEN_ID,
    "end": END_PEN_ID,
    "close": END_PEN_ID,
    "end_shape": END_PEN_ID,
    "end_all": END_PEN_ID,
}
ID_TO_RAW_PEN = {
    MOVE_PEN_ID: "move",
    DRAW_PEN_ID: "draw",
    END_PEN_ID: "end_all",
}


@dataclass
class CartesianTokenizerConfig:
    vocab_size: int = VOCAB_SIZE
    pad_id: int = PAD_TOKEN_ID
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
        self.start_input_pen_id = START_INPUT_PEN_ID
        self.pad_input_pen_id = PAD_INPUT_PEN_ID
        self.num_input_pen_states = NUM_INPUT_PEN_STATES
        self.num_pen_states = NUM_PEN_STATES
        self.grid_size = GRID_SIZE
        self.grid_cells = GRID_CELLS

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

    def normalize_pen_state(self, pen_state: int | str) -> int:
        if isinstance(pen_state, str):
            if pen_state not in RAW_PEN_TO_ID:
                raise ValueError(f"Unsupported pen_state string: {pen_state}")
            return RAW_PEN_TO_ID[pen_state]
        pen_id = int(pen_state)
        if pen_id < 0 or pen_id >= NUM_PEN_STATES:
            raise ValueError(f"pen_state must be in [0, 2], got {pen_state}")
        return pen_id

    def encode_action(self, dx: float, dy: float, pen_state: int | str) -> int:
        pen_id = self.normalize_pen_state(pen_state)
        bin_x = self._value_to_bin(dx)
        bin_y = self._value_to_bin(dy)
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

    def encode_sequence(self, actions: Iterable[dict | tuple | list | torch.Tensor]) -> list[int]:
        tokens: list[int] = []
        for action in actions:
            if isinstance(action, dict):
                x = action.get("x", action.get("dx"))
                y = action.get("y", action.get("dy"))
                pen_state = action["pen_state"]
            else:
                x, y, pen_state = action
            tokens.append(self.encode_action(float(x), float(y), pen_state))
        return tokens

    def decode_sequence(self, token_ids: Iterable[int] | torch.Tensor) -> list[dict[str, float | int]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        decoded: list[dict[str, float | int]] = []
        for token_id in token_ids:
            if int(token_id) == self.pad_id:
                continue
            x, y, pen_state = self.decode_action(int(token_id))
            decoded.append({"dx": x, "dy": y, "pen_state": pen_state})
        return decoded

    def strokes_to_cartesian_actions(
        self,
        strokes: Iterable[dict],
        *,
        canvas_size: float = 1.0,
    ) -> list[dict[str, float | int]]:
        x = 0.0
        y = 0.0
        actions: list[dict[str, float | int]] = []
        for step in strokes:
            x += float(step["dx"])
            y += float(step["dy"])
            actions.append(
                {
                    "x": self.normalize_absolute_coord(x, canvas_size),
                    "y": self.normalize_absolute_coord(y, canvas_size),
                    "pen_state": self.normalize_pen_state(step["pen_state"]),
                }
            )
        return actions

    def cartesian_actions_to_strokes(self, actions: Iterable[dict | tuple | list | torch.Tensor]) -> list[dict[str, float | int]]:
        prev_x = 0.0
        prev_y = 0.0
        strokes: list[dict[str, float | int]] = []
        for action in actions:
            if isinstance(action, dict):
                x = float(action.get("x", action.get("dx")))
                y = float(action.get("y", action.get("dy")))
                pen_state = self.normalize_pen_state(action["pen_state"])
            else:
                x, y, pen_state = action
                x = float(x)
                y = float(y)
                pen_state = self.normalize_pen_state(pen_state)
            strokes.append({"dx": x - prev_x, "dy": y - prev_y, "pen_state": ID_TO_RAW_PEN[pen_state]})
            prev_x, prev_y = x, y
        return strokes

    def valid_token_mask(self, positions: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(*positions.shape, self.vocab_size, dtype=torch.bool, device=positions.device)
        mask[..., self.pad_id] = False
        return mask

    def encode_strokes(self, strokes: Iterable[dict]) -> list[int]:
        return self.encode_sequence(self.strokes_to_cartesian_actions(strokes))

    def decode_tokens(self, tokens: Iterable[int] | torch.Tensor) -> list[dict[str, float | int]]:
        actions = self.decode_sequence(tokens)
        return self.cartesian_actions_to_strokes(actions)


class CompactActionTokenMapper:
    def __init__(self, raw_token_ids: Iterable[int]):
        unique = sorted({int(token_id) for token_id in raw_token_ids})
        if not unique:
            raise ValueError("No Cartesian action tokens found for compact vocabulary.")
        self.raw_to_compact = {raw_token_id: idx for idx, raw_token_id in enumerate(unique)}
        self.compact_to_raw = {idx: raw_token_id for idx, raw_token_id in enumerate(unique)}
        self.action_vocab_size = len(unique)
        self.pad_id = self.action_vocab_size
        self.vocab_size = self.action_vocab_size + 1

    @classmethod
    def from_vocab_file(cls, path: str | Path) -> "CompactActionTokenMapper":
        payload = torch.load(path) if str(path).endswith(".pt") else None
        if payload is not None:
            token_ids = payload["raw_token_ids"]
            return cls(token_ids)
        path = Path(path)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        token_ids = [int(item["raw_token_id"]) for item in data["tokens"]]
        return cls(token_ids)

    def encode(self, raw_token_id: int) -> int:
        raw_token_id = int(raw_token_id)
        try:
            return self.raw_to_compact[raw_token_id]
        except KeyError as exc:
            raise KeyError(f"raw token id {raw_token_id} is not in the observed Cartesian vocabulary") from exc

    def decode(self, compact_token_id: int) -> int:
        compact_token_id = int(compact_token_id)
        if compact_token_id < 0 or compact_token_id >= self.action_vocab_size:
            raise ValueError(f"not a compact Cartesian action token: {compact_token_id}")
        return self.compact_to_raw[compact_token_id]


ActionTokenizerConfig = CartesianTokenizerConfig
StrokeActionTokenizer = CartesianActionTokenizer
