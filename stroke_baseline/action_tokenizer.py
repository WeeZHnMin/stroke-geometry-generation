from dataclasses import asdict, dataclass

import torch

from .dataset import ID_TO_PEN, PEN_TO_ID


@dataclass
class ActionTokenizerConfig:
    bins: int = 512
    min_value: float = -1.0
    max_value: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class StrokeActionTokenizer:
    """OpenVLA-style per-dimension stroke action tokenizer.

    Each stroke step is represented as three autoregressive tokens:

    [DX_BIN, DY_BIN, PEN_STATE]

    Token ranges:
    - dx: 0 .. bins - 1
    - dy: bins .. 2 * bins - 1
    - pen: 2 * bins .. 2 * bins + 3
    - start: 2 * bins + 4
    - pad: 2 * bins + 5
    """

    def __init__(self, cfg: ActionTokenizerConfig | None = None):
        self.cfg = cfg or ActionTokenizerConfig()
        self.bins = self.cfg.bins
        self.dx_offset = 0
        self.dy_offset = self.bins
        self.pen_offset = self.bins * 2
        self.start_id = self.pen_offset + len(PEN_TO_ID)
        self.pad_id = self.start_id + 1
        self.vocab_size = self.pad_id + 1

    def _value_to_bin(self, value: float) -> int:
        value = max(self.cfg.min_value, min(self.cfg.max_value, float(value)))
        scale = (value - self.cfg.min_value) / (self.cfg.max_value - self.cfg.min_value)
        idx = int(round(scale * (self.bins - 1)))
        return max(0, min(self.bins - 1, idx))

    def _bin_to_value(self, idx: int) -> float:
        idx = max(0, min(self.bins - 1, int(idx)))
        scale = idx / max(self.bins - 1, 1)
        return self.cfg.min_value + scale * (self.cfg.max_value - self.cfg.min_value)

    def encode_step(self, dx: float, dy: float, pen_state: str) -> list[int]:
        return [
            self.dx_offset + self._value_to_bin(dx),
            self.dy_offset + self._value_to_bin(dy),
            self.pen_offset + PEN_TO_ID[pen_state],
        ]

    def decode_step(self, dx_token: int, dy_token: int, pen_token: int) -> dict:
        dx_bin = int(dx_token) - self.dx_offset
        dy_bin = int(dy_token) - self.dy_offset
        pen_id = int(pen_token) - self.pen_offset
        return {
            "dx": self._bin_to_value(dx_bin),
            "dy": self._bin_to_value(dy_bin),
            "pen_state": ID_TO_PEN[pen_id],
        }

    def encode_strokes(self, strokes: list[dict]) -> list[int]:
        tokens: list[int] = []
        for step in strokes:
            tokens.extend(self.encode_step(step["dx"], step["dy"], step["pen_state"]))
        return tokens

    def decode_tokens(self, tokens: list[int]) -> list[dict]:
        strokes = []
        usable = len(tokens) - (len(tokens) % 3)
        for idx in range(0, usable, 3):
            dx_token, dy_token, pen_token = tokens[idx : idx + 3]
            if pen_token < self.pen_offset or pen_token >= self.pen_offset + len(PEN_TO_ID):
                break
            step = self.decode_step(dx_token, dy_token, pen_token)
            strokes.append(step)
            if step["pen_state"] == "end_all":
                break
        return strokes

    def valid_token_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Return [*, vocab] mask for action-token type at each autoregressive position.

        Position 0 predicts dx, position 1 predicts dy, position 2 predicts pen,
        and then repeats. This keeps decoding in the valid sub-vocabulary.
        """
        mask = torch.zeros(*positions.shape, self.vocab_size, dtype=torch.bool, device=positions.device)
        phase = positions % 3
        mask[phase == 0, self.dx_offset : self.dx_offset + self.bins] = True
        mask[phase == 1, self.dy_offset : self.dy_offset + self.bins] = True
        mask[phase == 2, self.pen_offset : self.pen_offset + len(PEN_TO_ID)] = True
        return mask
