import math
from dataclasses import asdict, dataclass

from .dataset import ID_TO_PEN, PEN_TO_ID


@dataclass
class PolarActionTokenizerConfig:
    distance_buckets: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    theta_bins: int = 360

    def to_dict(self) -> dict:
        data = asdict(self)
        data["distance_buckets"] = list(self.distance_buckets)
        return data


class PolarActionTokenizer:
    """One-token-per-stroke-action tokenizer: (distance, theta, pen_state)."""

    def __init__(self, cfg: PolarActionTokenizerConfig | None = None):
        self.cfg = cfg or PolarActionTokenizerConfig()
        self.distance_buckets = tuple(float(v) for v in self.cfg.distance_buckets)
        self.theta_bins = int(self.cfg.theta_bins)
        self.num_pen_states = len(PEN_TO_ID)
        self.action_vocab_size = len(self.distance_buckets) * self.theta_bins * self.num_pen_states
        self.start_id = self.action_vocab_size
        self.pad_id = self.action_vocab_size + 1
        self.vocab_size = self.action_vocab_size + 2

    def encode_action(self, distance_id: int, theta_id: int, pen_state: str) -> int:
        distance_id = max(0, min(len(self.distance_buckets) - 1, int(distance_id)))
        theta_id = int(theta_id) % self.theta_bins
        pen_id = PEN_TO_ID[pen_state]
        return ((distance_id * self.theta_bins) + theta_id) * self.num_pen_states + pen_id

    def decode_action(self, token_id: int) -> dict:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.action_vocab_size:
            raise ValueError(f"not a polar action token: {token_id}")
        pen_id = token_id % self.num_pen_states
        tmp = token_id // self.num_pen_states
        theta_id = tmp % self.theta_bins
        distance_id = tmp // self.theta_bins
        distance = self.distance_buckets[distance_id]
        theta = 2 * math.pi * theta_id / self.theta_bins
        return {
            "distance_id": distance_id,
            "distance": distance,
            "theta_id": theta_id,
            "theta": theta,
            "pen_state": ID_TO_PEN[pen_id],
        }

    def token_to_stroke(self, token_id: int) -> dict:
        action = self.decode_action(token_id)
        distance = action["distance"]
        theta = action["theta"]
        return {
            "dx": distance * math.cos(theta),
            "dy": distance * math.sin(theta),
            "pen_state": action["pen_state"],
        }

    def decode_tokens(self, tokens: list[int]) -> list[dict]:
        strokes = []
        for token in tokens:
            if token < 0 or token >= self.action_vocab_size:
                break
            stroke = self.token_to_stroke(token)
            strokes.append(stroke)
            if stroke["pen_state"] == "end_all":
                break
        return strokes

    def end_padding_token(self) -> int:
        return self.encode_action(0, 0, "end_all")
