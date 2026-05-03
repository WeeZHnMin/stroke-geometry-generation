import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import read_jsonl
from .polar_tokenizer import PolarActionTokenizer, PolarActionTokenizerConfig


def _effective_len(max_action_len: int) -> int:
    if max_action_len <= 0:
        raise ValueError("max_action_len must be positive")
    return max_action_len


class PolarActionJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PolarActionTokenizer | None = None,
        max_action_len: int = 192,
        limit: int | None = None,
    ) -> None:
        self.raw_samples = read_jsonl(path, limit=limit)
        if not self.raw_samples:
            raise ValueError(f"No samples found in {path}")

        self.tokenizer = tokenizer or self._tokenizer_from_sample(self.raw_samples[0])
        self.max_action_len = _effective_len(max_action_len)

    @staticmethod
    def _tokenizer_from_sample(sample: dict) -> PolarActionTokenizer:
        metadata = sample.get("metadata", {})
        distances = tuple(metadata.get("distance_buckets", (0.1, 0.2, 0.3, 0.4, 0.5)))
        theta_bins = int(metadata.get("theta_bins", 360))
        return PolarActionTokenizer(PolarActionTokenizerConfig(distance_buckets=distances, theta_bins=theta_bins))

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
        tokens = [int(t) for t in raw["action_tokens"]]
        seq_len = min(len(tokens), self.max_action_len)
        actual = torch.tensor(tokens[:seq_len], dtype=torch.long)

        decoder_input = torch.full((self.max_action_len,), self.tokenizer.end_padding_token(), dtype=torch.long)
        target = torch.full((self.max_action_len,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.max_action_len, dtype=torch.bool)

        decoder_input[0] = self.tokenizer.start_id
        if seq_len > 1:
            decoder_input[1:seq_len] = actual[:-1]

        target[:seq_len] = actual
        target_mask[:seq_len] = True

        return {
            "prompt": str(raw["prompt"]),
            "decoder_input_ids": decoder_input,
            "target_ids": target,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
