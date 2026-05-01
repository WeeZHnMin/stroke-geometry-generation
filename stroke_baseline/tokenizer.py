import json
from pathlib import Path
from typing import Iterable, List

from transformers import AutoTokenizer


class CharTokenizer:
    """Tiny character-level tokenizer for the first training loop."""

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, stoi: dict[str, int] | None = None):
        if stoi is None:
            stoi = {self.pad_token: 0, self.unk_token: 1}
        self.stoi = dict(stoi)
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, texts: Iterable[str], lowercase: bool = True) -> "CharTokenizer":
        chars = set()
        for text in texts:
            if lowercase:
                text = text.lower()
            chars.update(text)

        stoi = {cls.pad_token: 0, cls.unk_token: 1}
        for ch in sorted(chars):
            if ch not in stoi:
                stoi[ch] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, max_len: int, lowercase: bool = True) -> List[int]:
        if lowercase:
            text = text.lower()
        ids = [self.stoi.get(ch, self.unk_id) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            if idx == self.pad_id:
                continue
            chars.append(self.itos.get(int(idx), self.unk_token))
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.stoi, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")))


class HFTokenizer:
    """Small wrapper around a HuggingFace tokenizer."""

    def __init__(self, name_or_path: str | Path):
        self.name_or_path = str(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def unk_id(self) -> int:
        return int(self.tokenizer.unk_token_id) if self.tokenizer.unk_token_id is not None else self.pad_id

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode(self, text: str, max_len: int, lowercase: bool = True) -> List[int]:
        if lowercase:
            text = text.lower()
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )
        return list(encoded.input_ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path) -> "HFTokenizer":
        return cls(path)
