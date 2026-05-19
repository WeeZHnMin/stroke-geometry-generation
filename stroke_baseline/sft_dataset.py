"""Dataset for SFT: pairs Chinese prompts with stroke token sequences."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .action_tokenizer import CompactDxDyTokenizer, DxDyPairTokenizer, DxDyPairTokenizerConfig

_KEEP_PEN = frozenset({"draw", "move"})


def _extract_steps(strokes: list[dict]) -> list[tuple[float, float]]:
    return [
        (float(s["dx"]), float(s["dy"]))
        for s in strokes
        if s.get("pen_state") in _KEEP_PEN
    ]


class SFTJsonlDataset(Dataset):
    """Loads a JSONL dataset with both 'prompt' and 'strokes' fields.

    Returns per sample:
        stroke_input_ids:   [T+1]  BOS + action tokens
        stroke_target_ids:  [T+1]  action tokens + -100
        coords:             [T+1, 2]
        enc_input_ids:      [L_text]
        enc_attention_mask: [L_text]
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: "CompactDxDyTokenizer | DxDyPairTokenizer",
        bert_tokenizer: BertTokenizer,
        max_stroke_len: int = 256,
        max_text_len: int = 64,
        limit: int | None = None,
        min_steps: int = 2,
    ):
        self.stroke_tok = tokenizer
        self.bert_tok = bert_tokenizer
        self.max_stroke_len = max_stroke_len
        self.max_text_len = max_text_len

        self.samples: list[dict] = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                steps = _extract_steps(obj.get("strokes", []))
                if len(steps) < min_steps:
                    continue
                prompt = obj.get("prompt", "")
                self.samples.append({"steps": steps, "prompt": prompt})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        steps = item["steps"]
        prompt = item["prompt"]
        tok = self.stroke_tok

        # --- stroke tokens ---
        ids = [tok.encode_step(dx, dy) for dx, dy in steps]
        ids = ids[: self.max_stroke_len]

        # cumulative (x, y) coords; BOS at origin
        cx, cy = 0.0, 0.0
        xy = [(0.0, 0.0)]
        for dx, dy in steps[: self.max_stroke_len]:
            cx += dx; cy += dy
            xy.append((cx, cy))

        input_ids = torch.tensor([tok.bos_id] + ids, dtype=torch.long)
        target_ids = torch.tensor(ids + [-100], dtype=torch.long)
        coords = torch.tensor(xy, dtype=torch.float32)

        # pad stroke sequence to max_stroke_len + 1
        total = self.max_stroke_len + 1
        pad_len = total - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids,
                                   torch.full((pad_len,), tok.pad_id, dtype=torch.long)])
            target_ids = torch.cat([target_ids,
                                    torch.full((pad_len,), -100, dtype=torch.long)])
            coords = torch.cat([coords, torch.zeros(pad_len, 2)])

        # --- text tokens (BERT) ---
        enc = self.bert_tok(
            prompt,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_input_ids = enc["input_ids"].squeeze(0)            # [L_text]
        enc_attention_mask = enc["attention_mask"].squeeze(0)  # [L_text]

        return {
            "stroke_input_ids":   input_ids,
            "stroke_target_ids":  target_ids,
            "coords":             coords,
            "enc_input_ids":      enc_input_ids,
            "enc_attention_mask": enc_attention_mask,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
