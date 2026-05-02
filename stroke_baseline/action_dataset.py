import torch
import warnings
from torch.utils.data import Dataset

from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from .dataset import read_jsonl


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


def _count_range_issues(
    raw_samples: list[dict],
    action_tokenizer: StrokeActionTokenizer,
    *,
    two_stage: bool,
) -> tuple[int, int]:
    total = 0
    out_of_range = 0
    if two_stage:
        lo = action_tokenizer.cfg.draw_min_value
        hi = action_tokenizer.cfg.draw_max_value
    else:
        lo = action_tokenizer.cfg.min_value
        hi = action_tokenizer.cfg.max_value

    for raw in raw_samples:
        strokes = raw["strokes"][1:] if two_stage else raw["strokes"]
        for step in strokes:
            total += 2
            if not (lo <= float(step["dx"]) <= hi):
                out_of_range += 1
            if not (lo <= float(step["dy"]) <= hi):
                out_of_range += 1
    return out_of_range, total


def _warn_if_tokenizer_clamps(raw_samples: list[dict], action_tokenizer: StrokeActionTokenizer, *, two_stage: bool) -> None:
    out_of_range, total = _count_range_issues(raw_samples, action_tokenizer, two_stage=two_stage)
    if out_of_range:
        mode = "two-stage draw" if two_stage else "single-stage"
        warnings.warn(
            f"{out_of_range}/{total} {mode} dx/dy values are outside the tokenizer range and will be clamped. "
            "This can make low token loss incompatible with accurate geometry.",
            stacklevel=2,
        )


def _decoder_input_with_end_padding(action_tokenizer: StrokeActionTokenizer, max_action_len: int) -> torch.Tensor:
    pad_step = action_tokenizer.end_padding_step()
    return torch.tensor(
        [pad_step[pos % 3] for pos in range(max_action_len)],
        dtype=torch.long,
    )


class ActionTokenJsonlDataset(Dataset):
    """原始单阶段数据集：整条 stroke 用同一 tokenizer 范围。"""

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
        _warn_if_tokenizer_clamps(self.raw_samples, self.action_tokenizer, two_stage=False)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
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

        return {
            "prompt": str(raw["prompt"]),
            "decoder_input_ids": decoder_input,
            "target_ids": target,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


class TwoStageActionTokenJsonlDataset(Dataset):
    """双阶段数据集：
       - 第一步 move 的 dx/dy → start_position（回归目标）
       - 后续 draw/end_shape/end_all 步 → tight range token（分类目标）
    """

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
        _warn_if_tokenizer_clamps(self.raw_samples, self.action_tokenizer, two_stage=True)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_samples[idx]
        strokes = raw["strokes"]

        # 第一步一定是 move
        move_step = strokes[0]
        start_position = torch.tensor([move_step["dx"], move_step["dy"]], dtype=torch.float32)
        assert move_step["pen_state"] == "move", f"Expected first step to be move, got {move_step['pen_state']}"

        # 剩余步用 draw 专用 tight range 编码
        draw_strokes = strokes[1:]
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

        return {
            "prompt": str(raw["prompt"]),
            "start_position": start_position,
            "decoder_input_ids": decoder_input,
            "target_ids": target,
            "target_mask": target_mask,
            "length": torch.tensor(seq_len, dtype=torch.long),
        }
