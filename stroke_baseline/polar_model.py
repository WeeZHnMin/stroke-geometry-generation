from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .action_model import ActionDecoderConfig, ActionTokenDecoder
from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR, FrozenChineseTextEncoder


@dataclass
class PolarDecoderConfig:
    vocab_size: int
    pad_token_id: int
    d_model: int = 384
    n_heads: int = 8
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_action_len: int = 192
    attention_variant: str = "legacy_qkv"
    trend_kernel_size: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    def as_action_cfg(self) -> ActionDecoderConfig:
        return ActionDecoderConfig(
            action_vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_decoder_layers=self.num_decoder_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_action_len=self.max_action_len,
            attention_variant=self.attention_variant,
            trend_kernel_size=self.trend_kernel_size,
        )


class TextConditionedPolarModel(nn.Module):
    def __init__(
        self,
        cfg: PolarDecoderConfig,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = ActionTokenDecoder(cfg.as_action_cfg())
        if self.text_encoder.hidden_size != cfg.d_model:
            self.context_proj = nn.Linear(self.text_encoder.hidden_size, cfg.d_model)
        else:
            self.context_proj = nn.Identity()

    def encode_text(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.max_text_len)
        return {"context": self.context_proj(text["context"]), "context_mask": text["context_mask"]}

    def forward(
        self,
        prompts: list[str],
        decoder_input_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        return self.decoder(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_input_ids=decoder_input_ids,
            target_mask=target_mask,
        )

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        return self.decoder.decode_step(context, context_mask, input_id, step_idx, cache)
