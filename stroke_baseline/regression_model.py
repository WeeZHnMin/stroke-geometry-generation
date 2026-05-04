from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR, DecoderBlock, FrozenChineseTextEncoder, StrokeDecoderConfig


@dataclass
class CoordRegressionDecoderConfig:
    d_model: int = 384
    n_heads: int = 8
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 96
    max_delta: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    def as_stroke_cfg(self) -> StrokeDecoderConfig:
        return StrokeDecoderConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_decoder_layers=self.num_decoder_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_stroke_len=self.max_len,
        )


class CoordRegressionDecoder(nn.Module):
    def __init__(self, cfg: CoordRegressionDecoderConfig):
        super().__init__()
        self.cfg = cfg
        stroke_cfg = cfg.as_stroke_cfg()
        self.coord_proj = nn.Sequential(
            nn.Linear(3, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([DecoderBlock(stroke_cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.delta_head = nn.Linear(cfg.d_model, 2)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_coords: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = decoder_coords.size(1)
        x = self.coord_proj(decoder_coords)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_coords.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = None if target_mask is None else ~target_mask.bool()
        for block in self.blocks:
            x = block(
                x=x,
                causal_mask=causal_mask,
                stroke_padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
            )
        hidden = self.norm(x)
        pred_dxdy = self.cfg.max_delta * torch.tanh(self.delta_head(hidden))
        return {"hidden": hidden, "pred_dxdy": pred_dxdy}


class TextConditionedCoordRegressionModel(nn.Module):
    def __init__(
        self,
        cfg: CoordRegressionDecoderConfig,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = CoordRegressionDecoder(cfg)
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
        decoder_coords: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        return self.decoder(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_coords=decoder_coords,
            target_mask=target_mask,
        )
