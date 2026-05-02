from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .dataset import NUM_INPUT_PEN_STATES, NUM_TARGET_PEN_STATES


@dataclass
class StrokeTransformerConfig:
    vocab_size: int
    max_text_len: int = 96
    max_stroke_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class StrokeTransformerBaseline(nn.Module):
    def __init__(self, cfg: StrokeTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.text_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.text_pos = nn.Embedding(cfg.max_text_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        self.stroke_proj = nn.Linear(2, cfg.d_model)
        self.pen_emb = nn.Embedding(NUM_INPUT_PEN_STATES, cfg.d_model)
        self.stroke_pos = nn.Embedding(cfg.max_stroke_len, cfg.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.delta_head = nn.Linear(cfg.d_model, 2)
        self.pen_head = nn.Linear(cfg.d_model, NUM_TARGET_PEN_STATES)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, text_len = text_ids.shape
        stroke_len = decoder_dxdy.size(1)

        text_pos = torch.arange(text_len, device=text_ids.device)
        text_x = self.text_emb(text_ids) + self.text_pos(text_pos)[None, :, :]
        memory = self.encoder(text_x, src_key_padding_mask=~text_mask)

        stroke_pos = torch.arange(stroke_len, device=decoder_dxdy.device)
        stroke_x = (
            self.stroke_proj(decoder_dxdy)
            + self.pen_emb(decoder_pen)
            + self.stroke_pos(stroke_pos)[None, :, :]
        )

        causal_mask = torch.triu(
            torch.ones(stroke_len, stroke_len, device=decoder_dxdy.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_padding_mask = None if target_mask is None else ~target_mask
        hidden = self.decoder(
            tgt=stroke_x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=~text_mask,
        )
        hidden = self.norm(hidden)
        return {
            "pred_dxdy": self.delta_head(hidden),
            "pred_pen_logits": self.pen_head(hidden),
        }

