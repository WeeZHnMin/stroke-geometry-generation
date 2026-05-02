from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from stroke_baseline.action_model import ActionDecoderBlock
from stroke_baseline.action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR, FrozenChineseTextEncoder

from .geometry_dataset import GEOMETRY_STATE_DIM


DEFAULT_GEOMETRY_ACTION_BINS = 500


@dataclass
class GeometryActionDecoderConfig:
    action_vocab_size: int
    pad_token_id: int
    d_model: int = 384
    n_heads: int = 8
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_action_len: int = 384
    geometry_dim: int = GEOMETRY_STATE_DIM

    def to_dict(self) -> dict:
        return asdict(self)

    def as_action_cfg(self):
        from stroke_baseline.action_model import ActionDecoderConfig

        return ActionDecoderConfig(
            action_vocab_size=self.action_vocab_size,
            pad_token_id=self.pad_token_id,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_decoder_layers=self.num_decoder_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_action_len=self.max_action_len,
        )


class GeometryActionTokenDecoder(nn.Module):
    def __init__(self, cfg: GeometryActionDecoderConfig):
        super().__init__()
        self.cfg = cfg
        action_cfg = cfg.as_action_cfg()
        self.token_emb = nn.Embedding(cfg.action_vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_action_len, cfg.d_model)
        self.phase_emb = nn.Embedding(3, cfg.d_model)
        self.geometry_proj = nn.Sequential(
            nn.Linear(cfg.geometry_dim, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([ActionDecoderBlock(action_cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.action_vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        geometry_states: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = decoder_input_ids.size(1)
        positions = torch.arange(seq_len, device=decoder_input_ids.device)
        phase = positions % 3
        x = (
            self.token_emb(decoder_input_ids)
            + self.pos_emb(positions)[None, :, :]
            + self.phase_emb(phase)[None, :, :]
            + self.geometry_proj(geometry_states)
        )
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = decoder_input_ids == self.cfg.pad_token_id
        if target_mask is not None:
            padding_mask = padding_mask | ~target_mask.bool()

        for block in self.blocks:
            x = block(x, causal_mask, padding_mask, context, context_mask)
        hidden = self.norm(x)
        return {"hidden": hidden, "logits": self.lm_head(hidden)}

    def init_cache(self, context: torch.Tensor) -> list[dict[str, torch.Tensor | None]]:
        return [block.init_cross_cache(context) for block in self.blocks]

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor,
        geometry_state: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        if input_id.dim() == 1:
            input_id = input_id[:, None]
        if geometry_state.dim() == 2:
            geometry_state = geometry_state[:, None, :]
        if cache is None:
            cache = self.init_cache(context)
        if isinstance(step_idx, int):
            pos = torch.full((1,), step_idx, device=input_id.device, dtype=torch.long)
        else:
            pos = step_idx.to(device=input_id.device, dtype=torch.long).reshape(1)
        phase = pos % 3
        x = (
            self.token_emb(input_id)
            + self.pos_emb(pos)[None, :, :]
            + self.phase_emb(phase)[None, :, :]
            + self.geometry_proj(geometry_state)
        )
        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, layer_cache = block.forward_step(x, layer_cache, context_mask)
            new_cache.append(layer_cache)
        hidden = self.norm(x)
        return {"hidden": hidden, "logits": self.lm_head(hidden)}, new_cache


class TextConditionedGeometryActionModel(nn.Module):
    def __init__(
        self,
        decoder_cfg: GeometryActionDecoderConfig,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = GeometryActionTokenDecoder(decoder_cfg)
        if self.text_encoder.hidden_size != decoder_cfg.d_model:
            self.context_proj = nn.Linear(self.text_encoder.hidden_size, decoder_cfg.d_model)
        else:
            self.context_proj = nn.Identity()

    def encode_text(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.max_text_len)
        return {"context": self.context_proj(text["context"]), "context_mask": text["context_mask"]}

    def forward(
        self,
        prompts: list[str],
        decoder_input_ids: torch.Tensor,
        geometry_states: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        return self.decoder(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_input_ids=decoder_input_ids,
            geometry_states=geometry_states,
            target_mask=target_mask,
        )

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor,
        geometry_state: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        return self.decoder.decode_step(context, context_mask, input_id, geometry_state, step_idx, cache)


def build_default_geometry_action_model(
    max_action_len: int = 384,
    text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
) -> tuple[TextConditionedGeometryActionModel, StrokeActionTokenizer]:
    action_tokenizer = StrokeActionTokenizer(ActionTokenizerConfig(bins=DEFAULT_GEOMETRY_ACTION_BINS))
    cfg = GeometryActionDecoderConfig(
        action_vocab_size=action_tokenizer.vocab_size,
        pad_token_id=action_tokenizer.pad_id,
        max_action_len=max_action_len,
    )
    return TextConditionedGeometryActionModel(cfg, text_encoder_dir=text_encoder_dir), action_tokenizer
