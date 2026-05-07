from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from stroke_baseline.pretrained_encoder_decoder import CrossAttention, DEFAULT_TEXT_ENCODER_DIR, FrozenChineseTextEncoder

from .dataset import STEP_DIM


@dataclass
class StrokeDiffusionConfig:
    step_dim: int = STEP_DIM
    d_model: int = 384
    n_heads: int = 8
    num_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 192
    max_text_len: int = 64
    predict_target: str = "x0"

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            timesteps = timesteps.reshape(-1)
        half = self.d_model // 2
        device = timesteps.device
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.d_model:
            emb = torch.nn.functional.pad(emb, (0, self.d_model - emb.size(-1)))
        return self.proj(emb)


class StepEmbedding(nn.Module):
    """Embed one stroke step [x, y, pen_id]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.step_embed = nn.Sequential(
            nn.Linear(STEP_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != STEP_DIM:
            raise ValueError(f"Expected last dim={STEP_DIM}, got {x.size(-1)}")
        return self.step_embed(x)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, cfg: StrokeDiffusionConfig):
        super().__init__()
        stroke_cfg = type("TmpCfg", (), {"d_model": cfg.d_model, "n_heads": cfg.n_heads, "dropout": cfg.dropout})()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn = CrossAttention(stroke_cfg)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * cfg.ff_mult, cfg.d_model),
        )
        self.time_scale = nn.Linear(cfg.d_model, cfg.d_model)
        self.time_shift = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def apply_time_condition(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale = self.time_scale(time_emb)[:, None, :]
        shift = self.time_shift(time_emb)[:, None, :]
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_emb: torch.Tensor,
        seq_mask: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.apply_time_condition(self.norm1(x), time_emb)
        key_padding_mask = None if seq_mask is None else ~seq_mask.bool()
        self_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(self_out)

        h = self.apply_time_condition(self.norm2(x), time_emb)
        x = x + self.dropout(self.cross_attn(h, context, context_mask))

        h = self.apply_time_condition(self.norm3(x), time_emb)
        x = x + self.dropout(self.ffn(h))
        return x


class StrokeDiffusionTransformer(nn.Module):
    def __init__(self, cfg: StrokeDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.step_embed = StepEmbedding(cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.time_embed = SinusoidalTimeEmbedding(cfg.d_model)
        self.blocks = nn.ModuleList([DiffusionTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.step_dim),
        )

    def forward(
        self,
        noisy_steps: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        seq_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, step_dim = noisy_steps.shape
        if step_dim != self.cfg.step_dim:
            raise ValueError(f"Expected step dim {self.cfg.step_dim}, got {step_dim}")
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos = torch.arange(seq_len, device=noisy_steps.device)
        x = self.step_embed(noisy_steps) + self.pos_emb(pos)[None, :, :]
        time_emb = self.time_embed(timesteps)
        x = x + time_emb[:, None, :]

        for block in self.blocks:
            x = block(
                x,
                time_emb=time_emb,
                seq_mask=seq_mask,
                context=context,
                context_mask=context_mask,
            )

        hidden = self.norm(x)
        pred = self.out(hidden)
        return {
            "hidden": hidden,
            "pred_steps": pred,
        }


class TextConditionedStrokeDiffusionModel(nn.Module):
    def __init__(
        self,
        cfg: StrokeDiffusionConfig | None = None,
        *,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
    ):
        super().__init__()
        self.cfg = cfg or StrokeDiffusionConfig()
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.denoiser = StrokeDiffusionTransformer(self.cfg)
        if self.text_encoder.hidden_size != self.cfg.d_model:
            self.context_proj = nn.Linear(self.text_encoder.hidden_size, self.cfg.d_model)
        else:
            self.context_proj = nn.Identity()

    def encode_text(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.cfg.max_text_len)
        return {
            "context": self.context_proj(text["context"]),
            "context_mask": text["context_mask"],
        }

    def forward(
        self,
        *,
        prompts: list[str],
        noisy_steps: torch.Tensor,
        timesteps: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        out = self.denoiser(
            noisy_steps=noisy_steps,
            timesteps=timesteps,
            context=text["context"],
            context_mask=text["context_mask"],
            seq_mask=seq_mask,
        )
        return {
            "context": text["context"],
            "context_mask": text["context_mask"],
            **out,
        }


if __name__ == "__main__":
    model = TextConditionedStrokeDiffusionModel(
        StrokeDiffusionConfig(
            d_model=256,
            n_heads=8,
            num_layers=4,
            max_seq_len=128,
        )
    )
    batch_size = 2
    seq_len = 64
    noisy_steps = torch.randn(batch_size, seq_len, STEP_DIM)
    timesteps = torch.randint(0, 1000, (batch_size,))
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    out = model(
        prompts=["draw a large circle near the center", "draw a small rectangle on the right"],
        noisy_steps=noisy_steps,
        timesteps=timesteps,
        seq_mask=seq_mask,
    )
    print("pred_steps:", tuple(out["pred_steps"].shape))
