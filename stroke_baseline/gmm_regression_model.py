import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR, DecoderBlock, FrozenChineseTextEncoder, StrokeDecoderConfig
from .regression_dataset import NUM_REGRESSION_PEN_STATES


@dataclass
class GMMCoordDecoderConfig:
    d_model: int = 384
    n_heads: int = 8
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 96
    max_delta: float = 0.5
    n_components: int = 5
    min_sigma: float = 1e-3
    max_sigma: float = 1.0
    n_pen_states: int = NUM_REGRESSION_PEN_STATES

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


class GMMCoordDecoder(nn.Module):
    def __init__(self, cfg: GMMCoordDecoderConfig):
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
        self.pi_head = nn.Linear(cfg.d_model, cfg.n_components)
        self.mu_head = nn.Linear(cfg.d_model, cfg.n_components * 2)
        self.sigma_head = nn.Linear(cfg.d_model, cfg.n_components * 2)
        self.pen_head = nn.Linear(cfg.d_model, cfg.n_pen_states)

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
        batch, steps, _ = hidden.shape
        pi_logits = self.pi_head(hidden)
        mu = self.cfg.max_delta * torch.tanh(self.mu_head(hidden).view(batch, steps, self.cfg.n_components, 2))
        sigma = F.softplus(self.sigma_head(hidden).view(batch, steps, self.cfg.n_components, 2)) + self.cfg.min_sigma
        sigma = sigma.clamp(max=self.cfg.max_sigma)
        pen_logits = self.pen_head(hidden)
        return {"hidden": hidden, "pi_logits": pi_logits, "mu": mu, "sigma": sigma, "pen_logits": pen_logits}


class TextConditionedGMMCoordModel(nn.Module):
    def __init__(
        self,
        cfg: GMMCoordDecoderConfig,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = GMMCoordDecoder(cfg)
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


def gmm_nll(out: dict[str, torch.Tensor], target_dxdy: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    pi_logits = out["pi_logits"]
    mu = out["mu"]
    sigma = out["sigma"]
    target = target_dxdy.unsqueeze(2)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    z = (target - mu) / sigma
    log_prob_xy = -0.5 * (z.square() + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))
    log_prob = log_prob_xy.sum(dim=-1)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)
    valid = target_mask.bool()
    if not valid.any():
        return log_mix.sum() * 0.0
    return -log_mix[valid].mean()


def decode_gmm(out: dict[str, torch.Tensor], strategy: str = "top") -> torch.Tensor:
    pi_logits = out["pi_logits"]
    mu = out["mu"]
    if strategy == "mean":
        weights = torch.softmax(pi_logits, dim=-1)
        return (weights.unsqueeze(-1) * mu).sum(dim=2)
    if strategy != "top":
        raise ValueError(f"unknown GMM decode strategy: {strategy}")
    idx = pi_logits.argmax(dim=-1)
    gather_idx = idx[..., None, None].expand(-1, -1, 1, 2)
    return mu.gather(dim=2, index=gather_idx).squeeze(2)


def gmm_summary_metrics(out: dict[str, torch.Tensor], target_mask: torch.Tensor) -> dict[str, float]:
    valid = target_mask.bool()
    if not valid.any():
        return {"avg_sigma": float("nan"), "pi_entropy": float("nan")}
    weights = torch.softmax(out["pi_logits"], dim=-1)
    entropy = -(weights * (weights.clamp_min(1e-12).log())).sum(dim=-1)
    return {
        "avg_sigma": float(out["sigma"][valid].mean().item()),
        "pi_entropy": float(entropy[valid].mean().item()),
    }
