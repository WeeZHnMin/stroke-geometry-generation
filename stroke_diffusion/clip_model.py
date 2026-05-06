from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stroke_baseline.pretrained_encoder_decoder import DEFAULT_TEXT_ENCODER_DIR, FrozenChineseTextEncoder
from stroke_diffusion.model import STEP_DIM, StepEmbedding


@dataclass
class StrokeCLIPConfig:
    step_dim: int = STEP_DIM
    d_model: int = 384
    n_heads: int = 8
    num_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 192
    max_text_len: int = 64
    proj_dim: int = 256
    init_logit_scale: float = 1 / 0.07

    def to_dict(self) -> dict:
        return asdict(self)


class SequenceTransformerBlock(nn.Module):
    def __init__(self, cfg: StrokeCLIPConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * cfg.ff_mult, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None if seq_mask is None else ~seq_mask.bool()
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class StrokeSequenceEncoder(nn.Module):
    def __init__(self, cfg: StrokeCLIPConfig):
        super().__init__()
        self.cfg = cfg
        self.step_embed = StepEmbedding(cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([SequenceTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.proj_dim),
        )

    def masked_mean_pool(self, hidden: torch.Tensor, seq_mask: torch.Tensor | None) -> torch.Tensor:
        if seq_mask is None:
            return hidden.mean(dim=1)
        mask = seq_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(self, steps: torch.Tensor, seq_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, seq_len, step_dim = steps.shape
        if step_dim != self.cfg.step_dim:
            raise ValueError(f"Expected step dim {self.cfg.step_dim}, got {step_dim}")
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos = torch.arange(seq_len, device=steps.device)
        x = self.step_embed(steps) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, seq_mask=seq_mask)
        hidden = self.norm(x)
        pooled = self.masked_mean_pool(hidden, seq_mask)
        embedding = self.proj(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return {
            "hidden": hidden,
            "pooled": pooled,
            "embedding": embedding,
        }


class TextProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class TextStrokeCLIPModel(nn.Module):
    def __init__(
        self,
        cfg: StrokeCLIPConfig | None = None,
        *,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
    ):
        super().__init__()
        self.cfg = cfg or StrokeCLIPConfig()
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.stroke_encoder = StrokeSequenceEncoder(self.cfg)
        self.text_proj = TextProjectionHead(self.text_encoder.hidden_size, self.cfg.proj_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(self.cfg.init_logit_scale))))

    def encode_text(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.cfg.max_text_len)
        cls = text["context"][:, 0, :]
        embedding = self.text_proj(cls)
        return {
            "context": text["context"],
            "context_mask": text["context_mask"],
            "pooled": cls,
            "embedding": embedding,
        }

    def encode_strokes(self, steps: torch.Tensor, seq_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.stroke_encoder(steps, seq_mask=seq_mask)

    def forward(
        self,
        *,
        prompts: list[str],
        steps: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        stroke = self.encode_strokes(steps, seq_mask=seq_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_text = logit_scale * (text["embedding"] @ stroke["embedding"].transpose(0, 1))
        logits_per_stroke = logits_per_text.transpose(0, 1)
        return {
            "text_embedding": text["embedding"],
            "stroke_embedding": stroke["embedding"],
            "logits_per_text": logits_per_text,
            "logits_per_stroke": logits_per_stroke,
            "logit_scale": logit_scale,
        }


def clip_contrastive_loss(logits_per_text: torch.Tensor, logits_per_stroke: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    batch = logits_per_text.size(0)
    targets = torch.arange(batch, device=logits_per_text.device)
    loss_text = F.cross_entropy(logits_per_text, targets)
    loss_stroke = F.cross_entropy(logits_per_stroke, targets)
    loss = 0.5 * (loss_text + loss_stroke)

    with torch.no_grad():
        text_acc = (logits_per_text.argmax(dim=-1) == targets).float().mean()
        stroke_acc = (logits_per_stroke.argmax(dim=-1) == targets).float().mean()

    metrics = {
        "loss": float(loss.item()),
        "loss_text": float(loss_text.item()),
        "loss_stroke": float(loss_stroke.item()),
        "text_acc": float(text_acc.item()),
        "stroke_acc": float(stroke_acc.item()),
    }
    return loss, metrics


if __name__ == "__main__":
    model = TextStrokeCLIPModel(
        StrokeCLIPConfig(
            d_model=128,
            n_heads=8,
            num_layers=2,
            max_seq_len=32,
            proj_dim=64,
        )
    )
    steps = torch.randn(2, 16, STEP_DIM)
    seq_mask = torch.ones(2, 16, dtype=torch.bool)
    out = model(
        prompts=["draw a large circle near the center", "draw a small rectangle on the right"],
        steps=steps,
        seq_mask=seq_mask,
    )
    loss, metrics = clip_contrastive_loss(out["logits_per_text"], out["logits_per_stroke"])
    print("logits_per_text:", tuple(out["logits_per_text"].shape))
    print("loss:", float(loss.item()))
    print("metrics:", metrics)
