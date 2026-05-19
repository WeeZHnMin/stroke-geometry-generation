"""Text-conditioned stroke decoder (SFT phase).

Architecture:
    BERT (frozen) → Adapter_i (trainable, per layer) → encoder_hidden [B, L_text, 768]
                         ↓
    stroke tokens → Embedding → CausalConv → Self-Attn → Cross-Attn → FFN → logits

Trainable weights:
    - Adapter modules injected after each BERT layer  (bottleneck MLP + residual)
    - Cross-attention sublayers in every decoder block
    - All decoder weights (embedding, conv, self-attn, FFN, lm_head)

BERT original weights stay frozen; adapters let the encoder adapt to the
stroke-generation domain without expensive full fine-tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel

from .action_model import CausalConv1d, apply_2d_rotary_pos_emb
from .dxdy_decoder_model import DxDyDecoderConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # decoder (must match the pretrained checkpoint)
    vocab_size: int = 10002
    pad_token_id: int = 10001
    bos_token_id: int = 10000
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512
    conv_kernel_size: int = 12
    # encoder
    bert_path: str = "models/bert-base-chinese"
    encoder_dim: int = 768
    adapter_dim: int = 64   # bottleneck size per adapter; 0 = no adapters (full freeze)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_decoder_cfg(self) -> DxDyDecoderConfig:
        return DxDyDecoderConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            conv_kernel_size=self.conv_kernel_size,
        )


# ---------------------------------------------------------------------------
# Adapter module (injected after each BERT layer via forward hook)
# ---------------------------------------------------------------------------

class _Adapter(nn.Module):
    """Bottleneck adapter: LayerNorm → down → GELU → up → residual."""

    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, d_model)
        # initialise up-projection near zero so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


def _inject_adapters(bert: BertModel, adapter_dim: int) -> nn.ModuleList:
    """Freeze BERT weights and register per-layer adapter hooks.

    Returns the ModuleList of adapters so they are tracked by the parent module.
    """
    for p in bert.parameters():
        p.requires_grad = False

    d_model = bert.config.hidden_size
    n_layers = bert.config.num_hidden_layers
    adapters = nn.ModuleList([_Adapter(d_model, adapter_dim) for _ in range(n_layers)])

    def _make_hook(adapter: _Adapter):
        def hook(module, inputs, output):
            # Newer transformers may return a bare Tensor instead of a tuple
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    for layer, adapter in zip(bert.encoder.layer, adapters):
        layer.register_forward_hook(_make_hook(adapter))

    return adapters


# ---------------------------------------------------------------------------
# Decoder block with cross-attention
# ---------------------------------------------------------------------------

class _SFTDecoderBlock(nn.Module):
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert (cfg.d_model // cfg.n_heads) % 4 == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        # --- self-attention (same as _DecoderBlock) ---
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # --- cross-attention (new) ---
        self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.cross_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.cross_kv = nn.Linear(cfg.encoder_dim, cfg.d_model * 2)
        self.cross_out = nn.Linear(cfg.d_model, cfg.d_model)

        # --- FFN ---
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * cfg.ff_mult, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_pad_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # --- self-attention ---
        normed = self.norm1(x)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if coords is not None:
            q, k = apply_2d_rotary_pos_emb(q, k, coords)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(causal_mask[None, None], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None], torch.finfo(scores.dtype).min)
        sa_out = (self.dropout(torch.softmax(scores, dim=-1)) @ v)
        sa_out = sa_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.dropout(self.out_proj(sa_out))

        # --- cross-attention ---
        normed_c = self.norm_cross(x)
        cq = self.cross_q(normed_c)                         # [B, T, d]
        ck, cv = self.cross_kv(encoder_hidden).chunk(2, -1) # [B, L_text, d] each
        cq = cq.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        ck = ck.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        cv = cv.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        c_scores = (cq @ ck.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if encoder_pad_mask is not None:
            # encoder_pad_mask: [B, L_text], True where padding
            c_scores = c_scores.masked_fill(
                encoder_pad_mask[:, None, None, :], torch.finfo(c_scores.dtype).min
            )
        ca_out = (self.dropout(torch.softmax(c_scores, dim=-1)) @ cv)
        ca_out = ca_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.dropout(self.cross_out(ca_out))

        # --- FFN ---
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Full SFT model
# ---------------------------------------------------------------------------

class SFTModel(nn.Module):
    """BERT encoder + cross-attention stroke decoder."""

    def __init__(self, cfg: SFTConfig):
        super().__init__()
        self.cfg = cfg

        # encoder: BERT always frozen; adapters are trainable if adapter_dim > 0
        self.encoder = BertModel.from_pretrained(cfg.bert_path)
        if cfg.adapter_dim > 0:
            self.encoder_adapters = _inject_adapters(self.encoder, cfg.adapter_dim)
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder_adapters = nn.ModuleList()

        # decoder
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                      padding_idx=cfg.pad_token_id)
        self.input_conv = CausalConv1d(cfg.d_model, cfg.d_model,
                                       kernel_size=cfg.conv_kernel_size)
        self.blocks = nn.ModuleList([_SFTDecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    # ------------------------------------------------------------------

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (encoder_hidden [B,L,768], encoder_pad_mask [B,L])."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                         # [B, L, 768]
        pad_mask = attention_mask == 0                         # True where padding
        return hidden, pad_mask

    def forward(
        self,
        stroke_input_ids: torch.Tensor,
        enc_input_ids: torch.Tensor,
        enc_attention_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        stroke_input_ids: [B, T]
        enc_input_ids:    [B, L_text]
        enc_attention_mask: [B, L_text]
        returns logits: [B, T, vocab_size]
        """
        encoder_hidden, encoder_pad_mask = self.encode_text(enc_input_ids, enc_attention_mask)

        B, T = stroke_input_ids.shape
        x = self.input_conv(self.token_emb(stroke_input_ids))

        causal_mask = torch.triu(
            torch.ones(T, T, device=stroke_input_ids.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = stroke_input_ids == self.cfg.pad_token_id

        for block in self.blocks:
            x = block(x, causal_mask, encoder_hidden,
                      encoder_pad_mask=encoder_pad_mask,
                      key_padding_mask=key_padding_mask,
                      coords=coords)

        return self.lm_head(self.norm(x))

    # ------------------------------------------------------------------

    def load_pretrained_decoder(self, checkpoint_path: str | Path) -> None:
        """Load decoder weights from a DxDyDecoder checkpoint.

        Cross-attention weights (_SFTDecoderBlock.norm_cross / cross_*) are
        intentionally skipped — they are randomly initialised as new weights.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        pretrained = ckpt["model"]

        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in pretrained.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(own)
        print(f"Loaded {len(loaded)} decoder tensors, skipped {len(skipped)} "
              f"(cross-attn / shape mismatch).", flush=True)
