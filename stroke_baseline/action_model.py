import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import ActionTokenizerConfig, StrokeActionTokenizer
from .pretrained_encoder_decoder import (
    DEFAULT_TEXT_ENCODER_DIR,
    CrossAttention,
    FrozenChineseTextEncoder,
    StrokeDecoderConfig,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_2d_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, _, head_dim = q.shape
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")
    dim_1d = head_dim // 2
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, dim_1d, 2, device=q.device, dtype=torch.float32) / dim_1d)
    )
    freqs_x = torch.einsum("bs,d->bsd", coords[:, :, 0].float(), inv_freq)
    freqs_y = torch.einsum("bs,d->bsd", coords[:, :, 1].float(), inv_freq)
    freqs_x = torch.repeat_interleave(freqs_x, 2, dim=-1)
    freqs_y = torch.repeat_interleave(freqs_y, 2, dim=-1)
    freqs = torch.cat([freqs_x, freqs_y], dim=-1).unsqueeze(1)
    sin_val = freqs.sin().to(dtype=q.dtype)
    cos_val = freqs.cos().to(dtype=q.dtype)
    q_embed = (q * cos_val) + (rotate_half(q) * sin_val)
    k_embed = (k * cos_val) + (rotate_half(k) * sin_val)
    return q_embed, k_embed


@dataclass
class ActionDecoderConfig:
    action_vocab_size: int
    pad_token_id: int
    d_model: int = 384
    n_heads: int = 8
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_action_len: int = 384
    attention_variant: str = "hetero"
    trend_kernel_size: int = 5
    input_mode: str = "cpcf"
    xy_hidden_dim: int = 128
    pen_emb_dim: int = 32
    input_kernel_size: int = 3
    use_2d_rope: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def as_stroke_cfg(self) -> StrokeDecoderConfig:
        return StrokeDecoderConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_decoder_layers=self.num_decoder_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_stroke_len=self.max_action_len,
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.left_padding = self.kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

    def forward_step(
        self,
        x: torch.Tensor,
        buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.size(1) != 1:
            raise ValueError("forward_step expects one token: [B, 1, C].")
        if self.left_padding == 0:
            out = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return out, None
        if buffer is None:
            buffer = torch.zeros(x.size(0), self.left_padding, x.size(2), dtype=x.dtype, device=x.device)
        else:
            buffer = buffer[:, -self.left_padding :, :]
        window = torch.cat([buffer, x], dim=1)
        out = self.conv(window.transpose(1, 2)).transpose(1, 2)
        return out, window[:, -self.left_padding :, :]


class CPCFInputEncoder(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        self.xy_mlp = nn.Sequential(
            nn.Linear(2, cfg.xy_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.xy_hidden_dim, cfg.xy_hidden_dim),
        )
        self.pen_emb = nn.Embedding(5, cfg.pen_emb_dim)
        self.fusion = CausalConv1d(cfg.xy_hidden_dim + cfg.pen_emb_dim, cfg.d_model, cfg.input_kernel_size)
        self.out_act = nn.SiLU()

    def forward(self, coords: torch.Tensor, pen_states: torch.Tensor) -> torch.Tensor:
        xy_feat = self.xy_mlp(coords)
        pen_feat = self.pen_emb(pen_states)
        fused = torch.cat([xy_feat, pen_feat], dim=-1)
        return self.out_act(self.fusion(fused))

    def forward_step(
        self,
        coords: torch.Tensor,
        pen_states: torch.Tensor,
        buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        xy_feat = self.xy_mlp(coords)
        pen_feat = self.pen_emb(pen_states)
        fused = torch.cat([xy_feat, pen_feat], dim=-1)
        out, buffer = self.fusion.forward_step(fused, buffer)
        return self.out_act(out), buffer


class LegacyCachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def init_state(self) -> dict[str, torch.Tensor | None]:
        return {"self_k": None, "self_v": None}

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del coords
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        del coords
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        past_k = cache.get("self_k")
        past_v = cache.get("self_v")
        k_all = k if past_k is None else torch.cat([past_k, k], dim=2)
        v_all = v if past_v is None else torch.cat([past_v, v], dim=2)
        scores = (q @ k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        cache["self_k"] = k_all
        cache["self_v"] = v_all
        return self.out(self._merge_heads(attn @ v_all)), cache


class CachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_static_heads = cfg.n_heads // 2
        self.n_trend_heads = cfg.n_heads - self.n_static_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj_static = nn.Linear(cfg.d_model, self.n_static_heads * self.head_dim)
        self.v_proj_static = nn.Linear(cfg.d_model, self.n_static_heads * self.head_dim)
        self.kernel_size = cfg.trend_kernel_size
        self.causal_conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=self.kernel_size, groups=cfg.d_model)
        self.k_proj_trend = nn.Linear(cfg.d_model, self.n_trend_heads * self.head_dim)
        self.v_proj_trend = nn.Linear(cfg.d_model, self.n_trend_heads * self.head_dim)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def init_state(self) -> dict[str, torch.Tensor | None]:
        return {"self_k": None, "self_v": None}

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del coords
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k_static = self._split_heads(self.k_proj_static(x), self.n_static_heads)
        v_static = self._split_heads(self.v_proj_static(x), self.n_static_heads)
        x_conv = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        x_conv = F.silu(self.causal_conv(x_conv).transpose(1, 2))
        k_trend = self._split_heads(self.k_proj_trend(x_conv), self.n_trend_heads)
        v_trend = self._split_heads(self.v_proj_trend(x_conv), self.n_trend_heads)
        k = torch.cat([k_static, k_trend], dim=1)
        v = torch.cat([v_static, v_trend], dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        del coords
        raise NotImplementedError("Streaming cache is only supported for hetero attention.")


class HeterogeneousCachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.n_heads % 2 != 0:
            raise ValueError("hetero attention requires an even number of heads")
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_static_heads = cfg.n_heads // 2
        self.n_trend_heads = cfg.n_heads - self.n_static_heads
        self.use_2d_rope = cfg.use_2d_rope
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj_static = nn.Linear(cfg.d_model, self.n_static_heads * self.head_dim)
        self.v_proj_static = nn.Linear(cfg.d_model, self.n_static_heads * self.head_dim)
        self.causal_conv = CausalConv1d(cfg.d_model, cfg.d_model, kernel_size=cfg.trend_kernel_size)
        self.trend_act = nn.SiLU()
        self.k_proj_trend = nn.Linear(cfg.d_model, self.n_trend_heads * self.head_dim)
        self.v_proj_trend = nn.Linear(cfg.d_model, self.n_trend_heads * self.head_dim)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.zeros_(self.k_proj_trend.weight)
        nn.init.zeros_(self.v_proj_trend.weight)
        if self.k_proj_trend.bias is not None:
            nn.init.zeros_(self.k_proj_trend.bias)
        if self.v_proj_trend.bias is not None:
            nn.init.zeros_(self.v_proj_trend.bias)

    def init_state(self) -> dict[str, torch.Tensor | None]:
        return {"self_k": None, "self_v": None, "conv_buffer": None}

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def _project_qkv(
        self,
        x: torch.Tensor,
        x_conv: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x), self.n_heads)
        q_static = q[:, : self.n_static_heads]
        q_trend = q[:, self.n_static_heads :]
        k_static = self._split_heads(self.k_proj_static(x), self.n_static_heads)
        v_static = self._split_heads(self.v_proj_static(x), self.n_static_heads)
        if coords is not None and self.use_2d_rope:
            q_static, k_static = apply_2d_rotary_pos_emb(q_static, k_static, coords)
        k_trend = self._split_heads(self.k_proj_trend(x_conv), self.n_trend_heads)
        v_trend = self._split_heads(self.v_proj_trend(x_conv), self.n_trend_heads)
        q_all = torch.cat([q_static, q_trend], dim=1)
        k_all = torch.cat([k_static, k_trend], dim=1)
        v_all = torch.cat([v_static, v_trend], dim=1)
        return q_all, k_all, v_all

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_conv = self.trend_act(self.causal_conv(x))
        q, k, v = self._project_qkv(x, x_conv, coords=coords)
        return self._attend(q, k, v, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        x_conv, conv_buffer = self.causal_conv.forward_step(x, cache.get("conv_buffer"))
        x_conv = self.trend_act(x_conv)
        q, k, v = self._project_qkv(x, x_conv, coords=coords)
        past_k = cache.get("self_k")
        past_v = cache.get("self_v")
        k_all = k if past_k is None else torch.cat([past_k, k], dim=2)
        v_all = v if past_v is None else torch.cat([past_v, v], dim=2)
        out = self._attend(q, k_all, v_all)
        cache["self_k"] = k_all
        cache["self_v"] = v_all
        cache["conv_buffer"] = conv_buffer
        return out, cache


def build_self_attention(cfg: ActionDecoderConfig) -> nn.Module:
    if cfg.attention_variant == "legacy_qkv":
        return LegacyCachedTokenSelfAttention(cfg)
    if cfg.attention_variant == "legacy":
        return CachedTokenSelfAttention(cfg)
    if cfg.attention_variant == "hetero":
        return HeterogeneousCachedTokenSelfAttention(cfg)
    raise ValueError(f"Unsupported attention_variant: {cfg.attention_variant}")


class ActionDecoderBlock(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        self.self_attn = build_self_attention(cfg)
        self.cross_attn = CrossAttention(cfg.as_stroke_cfg())
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * cfg.ff_mult, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), causal_mask, padding_mask, coords=coords))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context, context_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

    def init_cross_cache(self, context: torch.Tensor) -> dict[str, torch.Tensor | None]:
        cross_k, cross_v = self.cross_attn.project_kv(context)
        return {**self.self_attn.init_state(), "cross_k": cross_k, "cross_v": cross_v}

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        context_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        self_out, cache = self.self_attn.forward_step(self.norm1(x), cache, coords=coords)
        x = x + self.dropout(self_out)
        cross_k = cache.get("cross_k")
        cross_v = cache.get("cross_v")
        if cross_k is None or cross_v is None:
            raise ValueError("Missing cross-attention cache.")
        x = x + self.dropout(self.cross_attn.forward_with_kv(self.norm2(x), cross_k, cross_v, context_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x, cache


class ActionTokenDecoder(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = None
        self.pos_emb = None
        self.input_encoder = None
        if cfg.input_mode == "token":
            self.token_emb = nn.Embedding(cfg.action_vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
            self.pos_emb = nn.Embedding(cfg.max_action_len, cfg.d_model)
        elif cfg.input_mode == "cpcf":
            self.input_encoder = CPCFInputEncoder(cfg)
        else:
            raise ValueError(f"Unsupported input_mode: {cfg.input_mode}")
        self.blocks = nn.ModuleList([ActionDecoderBlock(cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.action_vocab_size)

    def _build_inputs(
        self,
        decoder_input_ids: torch.Tensor | None,
        decoder_coords: torch.Tensor | None,
        decoder_pen_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.cfg.input_mode == "token":
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids is required for token input mode.")
            seq_len = decoder_input_ids.size(1)
            positions = torch.arange(seq_len, device=decoder_input_ids.device)
            x = self.token_emb(decoder_input_ids) + self.pos_emb(positions)[None, :, :]
            return x, None
        if decoder_coords is None or decoder_pen_states is None:
            raise ValueError("decoder_coords and decoder_pen_states are required for CPCF input mode.")
        return self.input_encoder(decoder_coords, decoder_pen_states), decoder_coords

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_coords: torch.Tensor | None = None,
        decoder_pen_states: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.cfg.input_mode == "token":
            seq_len = decoder_input_ids.size(1)
            padding_mask = decoder_input_ids == self.cfg.pad_token_id
        else:
            seq_len = decoder_coords.size(1)
            padding_mask = decoder_pen_states == 4
        x, coords = self._build_inputs(decoder_input_ids, decoder_coords, decoder_pen_states)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        if target_mask is not None:
            padding_mask = padding_mask | ~target_mask.bool()
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask, context, context_mask, coords=coords)
        hidden = self.norm(x)
        return {"hidden": hidden, "logits": self.lm_head(hidden)}

    def init_cache(self, context: torch.Tensor) -> dict[str, object]:
        return {
            "input_conv_buffer": None,
            "blocks": [block.init_cross_cache(context) for block in self.blocks],
        }

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor | None = None,
        input_coords: torch.Tensor | None = None,
        input_pen_states: torch.Tensor | None = None,
        step_idx: int | torch.Tensor = 0,
        cache: dict[str, object] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        if cache is None:
            cache = self.init_cache(context)
        if self.cfg.input_mode == "token":
            if input_id is None:
                raise ValueError("input_id is required for token input mode.")
            if input_id.dim() == 1:
                input_id = input_id[:, None]
            if isinstance(step_idx, int):
                pos = torch.full((1,), step_idx, device=input_id.device, dtype=torch.long)
            else:
                pos = step_idx.to(device=input_id.device, dtype=torch.long).reshape(1)
            x = self.token_emb(input_id) + self.pos_emb(pos)[None, :, :]
            coords = None
        else:
            if input_coords is None or input_pen_states is None:
                raise ValueError("input_coords and input_pen_states are required for CPCF input mode.")
            if input_coords.dim() == 2:
                input_coords = input_coords[:, None, :]
            if input_pen_states.dim() == 1:
                input_pen_states = input_pen_states[:, None]
            x, input_conv_buffer = self.input_encoder.forward_step(
                input_coords,
                input_pen_states,
                cache.get("input_conv_buffer"),
            )
            cache["input_conv_buffer"] = input_conv_buffer
            coords = input_coords
        new_blocks = []
        for block, layer_cache in zip(self.blocks, cache["blocks"]):
            x, layer_cache = block.forward_step(x, layer_cache, context_mask, coords=coords)
            new_blocks.append(layer_cache)
        cache["blocks"] = new_blocks
        hidden = self.norm(x)
        return {"hidden": hidden, "logits": self.lm_head(hidden)}, cache


class TextConditionedActionModel(nn.Module):
    def __init__(
        self,
        decoder_cfg: ActionDecoderConfig,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = ActionTokenDecoder(decoder_cfg)
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
        decoder_input_ids: torch.Tensor | None = None,
        decoder_coords: torch.Tensor | None = None,
        decoder_pen_states: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.encode_text(prompts)
        return self.decoder(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_coords=decoder_coords,
            decoder_pen_states=decoder_pen_states,
            target_mask=target_mask,
        )

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor | None = None,
        input_coords: torch.Tensor | None = None,
        input_pen_states: torch.Tensor | None = None,
        step_idx: int | torch.Tensor = 0,
        cache: dict[str, object] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        return self.decoder.decode_step(
            context,
            context_mask,
            input_id=input_id,
            input_coords=input_coords,
            input_pen_states=input_pen_states,
            step_idx=step_idx,
            cache=cache,
        )


def build_default_action_model(
    max_action_len: int = 384,
    text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
) -> tuple[TextConditionedActionModel, StrokeActionTokenizer]:
    action_tokenizer = StrokeActionTokenizer(ActionTokenizerConfig())
    cfg = ActionDecoderConfig(
        action_vocab_size=action_tokenizer.vocab_size,
        pad_token_id=action_tokenizer.pad_id,
        max_action_len=max_action_len,
    )
    return TextConditionedActionModel(cfg, text_encoder_dir=text_encoder_dir), action_tokenizer
