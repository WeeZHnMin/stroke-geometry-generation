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
    attention_variant: str = "legacy"
    trend_kernel_size: int = 5

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

class LegacyCachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
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
    ) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        past_k = cache.get("self_k")
        past_v = cache.get("self_v")
        k_all = k if past_k is None else torch.cat([past_k, k], dim=2)
        v_all = v if past_v is None else torch.cat([past_v, v], dim=2)
        scores = (q @ k_all.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        cache["self_k"] = k_all
        cache["self_v"] = v_all
        return self.out(self._merge_heads(attn @ v_all)), cache


class CachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.n_static_heads = cfg.n_heads // 2
        self.n_trend_heads = cfg.n_heads - self.n_static_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.k_proj_static = nn.Linear(
            cfg.d_model,
            self.n_static_heads * self.head_dim,
        )
        self.v_proj_static = nn.Linear(
            cfg.d_model,
            self.n_static_heads * self.head_dim,
        )

        self.kernel_size = 5
        self.causal_conv = nn.Conv1d(
            in_channels=cfg.d_model,
            out_channels=cfg.d_model,
            kernel_size=self.kernel_size,
            groups=cfg.d_model,
        )
        self.k_proj_trend = nn.Linear(
            cfg.d_model,
            self.n_trend_heads * self.head_dim,
        )
        self.v_proj_trend = nn.Linear(
            cfg.d_model,
            self.n_trend_heads * self.head_dim,
        )

        nn.init.zeros_(self.k_proj_trend.weight)
        nn.init.zeros_(self.v_proj_trend.weight)
        if self.k_proj_trend.bias is not None:
            nn.init.zeros_(self.k_proj_trend.bias)
        if self.v_proj_trend.bias is not None:
            nn.init.zeros_(self.v_proj_trend.bias)

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
    ) -> torch.Tensor:
        q = self.q_proj(x)
        q = self._split_heads(q, self.n_heads)

        k_static = self.k_proj_static(x)
        v_static = self.v_proj_static(x)
        k_static = self._split_heads(k_static, self.n_static_heads)
        v_static = self._split_heads(v_static, self.n_static_heads)

        x_conv = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        x_conv = self.causal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        k_trend = self.k_proj_trend(x_conv)
        v_trend = self.v_proj_trend(x_conv)
        k_trend = self._split_heads(k_trend, self.n_trend_heads)
        v_trend = self._split_heads(v_trend, self.n_trend_heads)

        k = torch.cat([k_static, k_trend], dim=1)
        v = torch.cat([v_static, v_trend], dim=1)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = attn @ v
        out = self._merge_heads(out)
        return self.out(out)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        raise NotImplementedError("Streaming Conv1D cache for inference is pending")


class HeterogeneousCachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        if cfg.n_heads % 2 != 0:
            raise ValueError("hetero attention requires an even number of heads")
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_static_heads = cfg.n_heads // 2
        self.n_trend_heads = cfg.n_heads - self.n_static_heads
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

    def _project_kv(self, x: torch.Tensor, x_conv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_static = self._split_heads(self.k_proj_static(x), self.n_static_heads)
        v_static = self._split_heads(self.v_proj_static(x), self.n_static_heads)
        k_trend = self._split_heads(self.k_proj_trend(x_conv), self.n_trend_heads)
        v_trend = self._split_heads(self.v_proj_trend(x_conv), self.n_trend_heads)
        return torch.cat([k_static, k_trend], dim=1), torch.cat([v_static, v_trend], dim=1)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
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
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x), self.n_heads)
        x_conv = self.trend_act(self.causal_conv(x))
        k, v = self._project_kv(x, x_conv)
        return self._attend(q, k, v, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        q = self._split_heads(self.q_proj(x), self.n_heads)
        x_conv, conv_buffer = self.causal_conv.forward_step(x, cache.get("conv_buffer"))
        x_conv = self.trend_act(x_conv)
        k, v = self._project_kv(x, x_conv)
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
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), causal_mask, padding_mask))
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        self_out, cache = self.self_attn.forward_step(self.norm1(x), cache)
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
        self.token_emb = nn.Embedding(cfg.action_vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_action_len, cfg.d_model)
        self.blocks = nn.ModuleList([ActionDecoderBlock(cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.action_vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = decoder_input_ids.size(1)
        positions = torch.arange(seq_len, device=decoder_input_ids.device)
        x = self.token_emb(decoder_input_ids) + self.pos_emb(positions)[None, :, :]
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
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        if input_id.dim() == 1:
            input_id = input_id[:, None]
        if cache is None:
            cache = self.init_cache(context)
        if isinstance(step_idx, int):
            pos = torch.full((1,), step_idx, device=input_id.device, dtype=torch.long)
        else:
            pos = step_idx.to(device=input_id.device, dtype=torch.long).reshape(1)
        x = self.token_emb(input_id) + self.pos_emb(pos)[None, :, :]
        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, layer_cache = block.forward_step(x, layer_cache, context_mask)
            new_cache.append(layer_cache)
        hidden = self.norm(x)
        return {"hidden": hidden, "logits": self.lm_head(hidden)}, new_cache


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
        # 双阶段: 回归预测绝对起点位置
        self.start_head = nn.Linear(decoder_cfg.d_model, 2)

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
        # 从编码器输出中预测起点位置
        start_pred = self.start_head(text["context"][:, 0, :])  # [B, 2]
        dec_out = self.decoder(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_input_ids=decoder_input_ids,
            target_mask=target_mask,
        )
        return {"start_pred": start_pred, **dec_out}

    def predict_start(self, prompts: list[str]) -> torch.Tensor:
        text = self.encode_text(prompts)
        return self.start_head(text["context"][:, 0, :])  # [B, 2]

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_id: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        return self.decoder.decode_step(context, context_mask, input_id, step_idx, cache)


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
