from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

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


class CachedTokenSelfAttention(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

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
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        k_all = k if past_k is None else torch.cat([past_k, k], dim=2)
        v_all = v if past_v is None else torch.cat([past_v, v], dim=2)
        scores = (q @ k_all.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v_all)), k_all, v_all


class ActionDecoderBlock(nn.Module):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__()
        self.self_attn = CachedTokenSelfAttention(cfg)
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
        return {"self_k": None, "self_v": None, "cross_k": cross_k, "cross_v": cross_v}

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        self_out, self_k, self_v = self.self_attn.forward_step(
            self.norm1(x),
            past_k=cache.get("self_k"),
            past_v=cache.get("self_v"),
        )
        cache["self_k"] = self_k
        cache["self_v"] = self_v
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
