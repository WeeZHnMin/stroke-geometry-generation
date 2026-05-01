from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .dataset import NUM_INPUT_PEN_STATES, NUM_TARGET_PEN_STATES


DEFAULT_TEXT_ENCODER_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "chinese_roberta_L-2_H-128"
)


@dataclass
class StrokeDecoderConfig:
    d_model: int = 128
    n_heads: int = 4
    num_decoder_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    max_stroke_len: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


class FrozenChineseTextEncoder(nn.Module):
    """Local Chinese RoBERTa tiny encoder used as the text condition source."""

    def __init__(self, model_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.encoder = AutoModel.from_pretrained(self.model_dir)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def tokenize(
        self,
        prompts: list[str],
        max_text_len: int = 64,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded

    def forward(
        self,
        prompts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_text_len: int = 64,
    ) -> dict[str, torch.Tensor]:
        if prompts is not None:
            encoded = self.tokenize(prompts, max_text_len=max_text_len, device=self.encoder.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        if input_ids is None or attention_mask is None:
            raise ValueError("Provide either prompts or input_ids + attention_mask.")

        with torch.no_grad():
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "context": output.last_hidden_state,
            "context_mask": attention_mask.bool(),
        }


class StrokeFeatureEmbedding(nn.Module):
    def __init__(self, cfg: StrokeDecoderConfig):
        super().__init__()
        self.delta_proj = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.pen_emb = nn.Embedding(NUM_INPUT_PEN_STATES, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_stroke_len, cfg.d_model)

    def forward(self, decoder_dxdy: torch.Tensor, decoder_pen: torch.Tensor) -> torch.Tensor:
        seq_len = decoder_dxdy.size(1)
        positions = torch.arange(seq_len, device=decoder_dxdy.device)
        return (
            self.delta_proj(decoder_dxdy)
            + self.pen_emb(decoder_pen)
            + self.pos_emb(positions)[None, :, :]
        )

    def forward_step(
        self,
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        step_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        if decoder_dxdy.size(1) != 1 or decoder_pen.size(1) != 1:
            raise ValueError("forward_step expects one stroke token: [B, 1, ...].")
        if isinstance(step_idx, int):
            positions = torch.full((1,), step_idx, device=decoder_dxdy.device, dtype=torch.long)
        else:
            positions = step_idx.to(device=decoder_dxdy.device, dtype=torch.long).reshape(1)
        return (
            self.delta_proj(decoder_dxdy)
            + self.pen_emb(decoder_pen)
            + self.pos_emb(positions)[None, :, :]
        )


class CachedSelfAttention(nn.Module):
    def __init__(self, cfg: StrokeDecoderConfig):
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
        if x.size(1) != 1:
            raise ValueError("forward_step expects one token: [B, 1, D].")

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2)
        else:
            k_all = k
        if past_v is not None:
            v_all = torch.cat([past_v, v], dim=2)
        else:
            v_all = v

        scores = (q @ k_all.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = self.out(self._merge_heads(attn @ v_all))
        return out, k_all, v_all


class CrossAttention(nn.Module):
    def __init__(self, cfg: StrokeDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, cfg.d_model * 2)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def project_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.kv(context).chunk(2, dim=-1)
        return self._split_heads(k), self._split_heads(v)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        q = self._split_heads(self.q(x))
        k, v = self.project_kv(context)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        if context_mask is not None:
            scores = scores.masked_fill(context_mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ v))

    def forward_with_kv(
        self,
        x: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q(x))
        scores = (q @ cross_k.transpose(-2, -1)) / (self.head_dim**0.5)
        if context_mask is not None:
            scores = scores.masked_fill(context_mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(self._merge_heads(attn @ cross_v))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: StrokeDecoderConfig):
        super().__init__()
        self.self_attn = CachedSelfAttention(cfg)
        self.cross_attn = CrossAttention(cfg)
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
        stroke_padding_mask: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        self_out = self.self_attn(x_norm, causal_mask=causal_mask, key_padding_mask=stroke_padding_mask)
        x = residual + self.dropout(self_out)

        x = x + self.dropout(self.cross_attn(self.norm2(x), context, context_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

    def init_cross_cache(self, context: torch.Tensor) -> dict[str, torch.Tensor | None]:
        cross_k, cross_v = self.cross_attn.project_kv(context)
        return {
            "self_k": None,
            "self_v": None,
            "cross_k": cross_k,
            "cross_v": cross_v,
        }

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor | None],
        context_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        residual = x
        self_out, self_k, self_v = self.self_attn.forward_step(
            self.norm1(x),
            past_k=cache.get("self_k"),
            past_v=cache.get("self_v"),
        )
        cache["self_k"] = self_k
        cache["self_v"] = self_v
        x = residual + self.dropout(self_out)

        cross_k = cache.get("cross_k")
        cross_v = cache.get("cross_v")
        if cross_k is None or cross_v is None:
            raise ValueError("Cross-attention KV cache is missing.")
        x = x + self.dropout(self.cross_attn.forward_with_kv(self.norm2(x), cross_k, cross_v, context_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x, cache


class StrokeDecoder(nn.Module):
    """Autoregressive stroke decoder conditioned on token-level text features."""

    def __init__(self, cfg: StrokeDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.stroke_emb = StrokeFeatureEmbedding(cfg)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.delta_head = nn.Linear(cfg.d_model, 2)
        self.pen_head = nn.Linear(cfg.d_model, NUM_TARGET_PEN_STATES)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = decoder_dxdy.size(1)
        x = self.stroke_emb(decoder_dxdy, decoder_pen)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_dxdy.device, dtype=torch.bool),
            diagonal=1,
        )
        stroke_padding_mask = None if target_mask is None else ~target_mask.bool()

        for block in self.blocks:
            x = block(
                x=x,
                causal_mask=causal_mask,
                stroke_padding_mask=stroke_padding_mask,
                context=context,
                context_mask=context_mask,
            )

        hidden = self.norm(x)
        return {
            "hidden": hidden,
            "pred_dxdy": self.delta_head(hidden),
            "pred_pen_logits": self.pen_head(hidden),
        }

    def init_cache(self, context: torch.Tensor) -> list[dict[str, torch.Tensor | None]]:
        return [block.init_cross_cache(context) for block in self.blocks]

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        if cache is None:
            cache = self.init_cache(context)
        if len(cache) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} cache entries, got {len(cache)}.")

        x = self.stroke_emb.forward_step(decoder_dxdy, decoder_pen, step_idx)
        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, updated_cache = block.forward_step(x, layer_cache, context_mask)
            new_cache.append(updated_cache)

        hidden = self.norm(x)
        out = {
            "hidden": hidden,
            "pred_dxdy": self.delta_head(hidden),
            "pred_pen_logits": self.pen_head(hidden),
        }
        return out, new_cache


class TextConditionedStrokeModel(nn.Module):
    """Convenience wrapper combining the frozen Chinese encoder and stroke decoder."""

    def __init__(
        self,
        decoder_cfg: StrokeDecoderConfig | None = None,
        text_encoder_dir: str | Path = DEFAULT_TEXT_ENCODER_DIR,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.text_encoder = FrozenChineseTextEncoder(text_encoder_dir)
        self.decoder = StrokeDecoder(decoder_cfg or StrokeDecoderConfig())

        if self.text_encoder.hidden_size != self.decoder.cfg.d_model:
            self.context_proj = nn.Linear(self.text_encoder.hidden_size, self.decoder.cfg.d_model)
        else:
            self.context_proj = nn.Identity()

    def forward(
        self,
        prompts: list[str],
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.max_text_len)
        context = self.context_proj(text["context"])
        return self.decoder(
            context=context,
            context_mask=text["context_mask"],
            decoder_dxdy=decoder_dxdy,
            decoder_pen=decoder_pen,
            target_mask=target_mask,
        )

    def encode_text(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self.text_encoder(prompts=prompts, max_text_len=self.max_text_len)
        return {
            "context": self.context_proj(text["context"]),
            "context_mask": text["context_mask"],
        }

    def decode_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        decoder_dxdy: torch.Tensor,
        decoder_pen: torch.Tensor,
        step_idx: int | torch.Tensor,
        cache: list[dict[str, torch.Tensor | None]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor | None]]]:
        return self.decoder.decode_step(
            context=context,
            context_mask=context_mask,
            decoder_dxdy=decoder_dxdy,
            decoder_pen=decoder_pen,
            step_idx=step_idx,
            cache=cache,
        )


if __name__ == "__main__":
    batch_size = 2
    stroke_len = 16
    model = TextConditionedStrokeModel()
    decoder_dxdy = torch.zeros(batch_size, stroke_len, 2)
    decoder_pen = torch.zeros(batch_size, stroke_len, dtype=torch.long)
    target_mask = torch.ones(batch_size, stroke_len, dtype=torch.bool)

    out = model(
        prompts=["画一个圆", "在左边画一个矩形"],
        decoder_dxdy=decoder_dxdy,
        decoder_pen=decoder_pen,
        target_mask=target_mask,
    )
    print("pred_dxdy:", tuple(out["pred_dxdy"].shape))
    print("pred_pen_logits:", tuple(out["pred_pen_logits"].shape))

    text = model.encode_text(["画一个圆", "在左边画一个矩形"])
    cache = None
    last_dxdy = torch.zeros(batch_size, 1, 2)
    last_pen = torch.zeros(batch_size, 1, dtype=torch.long)
    for step_idx in range(3):
        step_out, cache = model.decode_step(
            context=text["context"],
            context_mask=text["context_mask"],
            decoder_dxdy=last_dxdy,
            decoder_pen=last_pen,
            step_idx=step_idx,
            cache=cache,
        )
        last_dxdy = step_out["pred_dxdy"]
        last_pen = step_out["pred_pen_logits"].argmax(dim=-1)
    print("cached_step_dxdy:", tuple(step_out["pred_dxdy"].shape))
    print("cached_step_pen_logits:", tuple(step_out["pred_pen_logits"].shape))
