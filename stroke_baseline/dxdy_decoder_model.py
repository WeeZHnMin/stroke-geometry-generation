from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .action_model import CausalConv1d, apply_2d_rotary_pos_emb
from .action_tokenizer import CompactDxDyTokenizer, DxDyPairTokenizer, DxDyPairTokenizerConfig


@dataclass
class DxDyDecoderConfig:
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

    def to_dict(self) -> dict:
        return asdict(self)


class _DecoderBlock(nn.Module):
    def __init__(self, cfg: DxDyDecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert (cfg.d_model // cfg.n_heads) % 4 == 0, \
            "head_dim must be divisible by 4 for 2D RoPE"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
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
        key_padding_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
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
        attn = self.dropout(torch.softmax(scores, dim=-1))
        sa_out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        x = x + self.dropout(self.out_proj(sa_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def init_cache(self) -> dict:
        return {"k": None, "v": None, "coords": None}

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, _, _ = x.shape
        normed = self.norm1(x)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        k_raw = k if cache["k"] is None else torch.cat([cache["k"], k], dim=2)
        v_all = v if cache["v"] is None else torch.cat([cache["v"], v], dim=2)

        if coords is not None:
            coords_all = coords if cache["coords"] is None \
                else torch.cat([cache["coords"], coords], dim=1)
            q, k_all = apply_2d_rotary_pos_emb(q, k_raw, coords, coords_all)
            cache["coords"] = coords_all
        else:
            k_all = k_raw

        cache["k"] = k_raw
        cache["v"] = v_all

        scores = (q @ k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        sa_out = (attn @ v_all).transpose(1, 2).contiguous().view(B, 1, self.d_model)
        x = x + self.dropout(self.out_proj(sa_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, cache


class DxDyDecoder(nn.Module):
    """
    Decoder-only LLM for (dx, dy) pair token sequences.

    Input flow:
        token_ids → Embedding → CausalConv(kernel=12) → X → QKV + 2D RoPE → Attention

    coords [B, T, 2]: cumulative (x, y) position for each token, used for 2D RoPE.
    BOS position is (0, 0); each subsequent position accumulates the decoded (dx, dy).
    """

    def __init__(self, cfg: DxDyDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.input_conv = CausalConv1d(cfg.d_model, cfg.d_model, kernel_size=cfg.conv_kernel_size)
        self.blocks = nn.ModuleList([_DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        coords:    [B, T, 2]  cumulative (x, y) per position; None disables 2D RoPE
        returns logits: [B, T, vocab_size]
        """
        B, T = input_ids.shape
        x = self.input_conv(self.token_emb(input_ids))

        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = input_ids == self.cfg.pad_token_id

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask, coords=coords)

        return self.lm_head(self.norm(x))

    def init_cache(self) -> dict:
        return {"conv_buffer": None, "blocks": [block.init_cache() for block in self.blocks]}

    def decode_step(
        self,
        input_id: torch.Tensor,
        cache: dict | None = None,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        input_id: [B] or [B, 1]
        coords:   [B, 1, 2]  position of this token
        returns logits: [B, 1, vocab_size], updated cache
        """
        if cache is None:
            cache = self.init_cache()
        if input_id.dim() == 1:
            input_id = input_id[:, None]
        x = self.token_emb(input_id)
        x, cache["conv_buffer"] = self.input_conv.forward_step(x, cache["conv_buffer"])
        for i, block in enumerate(self.blocks):
            x, cache["blocks"][i] = block.forward_step(x, cache["blocks"][i], coords=coords)
        return self.lm_head(self.norm(x)), cache

    @torch.no_grad()
    def generate(
        self,
        tokenizer: "DxDyPairTokenizer | CompactDxDyTokenizer",
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        bos = torch.tensor([[self.cfg.bos_token_id]], device=device)
        generated = [self.cfg.bos_token_id]
        cache = None
        cur = bos
        cx, cy = 0.0, 0.0
        # BOS sits at (0, 0)
        coords = torch.zeros(1, 1, 2, device=device)

        for _ in range(max_new_tokens):
            logits, cache = self.decode_step(cur, cache, coords=coords)
            logits = logits[:, -1, :] / temperature
            logits[:, self.cfg.bos_token_id] = torch.finfo(logits.dtype).min
            logits[:, self.cfg.pad_token_id] = torch.finfo(logits.dtype).min
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            tid = int(next_id[0, 0])
            generated.append(tid)
            cur = next_id

            # accumulate spatial position for next step's RoPE coords
            if 0 <= tid < tokenizer.action_vocab_size:
                dx, dy = tokenizer.decode_step(tid)
                cx += dx
                cy += dy
            coords = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=device)

        return torch.tensor(generated, device=device)


def build_dxdy_decoder(
    tokenizer: "DxDyPairTokenizer | CompactDxDyTokenizer | None" = None,
    dx_bins: int = 100,
    dy_bins: int = 100,
    min_val: float = -1.0,
    max_val: float = 1.0,
    **decoder_kwargs,
) -> tuple["DxDyDecoder", "DxDyPairTokenizer | CompactDxDyTokenizer"]:
    if tokenizer is None:
        tokenizer = DxDyPairTokenizer(DxDyPairTokenizerConfig(
            dx_bins=dx_bins, dy_bins=dy_bins, min_val=min_val, max_val=max_val
        ))
    cfg = DxDyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        **decoder_kwargs,
    )
    return DxDyDecoder(cfg), tokenizer
