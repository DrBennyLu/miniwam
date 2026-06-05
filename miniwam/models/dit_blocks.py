"""DiT-style transformer blocks with optional attention mask."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import TimestepEmbedder


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MiniWAMBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_token_types: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.type_embed = nn.Embedding(num_token_types, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.t_embed = TimestepEmbedder(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor | None,
        token_types: torch.Tensor,
        attn_mask_bool: torch.Tensor | None,
    ) -> torch.Tensor:
        type_emb = self._type_embedding(token_types)
        if t is not None:
            temb = self.t_embed(t)
            tokens = tokens + temb.unsqueeze(1) + type_emb
        else:
            tokens = tokens + type_emb

        attn_mask = None
        if attn_mask_bool is not None:
            n = attn_mask_bool.shape[-1]
            attn_mask = torch.zeros(
                (n, n), device=tokens.device, dtype=tokens.dtype
            )
            attn_mask.masked_fill_(~attn_mask_bool, float("-inf"))

        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)
        return self.norm(tokens)

    def _type_embedding(self, token_types: torch.Tensor) -> torch.Tensor:
        return self.type_embed(token_types)


def patchify_latents(latents: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*ps*ps)."""
    b, c, h, w = latents.shape
    assert h % patch_size == 0 and w % patch_size == 0
    gh, gw = h // patch_size, w // patch_size
    x = latents.reshape(b, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, gh * gw, c * patch_size * patch_size)


def unpatchify_latents(tokens: torch.Tensor, c: int, h: int, w: int, patch_size: int):
    b, n, _ = tokens.shape
    gh, gw = h // patch_size, w // patch_size
    x = tokens.reshape(b, gh, gw, c, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, c, h, w)
