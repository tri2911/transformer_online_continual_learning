from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from continuous_learning.config import ModelConfig


def strict_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=-1,
    )


def shift_labels_for_kv(labels: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    shifted = torch.full_like(labels, pad_value)
    shifted[:, 1:] = labels[:, :-1]
    return shifted


@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor | None = None,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        context_window: int | None = None,
        strict_causal: bool = True,
    ) -> tuple[torch.Tensor, KVCache]:
        if kv_input is None:
            kv_input = q_input

        bsz, chunk_len, _ = q_input.shape
        q = self.q_proj(q_input).view(bsz, chunk_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, D]

        k_new = self.k_proj(kv_input)  # [B, T, D]
        v_new = self.v_proj(kv_input)  # [B, T, D]

        cache_len = 0 if cache is None else int(cache.key.shape[1])
        if cache is None:
            key_all = k_new
            value_all = v_new
        else:
            key_all = torch.cat([cache.key, k_new], dim=1)
            value_all = torch.cat([cache.value, v_new], dim=1)

        key_positions = torch.arange(key_all.shape[1], device=q_input.device)
        query_positions = torch.arange(
            cache_len,
            cache_len + chunk_len,
            device=q_input.device,
        )

        force_no_context = False
        if context_window is not None and context_window <= 0:
            # Disable all attention context while keeping tensor ranks valid.
            key = k_new
            value = v_new
            key_positions = torch.arange(cache_len, cache_len + chunk_len, device=q_input.device)
            force_no_context = True
        elif context_window is not None and key_all.shape[1] > context_window:
            start = key_all.shape[1] - context_window
            key = key_all[:, start:, :]
            value = value_all[:, start:, :]
            key_positions = key_positions[start:]
        else:
            key = key_all
            value = value_all

        if force_no_context:
            full_mask = torch.zeros((chunk_len, key.shape[1]), dtype=torch.bool, device=q_input.device)
        elif attn_mask is not None and cache_len == 0 and key.shape[1] == chunk_len:
            full_mask = attn_mask.to(q_input.device)
        else:
            if strict_causal:
                full_mask = key_positions.unsqueeze(0) < query_positions.unsqueeze(1)
            else:
                full_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        scores = torch.einsum("bhtd,bsd->bhts", q, key) * self.scale
        scores = scores.masked_fill(~full_mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out = torch.einsum("bhts,bsd->bhtd", attn, value)
        out = out.transpose(1, 2).contiguous().view(bsz, chunk_len, -1)
        out = self.o_proj(out)

        new_cache = KVCache(key=key.detach(), value=value.detach())
        return out, new_cache


class ParallelTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = MultiQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        hidden = int(config.d_model * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.label_to_model = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        label_context: torch.Tensor,
        cache: KVCache | None,
        attn_mask: torch.Tensor,
        context_window: int,
    ) -> tuple[torch.Tensor, KVCache]:
        normed = self.norm(x)
        kv_input = normed + self.label_to_model(label_context)
        attn_out, new_cache = self.attn(
            q_input=normed,
            kv_input=kv_input,
            cache=cache,
            attn_mask=attn_mask,
            context_window=context_window,
        )
        mlp_out = self.mlp(normed)
        x = x + self.dropout(attn_out) + self.dropout(mlp_out)
        return x, new_cache


class PiTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        # reserve id=0 for "no label" padding in shifted labels
        self.label_embedding = nn.Embedding(config.num_classes + 1, config.d_model)
        self.blocks = nn.ModuleList(
            [ParallelTransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        caches: list[KVCache | None] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        if caches is None:
            caches = [None] * self.config.n_blocks

        label_ids = labels + 1
        shifted_ids = shift_labels_for_kv(label_ids, pad_value=0)
        label_context = self.label_embedding(shifted_ids)

        x = self.input_proj(features)
        attn_mask = strict_causal_mask(features.size(1), device=features.device)

        next_caches: list[KVCache] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, next_cache = block(
                x=x,
                label_context=label_context,
                cache=cache,
                attn_mask=attn_mask,
                context_window=self.config.context_window,
            )
            next_caches.append(next_cache)

        logits = self.classifier(self.norm_out(x))
        return logits, next_caches
