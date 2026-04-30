from __future__ import annotations

import torch
import torch.nn as nn

from continuous_learning.config import ModelConfig
from continuous_learning.models.pi_transformer import KVCache, MultiQueryAttention, shift_labels_for_kv


class TwoTokenBlock(nn.Module):
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
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None,
        causal_mask: torch.Tensor,
        context_window_tokens: int,
    ) -> tuple[torch.Tensor, KVCache]:
        normed = self.norm(x)
        attn_out, new_cache = self.attn(
            q_input=normed,
            kv_input=normed,
            cache=cache,
            attn_mask=causal_mask,
            context_window=context_window_tokens,
            strict_causal=False,
        )
        x = x + self.dropout(attn_out) + self.dropout(self.mlp(normed))
        return x, new_cache


class TwoTokenTransformerBaseline(nn.Module):
    """
    2-token baseline with cache-aware temporal context.
    Each chunk interleaves tokens as [f1,y1,f2,y2,...,fS,yS].
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.context_window_tokens = max(0, 2 * config.context_window)
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        self.label_embedding = nn.Embedding(config.num_classes + 1, config.d_model)
        self.blocks = nn.ModuleList([TwoTokenBlock(config) for _ in range(config.n_blocks)])
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        caches: list[KVCache | None] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        bsz, seq_len, _ = features.shape
        if caches is None:
            caches = [None] * len(self.blocks)

        feat_tokens = self.input_proj(features)
        shifted_ids = shift_labels_for_kv(labels + 1, pad_value=0)
        label_tokens = self.label_embedding(shifted_ids)

        tokens = torch.stack([feat_tokens, label_tokens], dim=2).reshape(bsz, 2 * seq_len, self.config.d_model)
        causal_mask = torch.tril(
            torch.ones((2 * seq_len, 2 * seq_len), dtype=torch.bool, device=features.device),
            diagonal=0,
        )

        x = tokens
        next_caches: list[KVCache] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, new_cache = block(
                x=x,
                cache=cache,
                causal_mask=causal_mask,
                context_window_tokens=self.context_window_tokens,
            )
            next_caches.append(new_cache)

        feat_out = x[:, 0::2, :]
        logits = self.classifier(self.norm(feat_out))
        return logits, next_caches
