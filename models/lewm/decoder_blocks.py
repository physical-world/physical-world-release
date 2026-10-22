"""Decoder cross-attention from learned spatial queries to CLS memory."""
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, queries, memory):
        attended, _ = self.attn(self.attn_norm(queries), memory, memory, need_weights=False)
        queries = queries + attended
        return queries + self.mlp(self.mlp_norm(queries))
