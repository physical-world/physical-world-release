"""The bound LeWM AdaLN-zero computation, including its inner pre-normalization."""
import torch
from torch import nn
from torch.nn import functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Identity() if heads == 1 and dim_head == dim else nn.Sequential(
            nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        batch, time = x.shape[:2]
        q, k, v = [value.reshape(batch, time, self.heads, -1).transpose(1, 2)
                   for value in self.to_qkv(self.norm(x)).chunk(3, -1)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.to_out(y.transpose(1, 2).reshape(batch, time, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, -1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
