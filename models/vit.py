"""Checkpoint-shaped attention with frame-wise causal visibility."""

import torch
from torch import nn


def generate_mask_matrix(num_patches, num_frames):
    """Allow every patch in the query frame and all preceding frames."""
    frame_ids = torch.arange(num_frames).repeat_interleave(num_patches)
    return (frame_ids[:, None] >= frame_ids[None, :])[None, None].float()


class Attention(nn.Module):
    """Pre-normalized multi-head attention with explicit probability matrices.

    Earlier Vanilla objects retain an ordinary float mask and explicit softmax.
    Bound Long History objects use SDPA and a nonpersistent boolean mask buffer.
    Full-object restoration bypasses the constructor; both layouts are preserved.
    Same-frame patches see each other regardless of their spatial order.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., *, num_patches, num_frames, use_sdpa=False):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        if not use_sdpa:
            self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if heads != 1 or dim_head != dim else nn.Identity()
        mask = generate_mask_matrix(num_patches, num_frames)
        if use_sdpa:
            self.register_buffer("bias", mask.bool(), persistent=False)
        else:
            self.bias = mask

    def forward(self, x):
        batch, tokens, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = [part.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
                   for part in qkv]
        self.bias = self.bias.to(device=q.device)
        mask = self.bias[:, :, :tokens, :tokens]
        if hasattr(self, "attend"):
            scores = (q @ k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(mask == 0, float("-inf"))
            probabilities = self.dropout(self.attend(scores))
            values = probabilities @ v
        else:
            # Bound Long History objects retain their SDPA serialization layout.
            self.bias = self.bias.to(dtype=torch.bool)
            values = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=self.bias[:, :, :tokens, :tokens],
                dropout_p=self.dropout.p if self.training else 0.0, scale=self.scale,
            )
        values = values.transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(values)


class FeedForward(nn.Module):
    """Pre-normalized MLP preserving the checkpoint's sequential indices."""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """Attention and MLP residual blocks followed by a final LayerNorm."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., *,
                 num_patches, num_frames, use_sdpa=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout,
                          num_patches=num_patches, num_frames=num_frames, use_sdpa=use_sdpa),
                FeedForward(dim, mlp_dim, dropout),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    """Position-conditioned Transformer returning every token, without pooling."""

    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0., use_sdpa=False):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            num_patches=num_patches, num_frames=num_frames, use_sdpa=use_sdpa,
        )
        self.pool = pool

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        return self.transformer(x)
