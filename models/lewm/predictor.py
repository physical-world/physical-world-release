"""Causal LeWM predictor with frame positions and independent action conditions."""
import torch
from torch import nn
from models.lewm.conditioning import ConditionalBlock


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.0, condition_dim=None):
        super().__init__()
        condition_dim = input_dim if condition_dim is None else condition_dim
        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Identity() if condition_dim == hidden_dim else nn.Linear(condition_dim, hidden_dim)
        self.output_proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, c):
        x, c = self.input_proj(x), self.cond_proj(c)
        for layer in self.layers:
            x = layer(x, c)
        return self.output_proj(self.norm(x))


class ARPredictor(nn.Module):
    def __init__(self, *, num_frames, input_dim, hidden_dim, output_dim=None, depth=6,
                 heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, emb_dropout=0.0, condition_dim=192):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(input_dim, hidden_dim, output_dim or input_dim, depth,
                                       heads, dim_head, mlp_dim, dropout, condition_dim)

    def forward(self, x, c):
        return self.transformer(self.dropout(x + self.pos_embedding[:, :x.shape[1]]), c)
