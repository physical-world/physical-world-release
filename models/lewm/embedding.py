"""Per-frame nonlinear LeWM action embedding, separate from proprio."""
from torch import nn


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, 1)
        self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                                   nn.Linear(mlp_scale * emb_dim, emb_dim))

    def forward(self, x):
        smoothed = self.patch_embed(x.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(smoothed)
