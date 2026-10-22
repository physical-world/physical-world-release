"""TSWM embeddings retain their learned LayerNorm, unlike DINO-WM embeddings."""
from torch import nn
from models.proprio import ProprioceptiveEmbedding as FrameEmbedding


class ProprioceptiveEmbedding(FrameEmbedding):
    def __init__(self, num_frames=1, tubelet_size=1, in_chans=8, emb_dim=10,
                 use_3d_pos=False, use_layernorm=True):
        super().__init__(num_frames, tubelet_size, in_chans, emb_dim, use_3d_pos)
        self.norm = nn.LayerNorm(emb_dim) if use_layernorm else nn.Identity()

    def forward(self, values):
        return self.norm(super().forward(values))
