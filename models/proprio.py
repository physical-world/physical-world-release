"""Per-frame action and proprio embeddings for the bound PushT Vanilla model."""

from torch import nn


class ProprioceptiveEmbedding(nn.Module):
    """Map (batch, time, input channels) to (batch, time, embedding channels).

    The paper checkpoint uses kernel/stride 1 for both embeddings: no temporal
    mixing, activation, or position encoding. Keep patch_embed keys unchanged
    so the exported action (10 -> 10) and proprio (4 -> 74) weights load strictly.
    Inputs are already normalized; packed actions retain all five control steps.
    """

    def __init__(self, num_frames=16, tubelet_size=1, in_chans=8, emb_dim=384, use_3d_pos=False):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, values):
        channels_first = values.permute(0, 2, 1)
        encoded = self.patch_embed(channels_first)
        return encoded.permute(0, 2, 1)
