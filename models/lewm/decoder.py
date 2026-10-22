"""Paper CLS decoder: patch queries and cross-attention, without spatial upsampling."""
from pathlib import Path
import torch
from torch import nn
from models.lewm.decoder_blocks import CrossAttentionBlock


class CLSImageDecoder(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=512, depth=3, num_heads=8, mlp_ratio=4,
                 image_size=224, patch_size=16):
        super().__init__()
        self.input_dim = input_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_queries = self.grid_size ** 2
        self.memory_projection = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(1, self.num_queries, hidden_dim))
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.patch_projection = nn.Linear(hidden_dim, 3 * patch_size ** 2)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, cls):
        leading = cls.shape[:-1]
        memory = self.memory_projection(cls.reshape(-1, self.input_dim)).unsqueeze(1)
        queries = self.queries.expand(len(memory), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        patches = self.patch_projection(self.output_norm(queries))
        images = patches.reshape(-1, self.grid_size, self.grid_size, self.patch_size, self.patch_size, 3)
        images = images.permute(0, 5, 1, 3, 2, 4).reshape(*leading, 3, self.image_size, self.image_size)
        return images


def load_decoder(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    decoder = CLSImageDecoder(**payload['decoder_config'])
    decoder.load_state_dict(payload['decoder_state'], strict=True)
    return decoder.to(device), payload


def save_decoder(path, decoder, payload):
    path = Path(path).resolve()
    from runtime.checkpoints import ROOT
    path.relative_to((ROOT / 'logs').resolve())
    with path.open('xb') as stream:
        torch.save(dict(payload, decoder_state=decoder.state_dict()), stream)
