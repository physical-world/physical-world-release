"""Checkpoint-compatible patch decoder, residual blocks and EMA codebook."""

import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):
    """EMA codebook retained by the checkpoint, bypassed by paper Vanilla."""

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.T)

    def forward(self, input):
        flat = input.reshape(-1, self.dim)
        distances = flat.square().sum(1, keepdim=True) - 2 * flat @ self.embed + self.embed.square().sum(0)
        ids = distances.argmin(1).view(input.shape[:-1])
        quantized = self.embed_code(ids)
        if self.training:
            assignments = F.one_hot(ids.flatten(), self.n_embed).to(flat.dtype)
            counts = assignments.sum(0)
            sums = flat.T @ assignments
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
                torch.distributed.all_reduce(sums)
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                sizes = (self.cluster_size + self.eps) / (total + self.n_embed * self.eps) * total
                self.embed.copy_(self.embed_avg / sizes.unsqueeze(0))
        difference = (quantized.detach() - input).square().mean()
        return input + (quantized - input).detach(), difference, ids


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        return self.conv(input) + input


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    """Decode row-major visual patch tokens with the original object layout."""

    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 emb_dim=64, n_embed=512, decay=0.99, quantize=True):
        super().__init__()
        self.quantize = quantize
        # The original constructor does not pass its decay argument to Quantize.
        self.quantize_b = Quantize(emb_dim, n_embed)
        self.upsample_b = Decoder(emb_dim, emb_dim, channel, n_res_block, n_res_channel, 4)
        self.dec = Decoder(emb_dim, in_channel, channel, n_res_block, n_res_channel, 4)
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        batch, time, patches, width = input.shape
        side = int(patches ** 0.5)
        spatial = input.reshape(batch * time, side, side, width)
        if self.quantize:
            spatial, difference, _ = self.quantize_b(spatial)
        else:
            difference = torch.zeros(1, device=input.device)
        return self.decode(spatial.permute(0, 3, 1, 2)), difference.unsqueeze(0)

    def decode(self, quant_b):
        return self.dec(self.upsample_b(quant_b))

    def decode_code(self, code_b):
        return self.decode(self.quantize_b.embed_code(code_b).permute(0, 3, 1, 2))


class ProjectorDecoder(nn.Module):
    """TSWM's learned inverse channel projection, preserving the spatial grid."""

    def __init__(self, conv_layers):
        super().__init__()
        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(c.out_dim, c.in_dim, c.kernel_size, stride=c.stride, padding=c.padding)
            for c in reversed(conv_layers)
        ])
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        for index, conv in enumerate(self.deconv_layers):
            features = conv(features)
            if index < len(self.deconv_layers) - 1:
                features = self.activation(features)
        return features
