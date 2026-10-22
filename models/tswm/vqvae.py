"""TSWM decoder composition; shared residual convolutions retain their identities."""
from torch import nn
from models.vqvae import VQVAE as PatchDecoder, Quantize, Decoder, ProjectorDecoder


class VQVAE(PatchDecoder):
    def __init__(self, in_channel=3, channel=384, n_res_block=4, n_res_channel=128,
                 emb_dim=8, n_embed=2048, decay=0.99, quantize=False, projector_cfg=None):
        nn.Module.__init__(self)
        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)
        self.projector_cfg = projector_cfg
        self.proj_decoder = ProjectorDecoder(projector_cfg.conv_layers)
        self.decoder_emb_dim = projector_cfg.conv_layers[0].in_dim
        self.upsample_b = Decoder(self.decoder_emb_dim, self.decoder_emb_dim, channel, n_res_block, n_res_channel, 4)
        self.dec = Decoder(self.decoder_emb_dim, in_channel, channel, n_res_block, n_res_channel, 4)
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def decode(self, quant_b):
        return self.dec(self.upsample_b(self.proj_decoder(quant_b)))
