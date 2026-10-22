"""Frozen official DINOv2 ViT-S/14 and patch inspection computations."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DINOV2_REPO = "facebookresearch/dinov2:b8931f7bf91576930313be2c6d6af376033b35f0"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def dino_weights_path(weights_path=None):
    if weights_path is not None:
        return Path(weights_path).expanduser()
    return Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_URL.rsplit("/", 1)[-1]


class FrozenDino(nn.Module):
    """Load official pretrained weights from Torch Hub or an explicit local file.

    Use the ViT-S/14 LVD-142M artifact identified above. The caller places the
    module and images on the same device. Inputs are NCHW tensors already
    preprocessed by the dataset; outputs are row-major patch tokens (N, P, 384).
    No CLS token, image normalization, or legacy model classes are added here.
    """

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__()
        if weights_path is not None:
            state_dict = torch.load(dino_weights_path(weights_path), map_location="cpu", weights_only=True)
        self.base_model = torch.hub.load(
            DINOV2_REPO, "dinov2_vits14", pretrained=weights_path is None,
            trust_repo=True, skip_validation=True,
        )
        if weights_path is not None:
            self.base_model.load_state_dict(state_dict, strict=True)
        self.base_model.requires_grad_(False)
        self.patch_size = self.base_model.patch_size
        self.emb_dim = self.base_model.num_features
        self.eval()

    def train(self, mode: bool = True):
        """Keep this frozen encoder in eval even when its parent enters train mode."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.base_model.forward_features(images)["x_norm_patchtokens"]


def inspect_patch_tokens(tokens: torch.Tensor, reference_frame: int, reference_patch: int) -> dict:
    """Shared three-component PCA and cosine maps for manual inspection.

    Fit once across all frames/patches; scale each PCA component globally for RGB
    display. PCA colors have arbitrary signs and are not physical predictions.
    Cosines compare every token to one fixed token in the reference frame.
    """
    features = tokens.detach().float().cpu()
    flat = features.flatten(0, 1)
    centered = flat - flat.mean(dim=0)
    _, singular_values, axes = torch.linalg.svd(centered, full_matrices=False)
    scores = centered @ axes[:3].T
    low = scores.amin(dim=0)
    span = (scores.amax(dim=0) - low).clamp_min(1e-12)
    colors = ((scores - low) / span).reshape(*features.shape[:2], 3)
    unit = F.normalize(features, dim=-1)
    cosine = unit @ unit[reference_frame, reference_patch]
    variance = singular_values.square()
    return {
        "pca_rgb": colors,
        "explained_variance": variance[:3] / variance.sum().clamp_min(1e-12),
        "cosine": cosine,
    }


class ChannelProjector(nn.Module):
    """The bound TSWM channel projector; its stride-one grid remains 14 by 14."""

    def __init__(self, in_dim=384, out_dim=8, norm_type='layer', conv_layers=None):
        super().__init__()
        self.norm_type = norm_type
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(c.in_dim, c.out_dim, c.kernel_size, stride=c.stride, padding=c.padding)
            for c in conv_layers
        ])
        self.batch_norm_layers = nn.ModuleList()
        self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(c.out_dim) for c in conv_layers])
        self.activation = nn.ReLU()

    def __setstate__(self, state):
        super().__setstate__(state)
        # The paper restoration discards serialized, unused BatchNorm modules.
        self.batch_norm_layers = nn.ModuleList()

    def forward(self, features):
        for index, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norm_layers)):
            features = conv(features)
            features = norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if index < len(self.conv_layers) - 1:
                features = self.activation(features)
        return features.flatten(2).transpose(1, 2)


class DinoV2Encoder(nn.Module):
    """Frozen official backbone with the paper's trainable projector and aggregator."""

    def __init__(self, name='dinov2_vits14', feature_key='x_norm_patchtokens',
                 projector_config=None, agg_out_dim=128, agg_mlp_hidden_dim=512, *, base_model=None):
        super().__init__()
        self.name = name
        self.dinov2_repo = DINOV2_REPO
        self.base_model = base_model if base_model is not None else torch.hub.load(
            DINOV2_REPO, name, pretrained=False, trust_repo=True, skip_validation=True)
        self.base_model.requires_grad_(False)
        self.feature_key = feature_key
        self.projector_name = 'channel'
        self.projector = projector_config
        self.emb_dim = self.projector.conv_layers[-1].out_channels
        self.patch_size = self.base_model.patch_size
        self.latent_ndim = 2
        self.agg_type = 'mlp'
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim
        self._agg_mlp_in_dim = 196 * self.emb_dim
        self._agg_out_dim = agg_out_dim
        self._agg_mlp_hidden_dim = agg_mlp_hidden_dim
        self.agg_mlp = nn.Sequential(
            nn.Linear(self._agg_mlp_in_dim, agg_mlp_hidden_dim), nn.ReLU(),
            nn.Linear(agg_mlp_hidden_dim, agg_mlp_hidden_dim), nn.ReLU(),
            nn.Linear(agg_mlp_hidden_dim, agg_out_dim),
        )
        self.agg_post_norm = nn.LayerNorm(agg_out_dim)
        self.base_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def agg(self, patches):
        return self.agg_post_norm(self.agg_mlp(patches.contiguous().flatten(1)))

    def forward(self, images, return_agg=False):
        features = self.base_model.forward_features(images)[self.feature_key]
        batch, patches, width = features.shape
        side = int(patches ** 0.5)
        spatial = features.reshape(batch, side, side, width).permute(0, 3, 1, 2).contiguous()
        projected = self.projector(spatial)
        return self.agg(projected) if return_agg else projected
