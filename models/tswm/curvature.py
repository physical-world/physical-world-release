"""The bound aggregation cosine curvature objective."""
from torch.nn import functional as F


def cosine_curvature(trajectory, valid=None, eps=1e-6):
    """Ignore zero-length displacements and triples containing padded frames."""
    first = trajectory[:, 1:-1] - trajectory[:, :-2]
    second = trajectory[:, 2:] - trajectory[:, 1:-1]
    values = 1 - F.cosine_similarity(first, second, dim=-1, eps=eps)
    mask = (first.norm(dim=-1) > eps) & (second.norm(dim=-1) > eps)
    if valid is not None:
        mask = mask & valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    selected = values[mask]
    return selected.mean() if selected.numel() else (first.sum() + second.sum()) * 0


def aggregation_curvature(encoder, features, valid=None):
    batch, time, patches, width = features.shape
    trajectory = encoder.agg(features.reshape(batch * time, patches, width)).reshape(batch, time, -1)
    return cosine_curvature(trajectory, valid)
