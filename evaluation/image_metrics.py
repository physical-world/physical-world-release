"""Bound PushT image metrics, operating directly on DINO-WM image values."""

from math import exp

import torch
from torch.nn import functional as F
from third_party.lpipsPyTorch import LPIPS


def ssim(predicted, target):
    """11x11 Gaussian, sigma=1.5, zero padding, reference unit-range constants."""
    gaussian = torch.tensor([exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)])
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(3, 1, 11, 11).contiguous().to(predicted)
    def smooth(x):
        return F.conv2d(x, window, padding=5, groups=3)
    a, b = smooth(predicted), smooth(target)
    va, vb = smooth(predicted.square()) - a.square(), smooth(target.square()) - b.square()
    covariance = smooth(predicted * target) - a * b
    return (((2 * a * b + .01 ** 2) * (2 * covariance + .03 ** 2)) /
            ((a.square() + b.square() + .01 ** 2) * (va + vb + .03 ** 2))).mean()


def image_metrics(predicted, target, lpips_model):
    """Pool the supplied frames; PSNR is computed after pooling MSE."""
    predicted, target = predicted.reshape(-1, *predicted.shape[-3:]), target.reshape(-1, *target.shape[-3:])
    error = (predicted - target).square().mean()
    return dict(l1=float((predicted - target).abs().mean()), l2=float(error),
                ssim=float(ssim(predicted, target)), mse=float(error),
                psnr=float(20 * torch.log10(1 / error.sqrt())), lpips=float(lpips_model(predicted, target)))
