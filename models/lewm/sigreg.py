"""Sketch isotropic Gaussian regularization on time-by-batch visual features."""
import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, projected):
        directions = torch.randn(projected.shape[-1], self.num_proj, device=projected.device)
        directions = directions / directions.norm(dim=0)
        phase = (projected @ directions).unsqueeze(-1) * self.t
        error = (phase.cos().mean(-3) - self.phi).square() + phase.sin().mean(-3).square()
        return ((error @ self.weights) * projected.shape[-2]).mean()
