"""Per-frame MLP projection with the paper's BatchNorm running statistics."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=None, norm_fn=nn.BatchNorm1d, act_fn=nn.GELU):
        super().__init__()
        norm = nn.Identity() if norm_fn is None else norm_fn(hidden_dim)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), norm, act_fn(),
                                 nn.Linear(hidden_dim, output_dim or input_dim))

    def forward(self, x):
        return self.net(x)
