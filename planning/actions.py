"""Relative proposal coordinates for the bound absolute-action PushT model."""

import torch

from datasets.pusht import denormalize, normalize


def pusht_proposal_statistics(manifest, action_scale):
    """Training action-minus-agent statistics, in scaled relative coordinates."""
    stats = manifest.metadata["action_stats_px"]["relative_delta"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32) / action_scale
    std = (torch.tensor(stats["std"], dtype=torch.float32) / action_scale).clamp_min(1e-6)
    return mean, std


class PushTActionConverter:
    """Packed (B,H,F*2) normalized actions <-> dense (B,H*F,2) pixel targets.

    Pass raw agent positions (B,2) as anchor, never normalized proprio. A single
    (1,2) anchor broadcasts over candidates for one target. The converter has
    no mutable planning state; callers explicitly select the current anchor.
    Relative proposals accumulate across token boundaries, without clipping.
    """

    def __init__(self, frameskip, action_scale, relative_mean, relative_std,
                 action_mean, action_std):
        self.frameskip = frameskip
        self.action_scale = action_scale
        self.relative_mean = torch.as_tensor(relative_mean, dtype=torch.float32)
        self.relative_std = torch.as_tensor(relative_std, dtype=torch.float32)
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)

    def proposal_to_absolute(self, proposal, anchor):
        """Denormalize each raw increment, then cumulatively add it to the anchor."""
        relative = denormalize(proposal.reshape(proposal.shape[0], -1, 2),
                               self.relative_mean.to(proposal), self.relative_std.to(proposal))
        return anchor.to(proposal)[:, None] + self.action_scale * relative.cumsum(dim=1)

    def absolute_to_proposal(self, absolute, anchor):
        """Invert cursor-target increments; later differences use previous targets."""
        previous = torch.cat([anchor.to(absolute).expand(absolute.shape[0], -1)[:, None], absolute[:, :-1]], dim=1)
        relative = (absolute - previous) / self.action_scale
        proposal = normalize(relative, self.relative_mean.to(absolute), self.relative_std.to(absolute))
        return proposal.reshape(absolute.shape[0], -1, self.frameskip * 2)

    def absolute_to_execution(self, absolute):
        """Pixel cursor targets -> checkpoint-normalized, chronologically packed actions."""
        execution = normalize(absolute, self.action_mean.to(absolute), self.action_std.to(absolute))
        return execution.reshape(absolute.shape[0], -1, self.frameskip * 2)

    def execution_to_absolute(self, execution):
        """Unpack and denormalize model actions to raw simulator cursor targets."""
        return denormalize(execution.reshape(execution.shape[0], -1, 2),
                           self.action_mean.to(execution), self.action_std.to(execution))

    def proposal_to_execution(self, proposal, anchor):
        return self.absolute_to_execution(self.proposal_to_absolute(proposal, anchor))

    def execution_to_proposal(self, execution, anchor):
        return self.absolute_to_proposal(self.execution_to_absolute(execution), anchor)
