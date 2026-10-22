"""Inference-only PPO with strict loading of the bound state-dict checkpoint."""

import numpy as np
import torch
from torch import nn


class Moments(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float64))


class Network(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, outputs))


class PushTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Network(2)
        self.actor.logstd = nn.Parameter(torch.full((2,), -0.5))
        self.critic = Network(1)
        self.obs_normalizer = Moments((14,))
        self.return_normalizer = Moments(())

    def forward(self, observation):
        norm = self.obs_normalizer
        normalized = torch.clamp((observation - norm.mean.to(observation.dtype)) /
                                 (norm.var.sqrt() + 1e-8).to(observation.dtype), -10, 10)
        return self.actor.net(normalized)

    @torch.no_grad()
    def action(self, env):
        observation = torch.as_tensor(policy_observation(env), device=self.actor.logstd.device)
        return self(observation).clamp(-1, 1).cpu().numpy()


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy = PushTPolicy()
    policy.load_state_dict(payload['model'], strict=True)
    return policy.eval().to(device)


def policy_observation(env):
    agent = np.asarray(env.agent.position)
    block = np.asarray(env.block.position)
    angle = env.block.angle % (2 * np.pi)
    error = env.goal_pose[2] - angle
    return np.r_[np.r_[agent, block] / 256 - 1, np.sin(angle), np.cos(angle),
                 (env.goal_pose[:2] - block) / 256, np.sin(error), np.cos(error),
                 np.asarray(env.agent.velocity) / 512,
                 np.asarray(env.block.velocity) / 512].astype(np.float32)
