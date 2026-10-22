"""FetchSlide H1 XY proposals, constant hit conditions, and terminal DINO cost."""

import numpy as np
import torch
from planning.candidates import rollout_candidates
from planning.cem import cem_iterations


def fetch_execution(proposals, normalization, horizon):
    mean = normalization['action_mean'].to(proposals)
    std = normalization['action_std'].to(proposals)
    hit = proposals[..., 0, :] * std[:2] + mean[:2]
    hit = torch.maximum(torch.minimum(hit, hit.new_tensor([.85, .8])), hit.new_tensor([.35, -.8]))
    raw = proposals.new_zeros(*hit.shape[:-1], 8)
    raw[..., :2] = hit
    raw[..., 4:6] = hit
    condition = ((raw - mean) / std).unsqueeze(-2)
    return condition.expand(*condition.shape[:-2], horizon, 8)


def fetch_raw_hit(execution, normalization):
    return execution[..., 0, :2] * normalization['action_std'][:2].to(execution) + normalization['action_mean'][:2].to(execution)


@torch.inference_mode()
def fetch_candidate_cost(model, current, goal, proposals, normalization, horizon, chunk_size=None):
    execution = fetch_execution(proposals, normalization, horizon)
    return torch.cat([rollout_candidates(model, current, goal, chunk, 0)[1]['total']
                      for chunk in execution.split(chunk_size or len(execution))])


def fetch_search(model, current, goal, normalization, horizon, settings, generator, chunk_size=None):
    """Yield CEM updates so equal-horizon targets can share the bound RNG order."""
    def score(proposals):
        return fetch_candidate_cost(model, current, goal, proposals, normalization, horizon, chunk_size)
    return cem_iterations(score, horizon=1, action_dim=2, device=next(model.parameters()).device,
                          generator=generator, **settings)


def fetch_success(state, goal_xy):
    error = float(np.linalg.norm(np.asarray(state)[3:5] - np.asarray(goal_xy)))
    return dict(endpoint_error_m=error, success=error <= .03)
