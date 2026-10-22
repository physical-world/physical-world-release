"""BeerPong H1 throw proposals and constant 2D model conditions."""
import torch
from planning.candidates import rollout_candidates
from planning.cem import cem_iterations


def beerpong_execution(proposals, normalization, horizon=20):
    mean, std = normalization['action_mean'].to(proposals), normalization['action_std'].to(proposals)
    raw = proposals[..., 0, :] * std + mean
    raw = torch.maximum(torch.minimum(raw, raw.new_tensor([3., 8.])), raw.new_tensor([2.5, 6.]))
    normalized = ((raw - mean) / std).unsqueeze(-2)
    return normalized.expand(*normalized.shape[:-2], horizon, 2)


def beerpong_raw_action(execution, normalization):
    return execution[..., 0, :] * normalization['action_std'].to(execution) + normalization['action_mean'].to(execution)


@torch.inference_mode()
def beerpong_candidate_cost(model, current, goal, proposals, normalization, horizon=20, chunk_size=8):
    execution = beerpong_execution(proposals, normalization, horizon)
    return torch.cat([rollout_candidates(model, current, goal, chunk, 0)[1]['total']
                      for chunk in execution.split(chunk_size or len(execution))])


def beerpong_search(model, current, goal, normalization, settings, generator, chunk_size=8):
    def score(proposals):
        return beerpong_candidate_cost(model, current, goal, proposals, normalization, chunk_size=chunk_size)
    return cem_iterations(score, horizon=1, action_dim=2, device=next(model.parameters()).device,
                          generator=generator, **settings)
