"""Iteration-major CEM updates, visiting target slots in their supplied order."""

import torch


def cem_target_iterations(scores, *, horizon, action_dim, num_samples, topk,
                          var_scale, opt_steps, device, generator=None,
                          initial_means=None, active=None):
    """Draw for every target before skipping inactive slots, as in the reference."""
    means = [torch.zeros(horizon, action_dim, device=device) for _ in scores]
    if initial_means is not None:
        for mean, warm in zip(means, initial_means):
            if warm is not None:
                mean[:len(warm)] = warm
    stds = [var_scale * torch.ones_like(mean) for mean in means]
    for iteration in range(opt_steps):
        updates = []
        for target, score in enumerate(scores):
            mean, std = means[target], stds[target]
            noise = torch.randn(num_samples, horizon, action_dim, generator=generator)
            proposals = noise.to(device) * std + mean
            proposals[0] = mean
            update = dict(iteration=iteration + 1, mean=mean, std=std, proposals=proposals,
                          next_mean=mean, next_std=std)
            if active is None or active[target]:
                costs = score(proposals)
                elite_indices = torch.argsort(costs)[:topk]
                elites = proposals[elite_indices]
                means[target], stds[target] = elites.mean(0), elites.std(0, unbiased=False)
                update.update(costs=costs, elite_indices=elite_indices,
                              next_mean=means[target], next_std=stds[target])
            updates.append(update)
        yield updates


def cem_iterations(score, *, initial_mean=None, **settings):
    """Single-target inspection view of the canonical target loop."""
    for updates in cem_target_iterations([score], initial_means=[initial_mean], **settings):
        yield updates[0]


def cem_plan(score, *, initial_mean=None, **settings):
    """Return final elite means (targets,H,D); accept one score or an ordered list."""
    scores = score if isinstance(score, list) else [score]
    warms = initial_mean if isinstance(score, list) else [initial_mean]
    for updates in cem_target_iterations(scores, initial_means=warms, **settings):
        pass
    return torch.stack([update['next_mean'] for update in updates])
