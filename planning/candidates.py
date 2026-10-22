"""Batched Vanilla rollouts and the bound final-frame latent objective."""

import torch


def latent_goal_cost(predicted, goal, alpha):
    """Return per-candidate final-frame MSEs against one encoded goal frame."""
    visual = (predicted["visual"][:, -1:] - goal["visual"]).square().flatten(1).mean(1)
    if "proprio" not in predicted:
        return {"visual": visual, "total": visual}
    proprio = (predicted["proprio"][:, -1:] - goal["proprio"]).square().flatten(1).mean(1)
    return {"visual": visual, "proprio": proprio, "total": visual + alpha * proprio}


@torch.inference_mode()
def rollout_candidates(model, current, goal, execution_actions, alpha, history_actions=None):
    """Score candidates for one target with an eval-mode model on the input device.

    Observations have batch size one; actions are (candidates, horizon, 10).
    Actions are normalized model controls, already converted from proposals.
    Prepend the H-1 executed actions linking H real history observations.
    Return the real history plus all future predictions for inspection.
    """
    count = execution_actions.shape[0]
    initial = {key: value.expand(count, *value.shape[1:]) for key, value in current.items()}
    encoded_goal = model.encode_obs(goal)
    if history_actions is not None:
        execution_actions = torch.cat((history_actions.expand(count, -1, -1), execution_actions), dim=1)
    # ponytail: retain full trajectories; chunk candidates if batches exceed GPU memory.
    trajectory, _ = model.rollout(initial, execution_actions)
    return trajectory, latent_goal_cost(trajectory, encoded_goal, alpha)
