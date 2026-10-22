"""PushT task metrics and single-target repeated-horizon budget results."""

import numpy as np


def pusht_state_metrics(goal, state):
    """Evaluate raw poses (and diagnostic velocities), with leading-axis broadcast.

    Success uses strict 20px and pi/9 thresholds; Joint combines both xy errors.
    Velocity affects only state_dist, whose raw angle is intentionally unwrapped.
    """
    goal, state = np.asarray(goal), np.asarray(state)
    agent = np.linalg.norm(goal[..., :2] - state[..., :2], axis=-1)
    block = np.linalg.norm(goal[..., 2:4] - state[..., 2:4], axis=-1)
    joint = np.linalg.norm(goal[..., :4] - state[..., :4], axis=-1)
    angle = np.abs((goal[..., 4] - state[..., 4] + np.pi) % (2 * np.pi) - np.pi)
    block_success = (block < 20) & (angle < np.pi / 9)
    joint_success = (joint < 20) & (angle < np.pi / 9)
    return dict(success=joint_success, block_success=block_success, joint_success=joint_success,
                state_dist=np.linalg.norm(goal - state, axis=-1), agent_position_error=agent,
                block_position_error=block, joint_position_error=joint, wrapped_block_angle_error=angle)


def pusht_budget_results(states, goal, budgets=(0, 25, 50, 75)):
    """Summarize one real trajectory, including initial state, at bound MPC budgets.

    states ends at a budget boundary, possibly after early Joint success.
    Only executed boundaries have endpoint results. Cumulative recovery carries
    observed boundary successes forward; ever_joint_success scans all real steps.
    Never pass display-held frames here.
    """
    metrics = pusht_state_metrics(goal, states)
    endpoint, cumulative = {}, {}
    recovered = dict(block_success=False, joint_success=False)
    for budget in budgets:
        if budget < len(states):
            endpoint[budget] = {key: value[budget].item() for key, value in metrics.items()}
            for key in recovered:
                recovered[key] |= endpoint[budget][key]
        cumulative[budget] = recovered.copy()
    return dict(executed_controls=len(states) - 1, endpoint=endpoint, cumulative=cumulative,
                final={key: value[-1].item() for key, value in metrics.items()},
                ever_joint_success=bool(metrics["joint_success"].any()))
