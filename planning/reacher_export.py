"""Reacher planning records and control-aligned four-panel inspection videos."""
import json
from pathlib import Path

import numpy as np
import torch

from datasets.reacher import reacher_observation
from env.reacher import ReacherEnv
from planning.reacher import reacher_target_rounds, reacher_success


def reacher_budget_results(states, goal):
    """Carry stopped boundary success to later budgets without fabricated states."""
    executed = len(states) - 1
    endpoints = {}
    for budget in (25, 50, 75):
        observed = min(budget, executed)
        state = states[observed]
        delta = np.asarray(state, dtype=float)[:2] - np.asarray(goal, dtype=float)[:2]
        endpoints[str(budget)] = dict(state=state.tolist(), joint1_error_rad=float(delta[0]),
                                     joint2_error_rad=float(delta[1]), max_abs_joint_error_rad=float(np.abs(delta).max()),
                                     success=reacher_success(goal, state), executed_raw_actions=observed,
                                     carried_after_stop=executed < budget)
    success = endpoints['75']['success']
    delta = np.asarray(states[-1], dtype=float) - np.asarray(goal, dtype=float)
    wrapped = np.abs(np.arctan2(np.sin(delta[:2]), np.cos(delta[:2])))
    metrics = dict(unwrapped_success=success, wrapped_success=bool(np.all(wrapped < np.deg2rad(10))),
                   joint1_unwrapped_error=float(abs(delta[0])), joint2_unwrapped_error=float(abs(delta[1])),
                   joint1_wrapped_error=float(wrapped[0]), joint2_wrapped_error=float(wrapped[1]),
                   qpos_l2=float(np.linalg.norm(delta[:2])), qvel_l2=float(np.linalg.norm(delta[2:])),
                   state_l2=float(np.linalg.norm(delta)))
    return dict(**metrics, success=success, cumulative_unwrapped_success_at_0_actions=False, executed_raw_actions=executed, executed_wm_tokens=executed // 5,
                first_success_raw_actions=executed if success else None, first_success_physics_substep=None,
                final_state=states[-1].tolist(), round_endpoints=endpoints,
                **{f'cumulative_unwrapped_success_at_{b}_actions': endpoints[str(b)]['success'] for b in (25, 50, 75)})


@torch.inference_mode()
def export_reacher_planning(model, collection, targets, normalization, output, metadata, cem_settings,
                            *, seed=42, inspection=False, num_visualizations=1, candidate_chunk_size=16, physics=None, contexts=None):
    output = Path(output).resolve()
    output.relative_to((Path(__file__).parents[1] / 'logs').resolve())
    output.mkdir(parents=True, exist_ok=False)
    records = []
    generator = torch.Generator().manual_seed(seed)
    planning_targets = []
    for target in targets:
        episode = collection.read_episode(target['source_episode'])
        start, end = target['current_row'], target['goal_row']
        goal_state = episode['state'][end]
        goal, _ = reacher_observation(episode['rgb'][end:end + 1], torch.from_numpy(goal_state[None]), 224, normalization)
        condition = None if physics is None else physics[target['source_episode']]
        if condition is not None:
            from datasets.privileged_physics import physics_observation
            goal = physics_observation(goal, condition)
        goal = {key: value[None].to(next(model.parameters()).device) for key, value in goal.items()}
        planning_targets.append(dict(env=ReacherEnv(episode['physics']), initial=episode['state'][start],
                                     goal_state=goal_state, goal=goal, physics_condition=condition,
                                     context=None if contexts is None else contexts[target['source_episode']]))
    histories = [[] for _ in targets]
    try:
        for results in reacher_target_rounds(model, planning_targets, normalization, cem_settings, generator,
                                             chunk_size=candidate_chunk_size):
            for history, result in zip(histories, results):
                if result is not None:
                    history.append(result)
    finally:
        for target in planning_targets:
            target['env'].physics.free()
    for index, (target, rounds) in enumerate(zip(targets, histories)):
        episode = collection.read_episode(target['source_episode'])
        start, end = target['current_row'], target['goal_row']
        goal_state = episode['state'][end]
        images = np.concatenate([rounds[0]['rgb'][:1], *[r['rgb'][1:] for r in rounds]])
        states = np.concatenate([rounds[0]['state'][:1], *[r['state'][1:] for r in rounds]])
        actions = torch.cat([r['taken'] for r in rounds], 1)
        records.append(dict(target, **reacher_budget_results(states, goal_state), actions=actions[0].cpu().tolist()))
        if physics is not None:
            records[-1]['round_proprio'] = [r['current']['proprio'][0].cpu().tolist() for r in rounds]
        np.savez(output / f'target_{index:03d}.npz', rgb=images, state=states,
                 raw_actions=np.concatenate([r['raw_actions'] for r in rounds]))
        if index < num_visualizations:
            planning_video(output / f'target_{index:03d}.mp4', model, rounds[0]['current'], actions,
                           episode['rgb'][start:], images, episode['rgb'][end])
    metric_names = ('success', 'unwrapped_success', 'wrapped_success', 'joint1_unwrapped_error',
                    'joint2_unwrapped_error', 'joint1_wrapped_error', 'joint2_wrapped_error',
                    'qpos_l2', 'qvel_l2', 'state_l2')
    final_metrics = {(f'{key}_rate' if key.endswith('success') else f'mean_{key}'):
                     sum(float(row[key]) for row in records) / len(records) for key in metric_names}
    summary = dict(metadata, per_target=records, inspection=inspection, seed=seed, search_budget=cem_settings,
                   candidate_evaluation=metadata.get('candidate_evaluation','wm_dino'), frameskip=5, collection_action_repeat=2, goal_H=5,
                   planning_num_hist=1, feedback_history_mode='rolling', final_selection='mean',
                   termination_metric='unwrapped_success', target_raw_action_horizon=25,
                   configured_execution_budget_raw_actions=75, objective_mode='terminal',
                   max_execution_raw_actions=max(r['executed_raw_actions'] for r in records),
                   successes=[r['success'] for r in records],
                   success_threshold_rad_per_joint=float(np.deg2rad(10)), check_initial_success=False,
                   freeze_on_success=False, stop_on_success=True, success_checkpoints_raw_actions=[25, 50, 75],
                   candidate_chunk_size=candidate_chunk_size,
                   rng_schedule='MPC round, CEM iteration, target (draw before inactive skip)',
                   metrics={**final_metrics, **{f'bounded_recovery/unwrapped_success_at_{b}_actions':
                            sum(r[f'cumulative_unwrapped_success_at_{b}_actions'] for r in records) / len(records)
                            for b in (0, 25, 50, 75)}})
    with (output / '0_result_planning.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    return summary


from planning.video import planning_video
