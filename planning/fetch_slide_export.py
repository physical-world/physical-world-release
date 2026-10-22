"""Bound FetchSlide direct planning with horizon-grouped CEM and physical outcomes."""
import json
from pathlib import Path

import numpy as np
import torch
from datasets.common import preprocess_rgb
from datasets.privileged_physics import physics_observation
from env.fetch_slide import FetchSlideEnv
from planning.fetch_slide import fetch_search, fetch_execution, fetch_raw_hit, fetch_success
from planning.video import planning_video


@torch.inference_mode()
def export_fetch_planning(model, collection, targets, normalization, output, metadata, cem_settings,
                          *, seed=42, inspection=False, num_visualizations=1, candidate_chunk_size=None, physics=None, contexts=None):
    output = Path(output).resolve()
    output.relative_to((Path(__file__).parents[1] / 'logs').resolve())
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    inputs, selected = {}, {}
    for horizon in sorted({t['wm_horizon_tokens'] for t in targets}):
        indices = [i for i, t in enumerate(targets) if t['wm_horizon_tokens'] == horizon]
        searches = []
        for index in indices:
            target = targets[index]
            _, _, begin, _ = collection.episode_info(target['source_episode'])
            rows = np.array([target['current_row'], target['goal_row']]) + begin
            images = preprocess_rgb(collection.root['data/img'].oindex[rows], 224).to(device)
            current, goal = {'visual': images[None, :1]}, {'visual': images[None, 1:]}
            if physics is not None:
                condition = physics[target['source_episode']]
                current = physics_observation(current,condition)
                goal = physics_observation(goal,condition)
            if contexts is not None:
                current.update({key:value[None].to(device) for key,value in contexts[target['source_episode']].items()})
            inputs[index] = current
            searches.append(fetch_search(model, current, goal, normalization, horizon, cem_settings, generator, candidate_chunk_size))
        # Match the reference order: horizon, iteration, then target in manifest order.
        for updates in zip(*searches):
            for index, update in zip(indices, updates):
                selected[index] = update['next_mean'][None]
    records = []
    for index, target in enumerate(targets):
        episode = collection.read_episode(target['source_episode'])
        execution = fetch_execution(selected[index], normalization, target['wm_horizon_tokens'])
        raw = fetch_raw_hit(execution, normalization)[0].cpu().numpy()
        env = FetchSlideEnv(episode['physics'])
        try:
            trajectory = env.rollout(raw)
            planned = env.render_qposes(trajectory['qpos'])
        finally:
            env.close()
        goal_xy = [target['goal_puck_x'], target['goal_puck_y']]
        outcome = fetch_success(trajectory['final_state'], goal_xy)
        records.append(dict(target, **outcome, action_xy=raw.tolist(), final_state=trajectory['final_state'].tolist(),
                            terminal=trajectory['terminal'], terminal_step=trajectory['terminal_step'],
                            retraction_start_step=trajectory['retraction_start_step'], retraction_step=trajectory['retraction_step'],
                            executed_controls=len(trajectory['state'])-1))
        np.savez(output / f'target_{index:03d}.npz', rgb=planned, **{key:trajectory[key] for key in ('state','action','qpos','park_position')})
        if index < num_visualizations:
            planning_video(output / f'target_{index:03d}.mp4', model, inputs[index], execution,
                           episode['rgb'], planned, episode['rgb'][target['goal_row']])
    summary = dict(metadata, per_target=records, inspection=inspection, seed=seed, search_budget=cem_settings,
                   candidate_chunk_size=candidate_chunk_size, final_selection='mean', proposal_horizon=1, proposal_action_dim=2, frameskip=5,
                   candidate_evaluation=metadata.get('candidate_evaluation','wm_dino'), objective_mode='terminal', success_threshold_m=.03,
                   rng_schedule='horizon, CEM iteration, target in manifest order',
                   success_rate=sum(row['success'] for row in records)/len(records),
                   mean_endpoint_error_m=sum(row['endpoint_error_m'] for row in records)/len(records))
    with (output / '0_result_planning.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    return summary
