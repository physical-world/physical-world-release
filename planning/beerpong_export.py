"""Direct BeerPong CEM, physical final-in-cup outcomes, and control-aligned videos."""
import json
from pathlib import Path
import numpy as np
import torch
from datasets.common import preprocess_rgb
from datasets.privileged_physics import physics_observation
from env.beerpong import BeerPongEnv, render_qpos
from planning.beerpong import beerpong_search, beerpong_execution, beerpong_raw_action
from planning.video import planning_video


@torch.inference_mode()
def export_beerpong_planning(model, collection, targets, normalization, output, metadata,
                            cem_settings, *, seed=42, inspection=False, num_visualizations=1, candidate_chunk_size=8, physics=None, contexts=None):
    output = Path(output).resolve()
    output.relative_to((Path(__file__).parents[1] / 'logs').resolve())
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    inputs, searches, selected = [], [], []
    for target in targets:
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
        inputs.append(current)
        searches.append(beerpong_search(model, current, goal, normalization, cem_settings, generator, candidate_chunk_size))
    for updates in zip(*searches):
        selected = [update['next_mean'][None] for update in updates]
    records = []
    for index, target in enumerate(targets):
        episode = collection.read_episode(target['source_episode'])
        execution = beerpong_execution(selected[index], normalization)
        raw = beerpong_raw_action(execution, normalization)[0].cpu().numpy()
        env = BeerPongEnv(episode['physics'])
        try:
            trajectory = env.rollout(raw)
            planned = env.render_qposes(render_qpos(trajectory['qpos']))
        finally:
            env.close()
        final = trajectory['qpos'][-1, 7:10]
        records.append(dict(target, selected_action=raw.tolist(), planned_final_ball_xyz=final.tolist(),
            goal_endpoint_error_m=float(np.linalg.norm(final - np.asarray(target['goal_ball_xyz']))),
            gt_action_l2_error=float(np.linalg.norm(raw - np.asarray(target['gt_action']))),
            final_in_cup=trajectory['final_in_cup'], ever_in_cup=trajectory['ever_in_cup'],
            off_table=trajectory['off_table'], executed_controls=300))
        np.savez(output / f'target_{index:03d}.npz', rgb=planned, **trajectory)
        if index < num_visualizations:
            planning_video(output / f'target_{index:03d}.mp4', model, inputs[index], execution,
                           episode['rgb'], planned, episode['rgb'][target['goal_row']], frameskip=15, fps=100)
    summary = dict(metadata, per_target=records, inspection=inspection, seed=seed,
        search_budget=cem_settings, candidate_chunk_size=candidate_chunk_size, final_selection='mean', proposal_horizon=1,
        expanded_wm_horizon=20, proposal_action_dim=2, frameskip=15,
        candidate_evaluation=metadata.get('candidate_evaluation','wm_dino'), objective_mode='terminal', success_metric='final_in_cup',
        rng_schedule='CEM iteration, target in manifest order',
        success_rate=float(np.mean([r['final_in_cup'] for r in records])),
        ever_in_cup_rate=float(np.mean([r['ever_in_cup'] for r in records])))
    with (output / '0_result_planning.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    return summary
