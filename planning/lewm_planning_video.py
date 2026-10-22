"""Decode saved LeWM planning controls without rerunning the search."""
import json
from pathlib import Path
import numpy as np
import torch

from datasets.common import preprocess_rgb
from planning.video import planning_video


@torch.inference_mode()
def render_saved_planning(model, collection, normalization, environment, method, run, output, contexts=None):
    run, output = Path(run), Path(output).resolve()
    from runtime.checkpoints import ROOT
    output.relative_to((ROOT/'logs').resolve())
    with output.open('xb'):
        pass
    target = json.loads((run/'planning/0_result_planning.json').read_text())['per_target'][0]
    trajectory = np.load(run/'planning/target_000.npz')
    device = next(model.parameters()).device
    if environment == 'pusht':
        from datasets.pusht import pusht_observation
        from planning.targets import read_planning_target, read_simulator_parameters
        from env.pusht import PushTEnv
        target_path = json.loads((run/'bindings.json').read_text())['targets']
        sample = read_planning_target(collection, target_path, 0, 224, normalization)
        env = PushTEnv(render_size=224)
        rgb, state = env.rollout(sample['state'][0].numpy(), sample['replay_absolute_actions'].numpy(),
                                 **read_simulator_parameters(collection, sample))
        initial, _ = pusht_observation(rgb[:1], torch.from_numpy(state[:1, :5]), 224, normalization)
        episode_id, goal = sample['episode_id'], rgb[-1]
        recorded = collection.group(sample['shard_path'])['data/img'][int(sample['shard_frames'][0]):int(sample['shard_frames'][1])+1]
        actions = torch.tensor(target['actions'], device=device)[None]
    else:
        episode_id = target['source_episode']
        episode = collection.read_episode(episode_id)
        recorded, goal = episode['rgb'][target['current_row']:], episode['rgb'][target['goal_row']]
        initial = {'visual':preprocess_rgb(recorded[:1], 224)}
        if environment == 'reacher':
            initial = {'visual':preprocess_rgb(trajectory['rgb'][:1], 224)}
            actions = torch.tensor(target['actions'], device=device)[None]
        elif environment == 'fetch_slide':
            from planning.fetch_slide import fetch_execution
            proposal = (torch.tensor(target['action_xy']) - normalization['action_mean'][:2])/normalization['action_std'][:2]
            actions = fetch_execution(proposal[None, None].to(device), normalization, target['wm_horizon_tokens'])
        else:
            from planning.beerpong import beerpong_execution
            proposal = (torch.tensor(target['selected_action']) - normalization['action_mean'])/normalization['action_std']
            actions = beerpong_execution(proposal[None, None].to(device), normalization)
    if contexts is not None:
        initial.update(contexts[episode_id])
    initial = {key:value[None].to(device) for key,value in initial.items()}
    planning_video(output, model, initial, actions, recorded, trajectory['rgb'], goal,
                   frameskip=15 if environment == 'beerpong' else 5, fps=100 if environment == 'beerpong' else 10)
    return dict(video=str(output), planned_controls=len(trajectory['rgb'])-1,
                context_shape=None if contexts is None else list(initial['ctx_emb'].shape))
