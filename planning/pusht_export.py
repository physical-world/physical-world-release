"""PushT bound MPC export with optional physical conditioning."""
import json
from pathlib import Path

import numpy as np
import torch

from datasets.pusht import pusht_observation
from datasets.privileged_physics import physics_observation
from env.pusht import PushTEnv
from planning.actions import PushTActionConverter,pusht_proposal_statistics
from planning.targets import read_planning_target,read_simulator_parameters
from planning.mpc import mpc_target_rounds
from planning.metrics import pusht_budget_results
from planning.video import planning_video


@torch.inference_mode()
def export_pusht_planning(model,manifest,targets,normalization,output,metadata,cem_settings,
                          *,alpha,seed=42,inspection=False,num_visualizations=1,condition_on_physics=False,physics=None,contexts=None):
    output = Path(output).resolve()
    output.relative_to((Path(__file__).parents[1]/'logs').resolve())
    output.mkdir(parents=True,exist_ok=False)
    relative_mean,relative_std = pusht_proposal_statistics(manifest,100.0)
    converter = PushTActionConverter(5,100.0,relative_mean,relative_std,
                                    normalization['action_mean'],normalization['action_std'])
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    records, samples, planning_targets, replay_goals = [], [], [], []
    for index in range(len(json.loads(Path(targets).read_text())['targets'])):
        sample = read_planning_target(manifest,targets,index,224,normalization)
        physical = read_simulator_parameters(manifest,sample)
        condition = (physics[sample['episode_id']] if physics is not None else
                     torch.as_tensor(physical['cog_local_normed']).float() if condition_on_physics else None)
        env = PushTEnv(render_size=224)
        initial = sample['state'][0].numpy()
        replay_rgb,replay_state = env.rollout(initial,sample['replay_absolute_actions'].numpy(),**physical)
        goal,_ = pusht_observation(replay_rgb[-1:],torch.from_numpy(replay_state[-1:,:5]),224,normalization)
        if condition is not None:
            goal = physics_observation(goal,condition)
        context = None if contexts is None else contexts[sample['episode_id']]
        goal = {key:value[None].to(device) for key,value in goal.items()}
        samples.append(sample)
        replay_goals.append((replay_rgb[-1],replay_state[-1]))
        planning_targets.append(dict(env=env,initial=initial,physical=physical,goal=goal,
                                     goal_state=replay_state[-1],physics_condition=condition,context=context))
    histories = [[] for _ in planning_targets]
    for results in mpc_target_rounds(model,planning_targets,converter,normalization,224,
                                     alpha,cem_settings,3,5,generator):
        for history,result in zip(histories,results):
            if result is not None:
                history.append(result)
    for index,(sample,target,rounds) in enumerate(zip(samples,planning_targets,histories)):
        env,goal,physical = target['env'],target['goal'],target['physical']
        goal_rgb,goal_state = replay_goals[index]
        if rounds:
            images = np.concatenate([rounds[0]['rgb'][:1],*[row['rgb'][1:] for row in rounds]])
            states = np.concatenate([rounds[0]['state'][:1],*[row['state'][1:] for row in rounds]])
            actions = torch.cat([row['taken'] for row in rounds],1)
        else:
            images,states = env.render()[None],env.state()[None]
            actions = torch.empty((1,0,10),device=device)
        outcome = pusht_budget_results(states,goal_state)
        records.append(dict(target=index,identity=sample['identity'],**outcome,actions=actions[0].cpu().tolist(),
                            round_proprio=[row['current']['proprio'][0].cpu().tolist() for row in rounds],
                            goal_proprio=goal['proprio'][0].cpu().tolist(),
                            simulator_cog=np.asarray(physical['cog_local_normed']).tolist()))
        np.savez(output/f'target_{index:03d}.npz',rgb=images,state=states,goal_state=goal_state)
        if index<num_visualizations and rounds:
            path = sample['shard_path']
            start = int(sample['shard_frames'][0])
            recorded = manifest.group(path)['data/img'][start:int(sample['shard_frames'][1])+1]
            planning_video(output/f'target_{index:03d}.mp4',model,rounds[0]['current'],actions,
                           recorded,images,goal_rgb,fps=10)
    summary = dict(metadata,per_target=records,inspection=inspection,seed=seed,search_budget=cem_settings,
                   goal_source='simulator replay of recorded controls',final_selection='mean',alpha=alpha,
                   rng_schedule='MPC round / CEM iteration / selected targets (draw before inactive skip)',
                   configured_execution_budget=75,success_rate=float(np.mean([r['final']['joint_success'] for r in records])))
    with (output/'0_result_planning.json').open('x') as stream:
        json.dump(summary,stream,indent=2)
    return summary
