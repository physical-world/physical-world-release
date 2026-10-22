"""Run the bound PushT paper evaluations using canonical components."""

import json
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from runtime.checkpoints import ROOT, create_run, source_revision
from datasets.pusht import PushTManifest, pusht_normalization, pusht_observation
from evaluation.common import evaluation_episode_ids
from models.dino import FrozenDino
from models.visual_world_model import VWorldModel
from planning.actions import PushTActionConverter, pusht_proposal_statistics
from planning.metrics import pusht_budget_results
from planning.mpc import mpc_target_rounds
from planning.targets import read_planning_target, read_simulator_parameters
from datasets.evaluation_targets import prepare_targets, check_target_geometry


@torch.inference_mode()
def export_planning(model, manifest, normalization, config, paper, targets, output, metadata):
    from env.pusht import PushTEnv
    from planning.reacher_export import planning_video
    plan = paper.plan
    search = plan.planner.sub_planner
    settings = dict(horizon=search.horizon, action_dim=10, num_samples=search.num_samples,
                    topk=search.topk, var_scale=search.var_scale, opt_steps=search.opt_steps)
    scale = 100.0
    relative_mean, relative_std = pusht_proposal_statistics(manifest, scale)
    converter = PushTActionConverter(5, scale, relative_mean, relative_std,
                                    normalization['action_mean'], normalization['action_std'])
    output.mkdir()
    records = []
    count = len(targets['targets'])
    limit = count
    samples, planning_targets, replay_goals = [], [], []
    for index in range(limit):
        sample = read_planning_target(manifest, output.parent / 'target_manifest.json', index, 224, normalization)
        physical = read_simulator_parameters(manifest, sample)
        env = PushTEnv()
        initial = sample['state'][0].numpy()
        replay_rgb, replay_state = env.rollout(initial, sample['replay_absolute_actions'].numpy(), **physical)
        observed, _ = pusht_observation(replay_rgb[-1:], torch.from_numpy(replay_state[-1:, :5]), 224, normalization)
        goal = {key: value[None].to(config.device) for key, value in observed.items()}
        samples.append(sample)
        replay_goals.append((replay_rgb[-1], replay_state[-1]))
        planning_targets.append(dict(env=env, initial=initial, physical=physical, goal=goal, goal_state=replay_state[-1]))
    histories = [[] for _ in planning_targets]
    for results in mpc_target_rounds(model, planning_targets, converter, normalization, 224, plan.objective.alpha,
                                     settings, plan.planner.max_iter, plan.planner.n_taken_actions,
                                     torch.Generator().manual_seed(plan.seed), plan.wm_candidate_chunk_size):
        for history, result in zip(histories, results):
            if result is not None:
                history.append(result)
    for index, (sample, target, rounds) in enumerate(zip(samples, planning_targets, histories)):
        env = target['env']
        goal_rgb, goal_state = replay_goals[index]
        states = np.concatenate([rounds[0]['state'][:1], *[r['state'][1:] for r in rounds]]) if rounds else env.state()[None]
        images = np.concatenate([rounds[0]['rgb'][:1], *[r['rgb'][1:] for r in rounds]]) if rounds else env.render()[None]
        actions = torch.cat([r['taken'].cpu() for r in rounds], 1) if rounds else torch.empty(1, 0, 10)
        record = dict(sample['identity'], **pusht_budget_results(states, goal_state), actions=actions[0].tolist())
        records.append(record)
        np.savez(output / f'target_{index:03d}.npz', rgb=images, state=states, actions=actions.numpy(), goal=goal_state)
        with (output / f'target_{index:03d}.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        if index < plan.n_plot_samples and rounds:
            episode = manifest.read_episode(sample['episode_id'])
            start = sample['identity']['offset']
            planning_video(output / f'target_{index:03d}.mp4', model, rounds[0]['current'], actions.to(config.device),
                           episode['rgb'][start:start + 16], images, goal_rgb, fps=10)
        print(f'Planning target {index + 1}/{limit}: {record["final"]}', flush=True)
    metrics = {f'bounded_recovery/{key}_at_{budget}_actions':
               sum(r['cumulative'][budget][key] for r in records) / len(records)
               for budget in (0, 25, 50, 75) for key in ('joint_success', 'block_success')}
    summary = dict(metadata, seed=plan.seed, target_seed=config.target_seed, search_budget=settings,
                   final_selection='mean', rng_schedule='MPC round / CEM iteration / selected targets (draw before inactive skip)',
                   target_set_hash=targets['target_set_hash'], per_target=records, metrics=metrics,
                   successes=[r['final']['joint_success'] for r in records])
    with (output / '0_result_planning.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    return summary


@hydra.main(version_base='1.3', config_path='conf', config_name='eval_pusht')
def main(config):
    """Hydra --cfg job --resolve prints settings without loading models or data."""
    if config.phase not in ('nonplanning', 'planning'):
        raise ValueError('phase must be nonplanning or planning')
    paper = OmegaConf.load(ROOT / config.paper_eval_config)
    paper.eval.seed, paper.plan.seed = config.eval_seed, config.plan_seed
    paper.plan = OmegaConf.merge(paper.plan, config.planning_overrides)
    run = create_run(config)
    OmegaConf.save(paper, run / 'paper_eval.yaml')
    (run / 'source_revision.txt').write_text(f"{source_revision() or 'unavailable'}\n")
    for name in ('eval_pusht.py', 'planning/cem.py', 'planning/mpc.py', 'planning/reacher_export.py', 'evaluation/result_export.py'):
        shutil.copy2(ROOT / name, run / Path(name).name)
    manifest = PushTManifest(ROOT / config.manifest)
    _, targets, sampling = prepare_targets(config, manifest, 'pusht', config.phase, run)
    bound = OmegaConf.load(ROOT / config.checkpoint_config)
    check_target_geometry(targets,bound.frameskip)
    torch.manual_seed(config.eval_seed)
    np.random.seed(config.eval_seed)
    payload = torch.load(ROOT / config.checkpoint, map_location='cpu', weights_only=False)
    model = VWorldModel(bound.img_size, FrozenDino(config.get('dino_weights')), payload['proprio_encoder'],
                        payload['action_encoder'], predictor=payload['predictor'], decoder=payload['decoder'],
                        num_hist=bound.num_hist, num_pred=bound.num_pred,
                        proprio_loss_weight=bound.model.proprio_loss_weight).to(config.device).eval()
    normalization = pusht_normalization(manifest)
    torch.save(normalization, run / 'normalization.pt')
    metadata = dict(ckpt_path=str((ROOT / config.checkpoint).resolve()), ckpt_epoch=payload['epoch'],
                    eval_seed=config.eval_seed, inspection=config.num_targets is not None or config.max_targets is not None or config.max_batches is not None,
                    target_set_hash=targets['target_set_hash'], targets=targets['targets'])
    print(f'Run directory: {run}', flush=True)
    if config.phase == 'nonplanning':
        from evaluation.image_metrics import LPIPS
        from evaluation.segmentation import load_segmentation_model
        from evaluation.result_export import export_nonplanning
        segmentation, _ = load_segmentation_model(ROOT / config.segmentation, config.device)
        episode_ids = evaluation_episode_ids(manifest, targets)
        visualizations = dict(teacher_forced=paper.eval.num_viz_batches,
                              fixed_horizon=paper.eval.num_viz_fixed_horizon_batches,
                              open_loop=paper.eval.num_viz_rollout_episodes)
        export_nonplanning(model, manifest, episode_ids, normalization, run / 'nonplanning', metadata,
                           LPIPS(net_type="vgg", version="0.1").to(config.device).eval().requires_grad_(False), segmentation, batch_size=paper.eval.batch_size,
                           max_batches=config.max_batches, num_visualizations=visualizations,
                           adaptive_history=bound.env.dataset.get('pad_last_frame', False), seed=config.eval_seed)
    else:
        export_planning(model, manifest, normalization, config, paper, targets, run / 'planning', metadata)


if __name__ == '__main__':
    main()
