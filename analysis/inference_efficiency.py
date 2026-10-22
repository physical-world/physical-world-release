"""Paper-bound PushT inputs and latent-only inference with fixed conditions."""
import json
from pathlib import Path

import torch

from runtime.checkpoints import load_world_model
from datasets.common import _read_window, normalize
from datasets.context import read_pusht_context, pusht_physics_targets
from datasets.privileged_physics import physics_observation
from datasets.pusht import PushTManifest, pusht_normalization, pusht_observation
from models.visual_world_model import VWorldModel


def load_method(config, method, device='cuda', *, load_decoder=False):
    asset = config['methods'][method]
    model, _, _ = load_world_model(asset['checkpoint'], asset['model_config'],
                                  device=device, load_decoder=load_decoder)
    return model.float().eval().requires_grad_(False)


def load_method_estimator(config, method, device='cuda'):
    """Frozen Estimated Physics regressor; PushT outputs are already Privileged Physics inputs."""
    asset = config['methods'][method]
    if 'estimator' not in asset:
        return None
    from models.physics.estimator import load_physics_estimator
    estimator, metadata, _ = load_physics_estimator(asset['estimator'], device)
    if metadata['mean'] != [0., 0.] or metadata['std'] != [1., 1.]:
        raise ValueError(f"{asset['estimator']}: expected unscaled PushT physics outputs")
    return estimator.float()


def attach_estimate(estimator, obs, *observations):
    """Estimate physics once from the context clip and append it to every observation."""
    physics = estimator({k:v for k,v in obs.items() if k.startswith('ctx_')})[:, None]
    return [physics_observation({k:v for k,v in o.items() if not k.startswith('ctx_')}, physics)
            for o in (obs, *observations)]


def input_samples(config, method, *, teacher=False, device='cuda', limit=None):
    collection = PushTManifest(config['manifest'])
    stats = pusht_normalization(collection)
    stats.update({k: torch.tensor(v) for k, v in config['normalization'].items()})
    validation = collection.split_episode_ids()['validation']
    history = 15 if teacher and method == 'long_history' else 3
    samples = []
    for row in config['samples'][:limit]:
        episode = validation[row['source_episode']]
        path, _, begin, end = collection.episode_info(episode)
        start = begin + row['current_row'] - (history-1)*5
        data = collection.group(path)['data']
        window = _read_window(data, start, history, 5, stats['action_mean'], stats['action_std'], episode_stop=end)
        obs, _ = pusht_observation(window['rgb'], window['state'], 224, stats)
        actions = torch.from_numpy(data['action'][start:start+(history-1+20)*5]).float()
        actions = normalize(actions, stats['action_mean'], stats['action_std']).reshape(history-1+20, -1)
        if method == 'privileged_physics':
            obs = physics_observation(obs, pusht_physics_targets(collection, [episode])[0])
        if method in ('context', 'prefix_context_joint', 'estimated_physics'):
            donor = validation[row['context_source_episode']]
            obs.update(read_pusht_context(collection, {'donor':donor, 'start':row['context_start']}, 60, 5, 224, stats))
        samples.append(({k:v[None].to(device) for k,v in obs.items()}, actions[None].to(device)))
    return samples


@torch.inference_mode()
def prepare_condition(model, obs, *, joint_prefix=False, estimator=None):
    if estimator is not None:
        obs, = attach_estimate(estimator, obs)
    if joint_prefix:
        keys = ['visual'] + (['proprio'] if model.proprio_encoder is not None else [])
        frames = obs['ctx_visual'].shape[1]
        encoded = model.encode_obs({k:torch.cat((obs['ctx_'+k], obs[k]), dim=1) for k in keys})
        main = {k:v[:, frames:] for k,v in encoded.items()}
        donor = {k:v[:, :frames] for k,v in encoded.items()}
        condition = VWorldModel.encode_precomputed_obs(model, donor, obs['ctx_act'])
    else:
        condition = model._rollout_condition(obs)
        main = model.encode_obs(obs)
    return {'z_obs':main, 'ctx_emb':condition}


@torch.inference_mode()
def rollout_final(model, prepared, actions):
    observation, condition = prepared['z_obs'], prepared['ctx_emb']
    initial = observation['visual'].shape[1]
    if model.condition_dim:
        z = model.encode_precomputed_obs(observation, actions[:, :initial], condition)
    else:
        z = model.encode_precomputed_obs(observation, actions[:, :initial])
    for step in range(initial, actions.shape[1]):
        predicted = model._rollout_prediction(z[:, -model.num_hist:], condition).clone()
        model.replace_actions_from_z(predicted, actions[:, step:step+1])
        keep = model.num_hist - 1
        z = torch.cat((z[:, -keep:] if keep else z[:, :0], predicted), dim=1)
    final = model._rollout_prediction(z[:, -model.num_hist:], condition)
    return model.separate_emb(final)[0]


def total_rollout(model, sample, horizon, *, joint_prefix=False, estimator=None):
    obs, actions = sample
    count = obs['visual'].shape[1] - 1 + horizon
    prepared = prepare_condition(model, obs, joint_prefix=joint_prefix, estimator=estimator)
    return rollout_final(model, prepared, actions[:, :count])


def cached_step(model, prepared, actions):
    count = prepared['z_obs']['visual'].shape[1]
    final = rollout_final(model, prepared, actions[:, :count])
    updated = {k:torch.cat((v[:, 1:], final[k]), dim=1) for k,v in prepared['z_obs'].items()}
    return {'z_obs':updated, 'ctx_emb':prepared['ctx_emb']}


def measure(operation, samples, warmup=20, repeats=100):
    import time
    import numpy as np
    for index in range(warmup):
        operation(samples[index % len(samples)])
    raw = []
    for index in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation(samples[index % len(samples)])
        torch.cuda.synchronize()
        raw.append((time.perf_counter()-start)*1000)
    return {'raw_ms':raw, 'mean_ms':float(np.mean(raw)), 'median_ms':float(np.median(raw)),
            'p90_ms':float(np.percentile(raw, 90)), 'warmup':warmup, 'repeats':repeats}


def benchmark_inference(model, samples, teacher, config, method, *, warmup=None, repeats=None, estimator=None):
    settings = {'warmup':config['warmup'] if warmup is None else warmup,
                'repeats':config['repeats'] if repeats is None else repeats}
    options = {'joint_prefix':method == 'prefix_context_joint', 'estimator':estimator}
    # Estimated Physics observations need the estimate, so their encoding is timed within initialization.
    operations = {} if estimator is not None else {'initial_observation':lambda s:model.encode_obs(s[0])}
    operations['initialization'] = lambda s:prepare_condition(model, s[0], **options)
    if hasattr(model, 'encode_context'):
        operations['condition'] = lambda s:model.encode_context(s[0])
    if estimator is not None:
        operations['condition'] = lambda s:estimator({k:v for k,v in s[0].items() if k.startswith('ctx_')})
    result = {name:measure(operation, samples, **settings) for name,operation in operations.items()}
    prepared = [(prepare_condition(model, obs, **options), actions) for obs,actions in samples]
    result['cached_step'] = measure(lambda s:cached_step(model, *s), prepared, **settings)
    result['teacher_fresh'] = measure(lambda s:total_rollout(model, s, 1, **options), teacher, **settings)
    for horizon in config['horizons']:
        result[f'total_h{horizon}'] = measure(lambda s:total_rollout(model, s, horizon, **options), samples, **settings)
    return result


def runtime_metadata():
    return {'gpu':torch.cuda.get_device_name(), 'torch':str(torch.__version__), 'cuda':torch.version.cuda,
            'dtype':'float32', 'batch_size':1, 'threads':torch.get_num_threads(),
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'timing':'synchronized wall clock; GPU-resident inputs; decoder excluded',
            'paper_table_statistic':'mean', 'paper_figure_statistic':'median'}



def planning_inputs(config, method, target_manifest, context_manifest, device='cuda'):
    from env.pusht import PushTEnv
    from planning.targets import read_planning_target, read_simulator_parameters
    from planning.actions import PushTActionConverter, pusht_proposal_statistics
    collection = PushTManifest(config['manifest'])
    stats = pusht_normalization(collection)
    stats.update({k:torch.tensor(v) for k,v in config['normalization'].items()})
    target = read_planning_target(collection, target_manifest, 0, 224, stats)
    physical = read_simulator_parameters(collection, target)
    env = PushTEnv(render_size=224)
    rgb, state = env.rollout(target['state'][0].numpy(), target['replay_absolute_actions'].numpy(), **physical)
    current, _ = pusht_observation(rgb[:1], torch.from_numpy(state[:1, :5]), 224, stats)
    goal, _ = pusht_observation(rgb[-1:], torch.from_numpy(state[-1:, :5]), 224, stats)
    if method == 'privileged_physics':
        physics = torch.as_tensor(physical['cog_local_normed']).float()
        current, goal = physics_observation(current, physics), physics_observation(goal, physics)
    if method in ('context', 'prefix_context_joint', 'estimated_physics'):
        record = context_manifest['protocols']['planning'][0]
        geometry = context_manifest['geometry']
        current.update(read_pusht_context(collection, record, geometry['ctx_length'],
                                         geometry['ctx_frameskip'], 224, stats))
    relative_mean, relative_std = pusht_proposal_statistics(collection, 100.)
    converter = PushTActionConverter(5, 100., relative_mean, relative_std, stats['action_mean'], stats['action_std'])
    return {'current':{k:v[None].to(device) for k,v in current.items()},
            'goal':{k:v[None].to(device) for k,v in goal.items()}, 'converter':converter,
            'anchor':torch.as_tensor(state[:1, :2], device=device), 'identity':target['identity']}


@torch.inference_mode()
def planning_solve(model, inputs, config, method, *, cem=None, estimator=None):
    from planning.cem import cem_plan
    from planning.candidates import latent_goal_cost
    settings = config['planning']
    current, goal = inputs['current'], inputs['goal']
    if estimator is not None:
        current, goal = attach_estimate(estimator, current, goal)
    prepared = prepare_condition(model, current, joint_prefix=method == 'prefix_context_joint')
    goal = model.encode_obs(goal)
    def score(proposals):
        execution = inputs['converter'].proposal_to_execution(proposals, inputs['anchor'])
        costs = []
        for actions in execution.split(settings['candidate_chunk_size']):
            count = actions.shape[0]
            expanded = {'z_obs':{k:v.expand(count, *v.shape[1:]) for k,v in prepared['z_obs'].items()},
                        'ctx_emb':None if prepared['ctx_emb'] is None else prepared['ctx_emb'].expand(count, *prepared['ctx_emb'].shape[1:])}
            final = rollout_final(model, expanded, actions)
            costs.append(latent_goal_cost(final, goal, settings['alphas'][method])['total'])
        return torch.cat(costs)
    return cem_plan(score, **(settings['cem'] if cem is None else cem),
                    device=inputs['anchor'].device, generator=torch.Generator().manual_seed(settings['seed']))


def benchmark_planning(model, inputs, config, method, *, warmup=None, repeats=None, cem=None, estimator=None):
    setting = config['planning']
    result = measure(lambda _:planning_solve(model, inputs, config, method, cem=cem, estimator=estimator), [inputs],
                     setting['warmup'] if warmup is None else warmup,
                     setting['repeats'] if repeats is None else repeats)
    return {'solve':result, 'selection':'mean', 'target':inputs['identity'],
            'included':['fresh condition/current/goal encoding', 'CEM candidate scoring'],
            'excluded':['model/data loading', 'replay and target preparation', 'planner construction',
                        'simulator execution', 'decoder', 'file writes'],
            'alpha':setting['alphas'][method], 'seed':setting['seed'],
            'cem':setting['cem'] if cem is None else cem}

def main():
    import argparse
    from analysis.artifacts import create_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('conf/analysis/inference_efficiency.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--planning', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU benchmark requires a CUDA GPU')
    config = json.loads(args.config.read_text())
    output = create_output(args.output, config)
    torch.set_num_threads(config['threads'])
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    torch.backends.cudnn.benchmark = False
    if args.planning:
        from datasets.evaluation_targets import generate_targets, sampling_protocol, manifest_text
        from datasets.evaluation_contexts import resolve_contexts
        collection = PushTManifest(config['manifest'])
        sampling = sampling_protocol('pusht')
        targets = generate_targets(collection, 'pusht', 'planning', sampling['target']['planning'],
                                   num_targets=1, seed=config['planning']['target_seed'])
        contexts = resolve_contexts(collection, targets, sampling['context'], 'planning',
                                    seed=config['planning']['ctx_seed'])
        target_path = output/'target_manifest.json'
        target_path.write_text(manifest_text(targets))
        (output/'ctx_manifest.json').write_text(manifest_text(contexts))
    for method in args.methods or config['methods']:
        model = load_method(config, method)
        estimator = load_method_estimator(config, method)
        if args.planning:
            inputs = planning_inputs(config, method, target_path, contexts)
            result = benchmark_planning(model, inputs, config, method, estimator=estimator)
        else:
            samples = input_samples(config, method)
            teacher = input_samples(config, method, teacher=True)
            result = benchmark_inference(model, samples, teacher, config, method, estimator=estimator)
        (output/f'{method}.json').write_text(json.dumps({'method':method, 'latency':result,
            'environment':runtime_metadata(), 'asset_binding':config['methods'][method]}, indent=2)+'\n')
        del model, estimator


if __name__ == '__main__':
    main()
