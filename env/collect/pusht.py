"""Paper PushT fresh-policy IID collection, independent of the original repo."""

import numpy as np

from env.pusht import PushTEnv, sample_t_cog


SIGMAS = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 15.0)


def episode_noise_sigmas(clean_episodes_per_dynamics=1, noise_sigmas_px=SIGMAS, noise_repeats=10):
    for name, count in [('clean_episodes_per_dynamics', clean_episodes_per_dynamics),
                        ('noise_repeats', noise_repeats)]:
        if type(count) is not int or count < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in noise_sigmas_px):
        raise ValueError('noise_sigmas_px must contain finite positive values')
    sigmas = [0.0] * clean_episodes_per_dynamics + [
        sigma for sigma in noise_sigmas_px for _ in range(noise_repeats)]
    if not sigmas:
        raise ValueError('The sampling recipe must produce at least one episode per dynamics')
    return sigmas


def episode_identity(cog_seed, episode_index, episode_sigmas=None):
    if episode_sigmas is None:
        episode_sigmas = episode_noise_sigmas()
    if not 0 <= episode_index < len(episode_sigmas):
        raise ValueError('episode_index is outside the sampling recipe')
    index = (cog_seed - 5001) * len(episode_sigmas) + episode_index
    sigma = episode_sigmas[episode_index]
    return dict(cog_seed=int(cog_seed), reset_seed=603000 + index,
                noise_seed=703000 + index, noise_sigma_px=sigma,
                split='train' if cog_seed <= 5135 else 'validation')


def initial_state(reset_seed):
    rng = np.random.default_rng(reset_seed)
    return np.array([rng.integers(50, 450), rng.integers(50, 450),
                     rng.integers(100, 400), rng.integers(100, 400),
                     rng.uniform(-np.pi, np.pi), 0.0, 0.0])


def normalized_cog(cog_seed):
    cog = sample_t_cog(cog_seed).astype(np.float32)
    return np.array([cog[0] / 60, cog[1] / 60 - 1], dtype=np.float32)


def reset_episode(env, identity):
    return env.reset(initial_state(identity['reset_seed']), identity['cog_seed'],
                     normalized_cog(identity['cog_seed']))


def keypoints(env):
    shapes = sorted(env.block.shapes, key=lambda shape: max(v.y for v in shape.get_vertices()))
    return np.array([tuple(v.rotated(env.block.angle) + env.block.position)
                     for shape in shapes for v in shape.get_vertices()], dtype=np.float32)


def coverage(env):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import pymunk
    goal = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    goal.position = tuple(env.goal_pose[:2])
    goal.angle = env.goal_pose[2]
    def geometry(body):
        return unary_union([Polygon([tuple(body.local_to_world(v)) for v in shape.get_vertices()])
                            for shape in env.block.shapes])
    target = geometry(goal)
    return geometry(env.block).intersection(target).area / target.area


def collect_episode(policy, identity, max_steps=300, image_size=224, *, stop_on_success=True):
    env = PushTEnv(image_size)
    rgb, state = reset_episode(env, identity)
    rng = np.random.default_rng(identity['noise_seed'])
    fields = ['img', 'state', 'agent_vel', 'action', 'base_action', 'iid_noise_px',
              'keypoint', 'n_contacts']
    rows = {name: [] for name in fields}
    contact_points = [0]
    def contact(arbiter, space, data):
        contact_points[0] += len(arbiter.contact_point_set.points)
    env.space.add_collision_handler(0, 0).post_solve = contact
    coverages = []
    for _ in range(max_steps):
        action = policy.action(env).astype(np.float32)
        target = np.clip(state[:2] + action * 30, 0, 512).astype(np.float32)
        sigma = identity['noise_sigma_px']
        noise = rng.normal(0, sigma, 2).astype(np.float32) if sigma > 0 else np.zeros(2, np.float32)
        absolute = (target + noise).astype(np.float32)
        rows['img'].append(rgb)
        rows['state'].append(state[:5])
        rows['agent_vel'].append(state[5:7])
        rows['action'].append(absolute)
        rows['base_action'].append(target)
        rows['iid_noise_px'].append(noise)
        rows['keypoint'].append(keypoints(env))
        contact_points[0] = 0
        velocity_action = ((absolute - state[:2]) / 30).astype(np.float32)
        executed_target = (state[:2] + velocity_action * 30).astype(np.float32)
        rgb, state = env.step(executed_target)
        rows['n_contacts'].append([np.ceil(contact_points[0] / 1000)])
        coverages.append(coverage(env))
        if stop_on_success and coverages[-1] > .95:
            break
    episode = {name: np.stack(values) for name, values in rows.items()}
    episode['n_contacts'] = episode['n_contacts'].astype(np.float32)
    length = len(episode['state'])
    for name in ['cog_seed', 'reset_seed']:
        episode[name] = np.full(length, identity[name], dtype=np.int32)
    episode['cog_local_normed'] = np.broadcast_to(normalized_cog(identity['cog_seed']), (length, 2)).copy()
    episode['noise_sigma_px'] = np.full(length, identity['noise_sigma_px'], dtype=np.float32)
    episode['noise_rho'] = np.zeros(length, dtype=np.float32)
    flags = np.full(length, identity['noise_sigma_px'] > 0, dtype=np.uint8)
    episode['is_perturbed'] = flags
    episode['is_noisy_replay'] = np.zeros(length, dtype=np.uint8)
    return episode, dict(identity, success=coverages[-1] > .95,
                         final_coverage=coverages[-1], max_coverage=max(coverages),
                         final_state=state.tolist(),
                         n_contact_steps=int(np.count_nonzero(episode['n_contacts'][:, 0])),
                         block_displacement=float(np.linalg.norm(state[2:4] - episode['state'][0, 2:4])))


def render_mask(env):
    """Match the original four-label rasterizer and nearest-neighbor resize."""
    import cv2
    import pymunk
    canvas = np.zeros((512, 512), dtype=np.uint8)
    goal = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    goal.position = tuple(env.goal_pose[:2])
    goal.angle = env.goal_pose[2]
    for label, body in [(1, goal), (2, env.block)]:
        for shape in env.block.shapes:
            vertices = [body.local_to_world(v) for v in shape.get_vertices()]
            points = np.array([[int(v.x), int(512 - v.y)] for v in vertices], dtype=np.int32)
            cv2.fillPoly(canvas, [points], label)
    x, y = env.agent.position
    cv2.circle(canvas, (int(x), int(512 - y)), 15, 3, thickness=-1)
    resized = cv2.resize(canvas, (env.render_size, env.render_size), interpolation=cv2.INTER_NEAREST)
    return resized[::-1].copy()


def episode_masks(episode, image_size=224):
    env = PushTEnv(image_size)
    seed = int(episode['cog_seed'][0])
    env.reset(np.r_[episode['state'][0], episode['agent_vel'][0]], seed, normalized_cog(seed))
    images = []
    for state in episode['state']:
        env.block.angle = float(state[4])
        env.block.position = tuple(state[2:4].astype(float))
        env.agent.position = tuple(state[:2].astype(float))
        images.append(render_mask(env))
    return np.stack(images)


def collect(output, checkpoint, cog_seeds=range(5001, 5151), clean_episodes_per_dynamics=1,
            noise_sigmas_px=SIGMAS, noise_repeats=10,
            max_steps=300, image_size=224, device='cpu'):
    """Write at most five CoGs per shard in input order, preserving split boundaries."""
    import json
    from pathlib import Path
    import zarr
    from env.collect.common import prepare_output, append_arrays, save_run, CollectionLog
    from env.collect.pusht_policy import load_policy
    episode_sigmas = episode_noise_sigmas(clean_episodes_per_dynamics, noise_sigmas_px, noise_repeats)
    episodes_per_dynamics = len(episode_sigmas)
    output = prepare_output(output)
    with CollectionLog(output, 'pusht', len(cog_seeds),
                       len(cog_seeds) * episodes_per_dynamics, 'cog_seed', directory=True) as progress:
        progress.logger.info('loading policy checkpoint=%s device=%s', checkpoint, device)
        policy = load_policy(checkpoint, device)
        config = dict(checkpoint=str(Path(checkpoint).resolve()), cog_seeds=list(cog_seeds),
                      clean_episodes_per_dynamics=clean_episodes_per_dynamics,
                      noise_sigmas_px=list(noise_sigmas_px), noise_repeats=noise_repeats,
                      episodes_per_dynamics=episodes_per_dynamics,
                      max_steps=max_steps, image_size=image_size, device=device)
        (output / 'collection.json').write_text(json.dumps(config, indent=2))
        save_run(output / 'manifest.json', config)
        shards = []
        count = 0
        sums = {name: np.zeros(2) for name in ['absolute', 'relative_delta']}
        squares = {name: np.zeros(2) for name in sums}
        seed_groups = []
        for seed in cog_seeds:
            split = episode_identity(seed, 0, episode_sigmas)['split']
            if (not seed_groups or len(seed_groups[-1]) == 5 or
                    split != episode_identity(seed_groups[-1][-1], 0, episode_sigmas)['split']):
                seed_groups.append([])
            seed_groups[-1].append(seed)
        for shard_index, seeds in enumerate(seed_groups):
            path = output / f'rollouts/shard_{shard_index:02d}.zarr'
            root = zarr.open_group(str(path), mode='w-')
            data = root.create_group('data')
            meta = root.create_group('meta')
            lengths, noise_seeds, successes, final_coverages, max_coverages = [], [], [], [], []
            contact_steps, displacements = [], []
            stop = 0
            for seed in seeds:
                progress.start_group(seed)
                group_frames = stop
                for episode_index in range(episodes_per_dynamics):
                    progress.episode = episode_index
                    identity = episode_identity(seed, episode_index, episode_sigmas)
                    episode, record = collect_episode(policy, identity, max_steps, image_size)
                    episode['img_mask'] = episode_masks(episode, image_size)
                    append_arrays(data, episode)
                    stop += len(episode['state'])
                    lengths.append(stop)
                    noise_seeds.append(identity['noise_seed'])
                    successes.append(record['success'])
                    final_coverages.append(record['final_coverage'])
                    max_coverages.append(record['max_coverage'])
                    contact_steps.append(record['n_contact_steps'])
                    displacements.append(record['block_displacement'])
                    if identity['split'] == 'train':
                        count += len(episode['action'])
                        for name, values in [('absolute', episode['action']),
                                             ('relative_delta', episode['action'] - episode['state'][:, :2])]:
                            values = values.astype(np.float64)
                            sums[name] += values.sum(0)
                            squares[name] += np.square(values).sum(0)
                progress.complete_group(episodes_per_dynamics, stop - group_frames)
            for name, values in [('episode_ends', np.array(lengths, dtype=np.int64)),
                                 ('split', np.full(len(lengths), int(episode_identity(seeds[0], 0, episode_sigmas)['split'] == 'validation'), dtype=np.int8)),
                                 ('noise_seed', np.array(noise_seeds, dtype=np.uint64)),
                                 ('success', np.array(successes, dtype=np.uint8)),
                                 ('final_coverage', np.array(final_coverages, dtype=np.float32)),
                                 ('max_coverage', np.array(max_coverages, dtype=np.float32)),
                                 ('n_contact_steps', np.array(contact_steps, dtype=np.int32)),
                                 ('block_displacement', np.array(displacements, dtype=np.float32))]:
                meta.create_dataset(name, data=values)
            root.attrs.update(recipe='dinowm_fresh_policy_iid_v1', collection_config=config,
                              action_space='absolute_pixel_target', action_clipping='base_policy_only_noise_unclipped',
                              noise_process='iid', clean_episodes_per_dynamics=clean_episodes_per_dynamics,
                              noise_sigmas_px=list(noise_sigmas_px), noise_repeats=noise_repeats,
                              episodes_per_dynamics=episodes_per_dynamics,
                              noise_seed_base=703000, reset_seed_base=603000, policy_kind='rl',
                              img_cog_recorded=False,
                              seed_formula='(cog_seed - 5001) * episodes_per_dynamics + episode_index')
            shards.append(dict(path=str(path.relative_to(output)), mask_path=None,
                               split=episode_identity(seeds[0], 0, episode_sigmas)['split'],
                               cog_seeds=seeds, episodes=len(lengths)))
        stats = {}
        for name in sums:
            mean = sums[name] / count if count else np.full(2, np.nan)
            variance = (squares[name] - count * mean**2) / (count - 1) if count > 1 else np.full(2, np.nan)
            stats[name] = dict(count=count, mean=mean.tolist(), std=np.sqrt(np.maximum(variance, 0)).tolist())
        manifest = dict(schema_version=1, recipe='dinowm_fresh_policy_iid_v1',
                        collection_config=config, episodes_per_dynamics=episodes_per_dynamics,
                        episodes=sum(s['episodes'] for s in shards), shards=shards,
                        train_episodes=sum(s['episodes'] for s in shards if s['split']=='train'),
                        validation_episodes=sum(s['episodes'] for s in shards if s['split']=='validation'),
                        action_stats_px=stats)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return manifest


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='pusht')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
