"""Paper damping-only Reacher collection with IID uniform controls."""

from itertools import groupby

import numpy as np

from env.reacher import ReacherEnv


PARAM_NAMES = ('link1_len', 'link1_mass', 'joint1_damping',
               'link2_len', 'link2_mass', 'joint2_damping')
DEFAULTS = (.12, .04188790204786391, .01, .10, .03560471674068432, .01)


def sample_table(n_dynamics=150, seed=3073):
    rng = np.random.default_rng(seed)
    unit = np.empty((n_dynamics, 2))
    for column in range(2):
        unit[:, column] = (rng.permutation(n_dynamics) + rng.random(n_dynamics)) / n_dynamics
    unit = unit[rng.permutation(n_dynamics)]
    values = np.tile(DEFAULTS, (n_dynamics, 1))
    values[:, [2, 5]] = .005 + unit * .015
    params = [dict(zip(PARAM_NAMES, map(float, row))) for row in values]
    raw = values.astype(np.float32)
    mean = raw.astype(np.float64).mean(0)
    std = raw.astype(np.float64).std(0)
    safe_std = np.where(std < 1e-8, 1, std)
    normed = ((raw.astype(np.float64) - mean) / safe_std).astype(np.float32)
    return dict(params=params, raw=raw, normed=normed,
                mean=mean.astype(np.float32), std=safe_std.astype(np.float32))


def episode_identity(dynamics_id, episode_index, seed=3073, n_dynamics=150):
    episode_seed = seed * 1_000_003 + dynamics_id * 10_007 + episode_index * 97
    return dict(dynamics_id=dynamics_id, episode_index_in_dynamics=episode_index,
                episode_seed=episode_seed, policy_seed=episode_seed + 17,
                split=int(dynamics_id >= round(.9 * n_dynamics)))


def episode_inputs(identity, length=100):
    rng = np.random.default_rng(identity['episode_seed'])
    rng.integers(0, 2**31 - 1)
    qpos = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.deg2rad(160), np.deg2rad(160))], dtype=np.float32)
    actions = np.random.default_rng(identity['policy_seed']).uniform(-1, 1, (length, 2)).astype(np.float32)
    return np.r_[qpos, np.zeros(2, dtype=np.float32)], actions


def collect_episode(table, identity, length=100, image_size=224):
    dynamics = identity['dynamics_id']
    env = ReacherEnv(table['params'][dynamics], image_size)
    initial, actions = episode_inputs(identity, length)
    rgb, state = env.reset(initial)
    images, states = [], []
    for action in actions:
        images.append(rgb)
        states.append(state)
        rgb, state = env.step(action)
    states = np.stack(states)
    zeros = np.zeros_like(actions)
    return dict(img=np.stack(images), state=states, velocity=states[:, 2:].copy(),
                action=actions, executed_action=actions.copy(), policy_action=actions.copy(),
                raw_action_noise=zeros, action_noise=zeros.copy(),
                dynamics_id=np.full(length, dynamics, dtype=np.int64),
                physics_params_raw=np.broadcast_to(table['raw'][dynamics], (length, 6)).copy(),
                physics_params_normed=np.broadcast_to(table['normed'][dynamics], (length, 6)).copy(),
                noise_sigma=np.zeros(length, dtype=np.float32), noise_rho=np.zeros(length, dtype=np.float32),
                policy_subgoal=np.full((length, 2), np.nan, dtype=np.float32),
                policy_subgoal_resampled=np.zeros(length, dtype=bool))


def episode_masks(table, identity, episode, image_size=224):
    import mujoco
    env = ReacherEnv(table['params'][identity['dynamics_id']], image_size)
    masks = []
    for state in episode['state']:
        env.reset(state)
        segmentation = env.physics.render(height=image_size, width=image_size,
                                          camera_id=0, segmentation=True)
        ids, types = segmentation[..., 0], segmentation[..., 1]
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        for name, label in [('root', 1), ('arm', 1), ('hand', 2), ('finger', 2)]:
            geom = env.physics.model.name2id(name, 'geom')
            mask[(types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids == geom)] = label
        masks.append(mask)
    return np.stack(masks)


def collect(output, dynamics_ids=None, episode_indices=range(20),
            length=100, image_size=224, n_dynamics=150, seed=3073):
    from env.collect.common import collection_indices
    dynamics_ids = collection_indices(dynamics_ids, n_dynamics, 'dynamics_ids')
    config = dict(dynamics_ids=list(dynamics_ids), episode_indices=list(episode_indices),
                  length=length, image_size=image_size, n_dynamics=n_dynamics, seed=seed)
    table = sample_table(n_dynamics, seed)
    identities = [episode_identity(dynamics, index, seed, n_dynamics)
                  for dynamics in config['dynamics_ids'] for index in config['episode_indices']]
    return collect_table(output, table, identities, config)


def collect_table(output, table, identities, config, *, sampling=None):
    import zarr
    from pathlib import Path
    from env.collect.common import prepare_output, append_episode, save_run, write_json, CollectionLog
    sampling_path = Path(output).with_suffix('.sampling.json')
    if sampling is not None and (sampling_path.exists() or sampling_path.is_symlink()):
        raise FileExistsError(sampling_path)
    path = prepare_output(output)
    groups = [(dynamics, list(rows)) for dynamics, rows in
              groupby(identities, key=lambda row: row['dynamics_id'])]
    env = 'reacher_extrapolation' if sampling is not None else 'reacher'
    with CollectionLog(path, env, len(groups), sum(len(rows) for _, rows in groups)) as progress:
        save_run(path, config)
        length, image_size = config['length'], config['image_size']
        root = zarr.open_group(str(path), mode='w-')
        mask_path = path.with_name(path.stem + '_mask.zarr')
        masks = zarr.open_group(str(mask_path), mode='w-')
        attrs = dict(schema_version='reacher_wthe.v4', env_name='reacher', policy='lewm_random',
                     frameskip=2, ep_len=length, img_size=image_size, seed=config['seed'],
                     param_names=list(PARAM_NAMES), physics_param_mean=table['mean'].tolist(),
                     physics_param_std=table['std'].tolist(), noise=dict(mode='none', rho=0.0, sigmas=[0.0]))
        if sampling is not None:
            attrs['sampling_tables'] = sampling
            write_json(sampling_path, sampling)
        root.attrs.update(attrs)
        masks.attrs.update(format_version='reacher_wthe.segmentation.v1',
                           labels={'0': 'background', '1': 'root+arm', '2': 'hand+finger'})
        recorded = []
        frames = 0
        for dynamics, rows in groups:
            progress.start_group(dynamics)
            group_frames = frames
            for identity in rows:
                progress.episode = identity['episode_index_in_dynamics']
                episode = collect_episode(table, identity, length, image_size)
                mask = episode_masks(table, identity, episode, image_size)
                frames += length
                metadata = dict(episode_ends=np.int64(frames), episode_dynamics_id=np.int64(identity['dynamics_id']),
                                episode_index_in_dynamics=np.int64(identity['episode_index_in_dynamics']),
                                episode_seed=np.int64(identity['episode_seed']), split=np.int8(identity['split']),
                                episode_noise_sigma=np.float32(0), episode_noise_rho=np.float32(0),
                                episode_gain_scale=np.float32(np.nan), episode_kp=np.float32(np.nan), episode_kd=np.float32(np.nan))
                append_episode(root, episode, metadata)
                append_episode(masks, {'img_mask': mask}, metadata)
                recorded.append(dict(identity, split='validation' if identity['split'] else 'train'))
            progress.complete_group(len(rows), frames - group_frames)
        manifest = dict(format_version='reacher_wthe.collection_manifest.v1', schema_version=attrs['schema_version'],
                        attrs=attrs, zarr_path=str(path), mask_zarr_path=str(mask_path), n_frames=frames,
                        n_episodes=len(recorded), physics_params_by_dynamics=table['params'], episode_identities=recorded)
        write_json(path.with_suffix('.manifest.json'), manifest)
        return manifest


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='reacher')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
