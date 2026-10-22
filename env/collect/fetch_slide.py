"""Paper FetchSlide three-physics XY collection with retraction."""

import numpy as np

from env.fetch_slide import FetchSlideEnv


PARAM_NAMES = ('mass', 'friction', 'fluid_density')


def sample_tables(episodes_per_dynamics=240, n_dynamics=150):
    seeds = np.random.SeedSequence(18).spawn(5)
    friction = np.random.default_rng(seeds[0]).uniform(.0305, .0435, n_dynamics)
    mass = np.random.default_rng(seeds[1]).uniform(.04, .36, n_dynamics)
    density = np.random.default_rng(seeds[2]).uniform(0, 40, n_dynamics)
    x = np.random.default_rng(seeds[3]).uniform(.35, .85, (n_dynamics, episodes_per_dynamics))
    y = np.random.default_rng(seeds[4]).uniform(-.8, .8, (n_dynamics, episodes_per_dynamics))
    return np.column_stack((mass, friction, density)).astype(np.float32), np.stack((x, y), axis=-1).astype(np.float32)


def normalize_physics(raw):
    return np.array([(raw[0] - .20) / .16, (raw[1] - .037) / .0065,
                     (raw[2] - 20) / 20], dtype=np.float32)


def episode_identity(physics, actions, dynamics_id, episode_index):
    return dict(dynamics_id=dynamics_id, episode_index_in_dynamics=episode_index,
                episode_seed=18, split=int(dynamics_id >= round(.9 * len(physics))),
                physics=dict(zip(PARAM_NAMES, map(float, physics[dynamics_id]))),
                action_xy=actions[dynamics_id, episode_index].tolist())


def collect_episode(identity):
    physics = identity['physics']
    env = FetchSlideEnv(physics, render=True)
    try:
        result = env.rollout(identity['action_xy'])
        images = env.render_qposes(result['qpos'])
    finally:
        env.close()
    length = len(images)
    raw = np.array([physics[name] for name in PARAM_NAMES], dtype=np.float32)
    normed = normalize_physics(raw)
    controls = result['action']
    zeros = np.zeros_like(controls)
    episode = dict(img=images, state=result['state'], action=controls,
                   executed_action=controls.copy(), policy_action=controls.copy(),
                   action_noise=zeros, raw_action_noise=zeros.copy(),
                   dynamics_id=np.full(length, identity['dynamics_id'], dtype=np.int64),
                   physics_params_raw=np.broadcast_to(raw, (length, 3)).copy(),
                   physics_params_normed=np.broadcast_to(normed, (length, 3)).copy())
    for name in PARAM_NAMES:
        episode[name] = np.full(length, physics[name], dtype=np.float32)
    return episode, result


def episode_masks(identity, result):
    env = FetchSlideEnv(identity['physics'], render=True)
    try:
        return env.render_qposes(result['qpos'], with_masks=True)[1]
    finally:
        env.close()


def collect(output, dynamics_ids=None, episode_indices=None, episodes_per_dynamics=240, n_dynamics=150):
    import zarr
    from env.collect.common import collection_indices, prepare_output, append_episode, save_run, write_json, CollectionLog
    dynamics_ids = collection_indices(dynamics_ids, n_dynamics, 'dynamics_ids')
    episode_indices = collection_indices(episode_indices, episodes_per_dynamics, 'episode_indices')
    path = prepare_output(output)
    with CollectionLog(path, 'fetch_slide', len(dynamics_ids),
                       len(dynamics_ids) * len(episode_indices)) as progress:
        config = dict(dynamics_ids=list(dynamics_ids), episode_indices=list(episode_indices),
                      episodes_per_dynamics=episodes_per_dynamics, n_dynamics=n_dynamics)
        save_run(path, config)
        physics, actions = sample_tables(episodes_per_dynamics, n_dynamics)
        root = zarr.open_group(str(path), mode='w-')
        mask_path = path.with_name(path.stem + '_mask.zarr')
        masks = zarr.open_group(str(mask_path), mode='w-')
        attrs = dict(schema_version='fetch_slide_wthe.friction_density_mass_xy.v2',
                     env_name='FetchSlide-v4', physics_order=list(PARAM_NAMES), master_seed=18,
                     episodes_per_dynamics=episodes_per_dynamics, n_dynamics=n_dynamics, complete=False,
                     physics_normalization=dict(order=list(PARAM_NAMES), center=[.20, .037, 20.],
                                                half_range=[.16, .0065, 20.]))
        root.attrs.update(attrs)
        masks.attrs.update(format_version='fetch_slide_wthe.segmentation_friction_density_mass_xy.v2',
                           labels={'0': 'background', '1': 'puck'})
        identities = []
        frames = 0
        terminal_codes = {'settled': 0, 'off_table': 1, 'no_hit': 2, 'watchdog': 3}
        for dynamics in dynamics_ids:
            progress.start_group(dynamics)
            group_frames = frames
            for index in episode_indices:
                progress.episode = index
                identity = episode_identity(physics, actions, dynamics, index)
                episode, result = collect_episode(identity)
                mask = episode_masks(identity, result)
                frames += len(episode['img'])
                metadata = dict(episode_ends=np.int64(frames), episode_dynamics_id=np.int64(dynamics),
                                episode_index_in_dynamics=np.int64(index), episode_seed=np.int64(18),
                                split=np.int8(identity['split']), episode_action_xy=np.asarray(identity['action_xy'], dtype=np.float32),
                                terminal_reason=np.int8(terminal_codes[result['terminal']]), terminal_step=np.int64(result['terminal_step']),
                                retraction_start_step=np.int64(result['retraction_start_step']),
                                retraction_step=np.int64(result['retraction_step']), park_position=result['park_position'])
                for name in PARAM_NAMES:
                    metadata['episode_' + name] = np.float32(identity['physics'][name])
                append_episode(root, episode, metadata)
                append_episode(masks, {'img_mask': mask}, metadata)
                identities.append(dict(identity, **identity['physics'],
                                       split='validation' if identity['split'] else 'train'))
            progress.complete_group(len(episode_indices), frames - group_frames)
        root.attrs.update(complete=True, completed_episodes=len(identities), completed_frames=frames)
        manifest = dict(format_version='fetch_slide_wthe.collection_manifest_friction_density_mass_xy.v2',
                        schema_version=attrs['schema_version'], attrs=dict(root.attrs),
                        output_rgb_zarr=str(path), output_mask_zarr=str(mask_path),
                        n_frames=frames, n_episodes=len(identities),
                        physics_params_by_dynamics=[dict(zip(PARAM_NAMES, map(float, row))) for row in physics],
                        episode_identities=identities)
        write_json(path.with_suffix('.manifest.json'), manifest)
        return manifest


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='fetch_slide')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
