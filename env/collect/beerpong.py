"""Paper aligned-v2 BeerPong collection with per-episode cup locations."""

from functools import lru_cache

import numpy as np

from env.beerpong import BeerPongEnv, render_qpos


def sample_tables(episodes_per_dynamics=60, n_dynamics=150):
    damping_seed, action_seed, cup_seed = np.random.SeedSequence(18).spawn(3)
    damping = np.random.default_rng(damping_seed).uniform(10, 40, n_dynamics).astype(np.float32)
    actions = np.random.default_rng(action_seed).uniform([2.5, 6], [3, 8],
                                                       (n_dynamics, episodes_per_dynamics, 2)).astype(np.float32)
    cups = np.random.default_rng(cup_seed).uniform(-2.58984375, -2.01796875,
                                                  (n_dynamics, episodes_per_dynamics)).astype(np.float32)
    return damping, actions, cups


@lru_cache(maxsize=1)
def cup_goal_offset():
    env = BeerPongEnv(dict(cup_body_y=0.0, contact_damping=25.0))
    return float(env.data.site('cup_goal_table').xpos[1])


def episode_identity(damping, actions, cups, dynamics_id, episode_index):
    goal = cups[dynamics_id, episode_index]
    return dict(dynamics_id=dynamics_id, episode_index_in_dynamics=episode_index,
                episode_seed=18, split=int(dynamics_id >= round(.9 * len(damping))),
                contact_damping=float(damping[dynamics_id]), action=actions[dynamics_id, episode_index].tolist(),
                cup_goal_y=float(goal), cup_body_y=float(goal - cup_goal_offset()))


def collect_episode(identity):
    env = BeerPongEnv(identity)
    try:
        result = env.rollout(identity['action'])
        images = env.render_qposes(render_qpos(result['qpos']))
    finally:
        env.close()
    drift = result['max_locked_joint_drift_rad']
    if drift > .005:
        raise ValueError(f'Locked BeerPong joints drifted by {drift}')
    length = len(images)
    controls = np.zeros((length, 2), dtype=np.float32)
    controls[:5] = identity['action']
    damping = identity['contact_damping']
    episode = dict(img=images, state=np.concatenate((result['qpos'], result['qvel']), axis=1).astype(np.float32),
                   action=controls, dynamics_id=np.full(length, identity['dynamics_id'], dtype=np.int64),
                   physics_params_raw=np.full((length, 1), damping, dtype=np.float32),
                   physics_params_normed=np.full((length, 1), (damping - 25) / 15, dtype=np.float32))
    return episode, result


def episode_masks(identity, result):
    env = BeerPongEnv(identity)
    try:
        return env.render_qposes(render_qpos(result['qpos']), with_masks=True)[1]
    finally:
        env.close()


def collect(output, dynamics_ids=None, episode_indices=None, episodes_per_dynamics=60, n_dynamics=150):
    import zarr
    from env.collect.common import collection_indices, prepare_output, append_episode, save_run, write_json, CollectionLog
    dynamics_ids = collection_indices(dynamics_ids, n_dynamics, 'dynamics_ids')
    episode_indices = collection_indices(episode_indices, episodes_per_dynamics, 'episode_indices')
    path = prepare_output(output)
    with CollectionLog(path, 'beerpong', len(dynamics_ids),
                       len(dynamics_ids) * len(episode_indices)) as progress:
        config = dict(dynamics_ids=list(dynamics_ids), episode_indices=list(episode_indices),
                      episodes_per_dynamics=episodes_per_dynamics, n_dynamics=n_dynamics)
        save_run(path, config)
        damping, actions, cups = sample_tables(episodes_per_dynamics, n_dynamics)
        root = zarr.open_group(str(path), mode='w-')
        mask_path = path.with_name(path.stem + '_mask.zarr')
        masks = zarr.open_group(str(mask_path), mode='w-')
        attrs = dict(schema_version='beerpong_wthe.damping.v2', geometry_version='beerpong.cup_geometry.v2',
                     master_seed=18, complete=False, episodes_per_dynamics=episodes_per_dynamics,
                     n_dynamics=n_dynamics,
                     physics_param_names=['contact_damping'], physics_param_center=[25.], physics_param_half_range=[15.])
        root.attrs.update(attrs)
        masks.attrs.update(format_version='beerpong_wthe.segmentation.v2', labels={'0': 'background', '1': 'ball'})
        identities = []
        frames = 0
        directions = {None: 0, '-x': 1, '+x': 2, '-y': 3, '+y': 4, 'ground_within_xy': 5}
        for dynamics in dynamics_ids:
            progress.start_group(dynamics)
            group_frames = frames
            for index in episode_indices:
                progress.episode = index
                identity = episode_identity(damping, actions, cups, dynamics, index)
                episode, result = collect_episode(identity)
                mask = episode_masks(identity, result)
                frames += len(episode['img'])
                metadata = dict(episode_ends=np.int64(frames), episode_dynamics_id=np.int64(dynamics),
                                episode_index_in_dynamics=np.int64(index), episode_seed=np.int64(identity['episode_seed']),
                                split=np.int8(identity['split']), episode_action=np.asarray(identity['action'], dtype=np.float32),
                                episode_in_cup=bool(result['ever_in_cup']), episode_final_in_cup=bool(result['final_in_cup']),
                                episode_off_table=bool(result['off_table']), off_table_direction=np.int8(directions[result['off_table_direction']]),
                                table_bounces_before_cup=np.int8(result['table_bounces_before_cup']))
                for name in ['contact_damping', 'cup_body_y', 'cup_goal_y']:
                    metadata['episode_' + name] = np.float32(identity[name])
                for event in ['in_cup', 'off_table']:
                    metadata[f'first_{event}_control_step'] = np.int64(result[f'first_{event}_control_step'])
                    metadata[f'first_{event}_physics_substep'] = np.int8(result[f'first_{event}_physics_substep'])
                append_episode(root, episode, metadata)
                append_episode(masks, {'img_mask': mask}, metadata)
                identities.append(dict(identity, split='validation' if identity['split'] else 'train'))
            progress.complete_group(len(episode_indices), frames - group_frames)
        for store in [root, masks]:
            store.attrs.update(complete=True, completed_episodes=len(identities), completed_frames=frames)
        manifest = dict(format_version='beerpong_wthe.collection_manifest.v2', schema_version=attrs['schema_version'],
                        attrs=dict(root.attrs), output_rgb_zarr=str(path), output_mask_zarr=str(mask_path),
                        n_frames=frames, n_episodes=len(identities),
                        physics_params_by_dynamics=[dict(contact_damping=float(value)) for value in damping],
                        episode_identities=identities)
        write_json(path.with_suffix('.manifest.json'), manifest)
        return manifest


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='beerpong')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
