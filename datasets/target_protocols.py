"""Environment-specific eligibility and recorded target timelines."""
import numpy as np

from datasets.context import pusht_episode_values


def target_candidates(collection, environment, workflow, protocol, ids):
    if environment == 'pusht':
        conditions = pusht_episode_values(collection, ids, 'cog_seed')
    else:
        conditions = collection.dynamics[ids]
    eligible = None
    if environment == 'fetch_slide' and workflow == 'planning':
        settled = collection.metadata['attrs']['terminal_codes']['settled']
        eligible = collection.root['meta/terminal_reason'][:] == settled
    elif environment == 'beerpong' and workflow == 'planning':
        eligible = collection.root['meta/episode_final_in_cup'][:]
    for episode, condition in zip(ids, conditions):
        _, _, begin, end = collection.episode_info(episode)
        if eligible is not None and not eligible[episode]:
            continue
        if end - begin >= protocol['minimum_episode_length']:
            yield episode, int(condition)


def target_record(collection, environment, workflow, protocol, episode, rng):
    if environment == 'fetch_slide':
        from datasets.fetch_targets import fetch_target
        return fetch_target(collection, workflow, protocol, episode)
    if environment == 'beerpong':
        from datasets.beerpong_targets import beerpong_target
        return beerpong_target(collection, workflow, protocol, episode)
    if environment == 'reacher':
        from datasets.reacher_targets import reacher_target
        return reacher_target(collection, protocol, episode, rng)
    if environment != 'pusht':
        raise ValueError(environment)
    validation = collection.split_episode_ids()['validation']
    path, _, begin, end = collection.episode_info(episode)
    data = collection.group(path)['data']
    gap = protocol['goal_H'] * protocol['frameskip']
    max_offset = end - begin - gap - 1
    offset = int(rng.randint(max_offset + 1))
    if protocol['filter_nontrivial_targets']:
        states = data['state'][begin:end]
        starts, goals = states[:-gap], states[gap:]
        displacement = np.linalg.norm(goals[:, 2:4] - starts[:, 2:4], axis=1)
        angle = np.abs((goals[:, 4] - starts[:, 4] + np.pi) % (2 * np.pi) - np.pi)
        legal = np.flatnonzero((displacement >= protocol['nontrivial_min_block_displacement']) |
                              (angle >= protocol['nontrivial_min_block_angle_rad']))
        if not len(legal):
            return None
        if offset not in legal:
            offset = int(legal[rng.randint(len(legal))])
    return dict(source_episode=validation.index(episode), offset=offset,
                cog_seed=int(data['cog_seed'][begin]))
