"""Recorded Reacher start/goal identities, with one seed-local offset per episode."""


def reacher_target(collection, protocol, episode, rng):
    _, _, begin, end = collection.episode_info(episode)
    gap = protocol['goal_H'] * protocol['frameskip']
    offset = int(rng.randint(end - begin - gap))
    rows = collection.root['data/state'].oindex[[begin + offset, begin + offset + gap]]
    return dict(source_episode=episode,
                validation_episode=collection.split_episode_ids()['validation'].index(episode),
                dynamics_id=int(collection.dynamics[episode]), offset=offset,
                episode_seed=int(collection.seeds[episode]), current_row=offset, goal_row=offset + gap,
                history_rows=[offset], current_state=rows[0].tolist(), goal_state=rows[1].tolist())
