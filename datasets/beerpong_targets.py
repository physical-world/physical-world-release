"""Aligned BeerPong outcomes and bounded CPU ground-truth replay eligibility."""
import numpy as np


def beerpong_target(collection, workflow, protocol, episode):
    meta = collection.root['meta']
    record = dict(source_episode=episode, dynamics_id=int(collection.dynamics[episode]),
                  episode_index_in_dynamics=int(meta['episode_index_in_dynamics'][episode]))
    if workflow != 'planning':
        return record
    from env.beerpong import BeerPongEnv
    _, _, begin, end = collection.episode_info(episode)
    goal = collection.root['data/state'][end-1]
    action = meta['episode_action'][episode]
    physics = collection.episode_physics(episode)
    env = BeerPongEnv(physics)
    try:
        replay = env.rollout(action)
    finally:
        env.close()
    error = float(np.linalg.norm(replay['qpos'][-1, 7:10]-goal[7:10]))
    if not replay['final_in_cup'] or error > protocol['gt_replay_endpoint_limit_m']:
        return None
    record.update(physics)
    record.update(gt_action=action.tolist(), current_row=0, goal_row=end-begin-1,
                  wm_horizon_tokens=(end-begin-1)//protocol['frameskip'],
                  goal_ball_xyz=goal[7:10].tolist(), gt_replay_endpoint_error_m=error)
    return record
