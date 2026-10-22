"""FetchSlide recorded controls, settled targets and retraction-tail alignment."""
import numpy as np


def fetch_target(collection, workflow, protocol, episode):
    _, _, begin, end = collection.episode_info(episode)
    meta = collection.root['meta']
    record = dict(source_episode=episode, dynamics_id=int(collection.dynamics[episode]),
                  episode_index_in_dynamics=int(meta['episode_index_in_dynamics'][episode]),
                  episode_length=end-begin)
    if workflow != 'planning':
        return dict(record, offset=0)
    goal = end-begin-1
    last = goal // protocol['frameskip'] * protocol['frameskip']
    current, final, wm = collection.root['data/state'].oindex[[begin, end-1, begin+last]]
    tail_error = float(np.linalg.norm(final[3:5]-wm[3:5]))
    if tail_error > protocol['tail_alignment_limit_m']:
        return None
    record.update(collection.episode_physics(episode))
    record.update(current_row=0, goal_row=goal, last_wm_row=last,
                  wm_horizon_tokens=last//protocol['frameskip'], tail_alignment_raw_steps=goal-last,
                  goal_puck_x=float(final[3]), goal_puck_y=float(final[4]),
                  current_state=current.tolist(), goal_state=final.tolist(),
                  tail_puck_xy_delta_m=tail_error, terminal_reason='settled',
                  gt_action_xy=meta['episode_action_xy'][episode].tolist())
    for key in ('terminal_step', 'retraction_start_step', 'retraction_step', 'park_position'):
        record[key] = np.asarray(meta[key][episode]).tolist()
    return record
