"""Reacher proposal conversion in the bound normalized control coordinates."""
import torch


def reacher_raw_actions(actions, normalization):
    mean = normalization['action_mean'].to(actions)
    std = normalization['action_std'].to(actions)
    return (actions.reshape(*actions.shape[:-2], -1, 2) * std + mean).clamp(-1, 1)


def clip_reacher_actions(actions, normalization):
    raw = reacher_raw_actions(actions, normalization)
    return ((raw - normalization['action_mean'].to(raw)) /
            normalization['action_std'].to(raw)).reshape_as(actions)


@torch.inference_mode()
def reacher_candidate_cost(model, current, goal, proposals, normalization, history_actions=None, chunk_size=16):
    from planning.candidates import rollout_candidates
    execution = clip_reacher_actions(proposals, normalization)
    return torch.cat([rollout_candidates(model, current, goal, chunk, 0, history_actions)[1]['total']
                      for chunk in execution.split(chunk_size)])


def reacher_success(goal_state, state):
    """Paper criterion: both unwrapped joint errors are strictly below 10 degrees."""
    import numpy as np
    return bool(np.all(np.abs(np.asarray(state, dtype=float)[:2] - np.asarray(goal_state, dtype=float)[:2]) < np.deg2rad(10)))


@torch.inference_mode()
def reacher_target_rounds(model, targets, normalization, cem_settings, generator, *, chunk_size=16):
    """Advance all target slots per MPC round, including inactive RNG draws."""
    import numpy as np
    from datasets.reacher import reacher_observation
    from planning.cem import cem_plan
    from planning.mpc import append_feedback
    device = next(model.parameters()).device
    contexts = []
    for target in targets:
        rgb, state = target['env'].reset(target['initial'])
        observed, _ = reacher_observation(rgb[None], torch.from_numpy(state[None]), 224, normalization)
        if target.get('physics_condition') is not None:
            from datasets.privileged_physics import physics_observation
            observed = physics_observation(observed, target['physics_condition'])
        if target.get('context') is not None:
            observed.update(target['context'])
        contexts.append(dict(target, rgb=rgb, state=state, history_actions=None,
                             current={key: value[None].to(device) for key, value in observed.items()}))
    active = [True] * len(contexts)
    for index in range(3):
        if not any(active):
            break
        scores = []
        for context in contexts:
            def score(proposals, context=context):
                return reacher_candidate_cost(model, context['current'], context['goal'], proposals,
                                              normalization, context['history_actions'], chunk_size)
            scores.append(score)
        selected = cem_plan(scores, horizon=5, action_dim=10, device=device,
                            generator=generator, active=active, **cem_settings)
        results = []
        for i, context in enumerate(contexts):
            if not active[i]:
                results.append(None)
                continue
            taken = clip_reacher_actions(selected[i:i+1], normalization)
            raw = reacher_raw_actions(taken, normalization)[0].cpu().numpy()
            images, states = [context['rgb']], [context['state']]
            for action in raw:
                rgb, state = context['env'].step(action)
                images.append(rgb)
                states.append(state)
            images, states = np.stack(images), np.stack(states)
            success = reacher_success(context['goal_state'], state)
            results.append(dict(round=index, current=context['current'], history_actions=context['history_actions'],
                                selected=selected[i:i+1], taken=taken, raw_actions=raw,
                                rgb=images, state=states, success=success))
            active[i] = not success
            feedback, _ = reacher_observation(images[5::5], torch.from_numpy(states[5::5]), 224, normalization)
            if context.get('physics_condition') is not None:
                feedback = physics_observation(feedback, context['physics_condition'])
            current, history = append_feedback(context['current'], context['history_actions'], feedback, taken, model.num_hist)
            context.update(current=current, history_actions=history, rgb=rgb, state=state)
        yield results


def reacher_mpc_rounds(model, env, initial, goal_state, goal, normalization, cem_settings, generator, physics_condition=None, context=None):
    """Single-target inspection view of the canonical round/iteration/target loop."""
    target = dict(env=env, initial=initial, goal_state=goal_state, goal=goal, physics_condition=physics_condition, context=context)
    for results in reacher_target_rounds(model, [target], normalization, cem_settings, generator):
        yield results[0]
