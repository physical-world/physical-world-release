"""PushT MPC: advance all targets through each CEM solve and execution round."""

import numpy as np
import torch

from datasets.pusht import pusht_observation
from planning.candidates import rollout_candidates
from planning.cem import cem_plan
from planning.metrics import pusht_state_metrics


@torch.inference_mode()
def mpc_target_rounds(model, targets, converter, normalization, image_size, alpha,
                      cem_settings, num_rounds, n_taken_actions, generator,
                      candidate_chunk_size=None):
    """Yield one ordered list of target results per round; stopped slots are None.

    Each target supplies env, initial, physical, goal and optional goal_state.
    A shared CPU generator advances iteration first, then target, including
    stopped slots. Simulators retain real state; only live targets execute.
    """
    device = next(model.parameters()).device
    contexts = []
    for target in targets:
        rgb, state = target['env'].reset(target['initial'], **target['physical'])
        observation, _ = pusht_observation(rgb[None], torch.from_numpy(state[None, :5]), image_size, normalization)
        if target.get('physics_condition') is not None:
            from datasets.privileged_physics import physics_observation
            observation = physics_observation(observation, target['physics_condition'])
        if target.get('context') is not None:
            observation.update(target['context'])
        contexts.append(dict(target, rgb=rgb, state=state, tail=None, history_actions=None,
                             current={key: value[None].to(device) for key, value in observation.items()}))
    active = [True] * len(contexts)
    for index in range(num_rounds):
        for i, context in enumerate(contexts):
            if context.get('goal_state') is not None:
                active[i] = active[i] and not pusht_state_metrics(context['goal_state'], context['state'])['joint_success']
        if not any(active):
            break
        scores, warms, anchors = [], [], []
        for context in contexts:
            anchor = torch.as_tensor(context['state'][None, :2], device=device)
            anchors.append(anchor)
            warms.append(None if context['tail'] is None else converter.execution_to_proposal(context['tail'], anchor)[0])
            def score(proposals, context=context, anchor=anchor):
                execution = converter.proposal_to_execution(proposals, anchor)
                chunks = execution.split(candidate_chunk_size or len(execution))
                return torch.cat([rollout_candidates(model, context['current'], context['goal'], chunk,
                                  alpha, context['history_actions'])[1]['total'] for chunk in chunks])
            scores.append(score)
        # Keep the single-target callable interface used by teaching inspections.
        selected = cem_plan(scores[0] if len(scores) == 1 else scores, **cem_settings,
                            device=device, generator=generator, active=active,
                            initial_mean=warms[0] if len(scores) == 1 else warms)
        results = []
        for i, context in enumerate(contexts):
            if not active[i]:
                results.append(None)
                continue
            planned = converter.proposal_to_execution(selected[i:i+1], anchors[i])
            taken = planned[:, :n_taken_actions]
            images, states = [context['rgb']], [context['state']]
            for action in converter.execution_to_absolute(taken)[0].cpu().numpy():
                rgb, state = context['env'].step(action)
                images.append(rgb)
                states.append(state)
            images, states = np.stack(images), np.stack(states)
            results.append(dict(round=index, current=context['current'], history_actions=context['history_actions'],
                                anchor=anchors[i], warm=warms[i], selected=selected[i:i+1], planned=planned,
                                taken=taken, rgb=images, state=states))
            tail = planned[:, taken.shape[1]:]
            context.update(rgb=images[-1], state=states[-1], tail=tail if tail.shape[1] else None)
            feedback, _ = pusht_observation(images[converter.frameskip::converter.frameskip],
                                           torch.from_numpy(states[converter.frameskip::converter.frameskip, :5]),
                                           image_size, normalization)
            if context.get('physics_condition') is not None:
                feedback = physics_observation(feedback, context['physics_condition'])
            context['current'], context['history_actions'] = append_feedback(
                context['current'], context['history_actions'], feedback, taken, model.num_hist)
        yield results


def mpc_rounds(model, env, initial, physical, goal, converter, normalization,
               image_size, alpha, cem_settings, num_rounds, n_taken_actions, generator,
               goal_state=None, candidate_chunk_size=None, physics_condition=None, context=None):
    """Single-target view; all computation lives in mpc_target_rounds."""
    target = dict(env=env, initial=initial, physical=physical, goal=goal, goal_state=goal_state, physics_condition=physics_condition, context=context)
    for results in mpc_target_rounds(model, [target], converter, normalization, image_size,
                                   alpha, cem_settings, num_rounds, n_taken_actions, generator, candidate_chunk_size):
        yield results[0]


def append_feedback(current, history_actions, feedback, taken, num_hist):
    """Append real observations and keep exactly their H-1 connecting actions."""
    device = current['visual'].device
    conditions = {key:value for key,value in current.items() if key.startswith('ctx_')}
    current = {key: torch.cat((current[key], value[None].to(device)), dim=1)[:, -num_hist:]
               for key, value in feedback.items()}
    current.update(conditions)
    complete = taken if history_actions is None else torch.cat((history_actions, taken), dim=1)
    needed = current['visual'].shape[1] - 1
    return current, complete[:, -needed:] if needed else None
