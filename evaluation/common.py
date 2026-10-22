"""Sampling timelines for the bound PushT non-planning protocols."""

import random

import numpy as np

import torch

from datasets.pusht import _read_frames, normalize, pusht_observation, window_starts


def evaluation_episode_ids(manifest, targets):
    """Map bound target source IDs (validation-local) to global manifest IDs.

    Preserve target order and multiplicity. Planning offsets do not define the
    non-planning windows in the bound paired_history=False evaluation.
    """
    validation = manifest.split_episode_ids()["validation"]
    return [validation[target["source_episode"]] for target in targets["targets"]]


def evaluation_samples(manifest, episode_ids, protocol, *, frameskip=5, num_hist=3,
                       fixed_horizon=5, windows_per_episode=2, seed=42,
                       fixed_start=0, init_context=3, max_horizon=None, adaptive_history=False):
    """Yield index-only records; read images only when a sample is requested.

    Teacher forcing uses all legal H+1-frame windows, including every raw phase.
    Adaptive history retains Vanilla targets and extends their preceding context.
    Short episodes use a real variable prefix and repeat only terminal actions.
    Fixed-horizon samples use one initial frame and H future transitions.
    Open-loop uses phase zero through the last recorded strided observation.
    """
    if protocol not in ("teacher_forced", "fixed_horizon", "open_loop"):
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    if min(frameskip, num_hist, fixed_horizon, init_context) < 1:
        raise ValueError("Temporal sizes must be positive")
    if windows_per_episode is not None and windows_per_episode < 1:
        raise ValueError("windows_per_episode must be positive or None")
    rng = random.Random(seed)
    for episode_id in episode_ids:
        _, _, begin, end = manifest.episode_info(episode_id)
        length = end - begin
        variable_prefix = False
        if protocol == "teacher_forced":
            starts = window_starts(length, 4 if adaptive_history else num_hist + 1, frameskip)
            count = num_hist + 1
            variable_prefix = not starts
            if not starts:
                starts = [0]
                count = (length - 1) // frameskip + 1
            initial = count - 1
        elif protocol == "fixed_horizon":
            starts = list(range(max(length - fixed_horizon * frameskip, 0)))
            if windows_per_episode is None:
                starts = [fixed_start]
            elif len(starts) > windows_per_episode:
                starts = sorted(rng.sample(starts, windows_per_episode))
            count, initial = fixed_horizon + 1, 1
        else:
            starts = [0]
            count = (length - 1) // frameskip + 1
            if max_horizon is not None:
                count = min(count, max_horizon + 1)
            initial = init_context
        if not starts or count <= initial or initial < 1:
            raise ValueError(f"Episode {episode_id} has no real {protocol} target")
        for start in starts:
            frames = torch.arange(4 if protocol == "teacher_forced" and adaptive_history else count) * frameskip + start
            if start < 0 or frames[-1] >= length:
                raise IndexError(f"Evaluation window exceeds episode {episode_id}")
            groups = count if protocol == "teacher_forced" else count - 1
            action_frames = torch.arange(groups * frameskip).reshape(groups, frameskip) + start
            target_indices = torch.arange(1 if protocol == "teacher_forced" else initial, count)
            if protocol == "teacher_forced" and adaptive_history:
                target_frame = start + 3 * frameskip
                initial = min(num_hist, start // frameskip + 3)
                real = torch.arange(target_frame - initial * frameskip, target_frame + 1, frameskip)
                frames = torch.cat((real, real[-1:].expand(num_hist - initial)))
                action_frames = torch.arange((num_hist + 1) * frameskip).reshape(num_hist + 1, frameskip) + real[0]
                target_indices = torch.arange(initial - 2, initial + 1)
            yield {
                "baseline_start": start,
                "variable_prefix": variable_prefix,
                "adaptive_history": protocol == "teacher_forced" and adaptive_history,
                "protocol": protocol, "episode_id": episode_id,
                "observation_frames": frames, "action_frames": action_frames,
                "action_valid": action_frames < (int(frames[-1]) if protocol == "teacher_forced" and adaptive_history else length),
                "initial_indices": torch.arange(initial), "target_indices": target_indices,
                "target_frames": frames[target_indices],
            }


def read_evaluation_sample(manifest, sample, image_size, normalization, observation_fn=pusht_observation, hold_action=True):
    """Return unbatched CPU observations/actions/state and an explicit timeline.

    Read only strided RGB/state and required dense controls. Terminal action
    padding is used only by short teacher-forced prefixes, never scored targets.
    The final teacher-forced action group is encoded but not a predictor input.
    """
    path, _, begin, end = manifest.episode_info(sample["episode_id"])
    frames, action_frames = sample["observation_frames"], sample["action_frames"]
    stride = int(frames[1] - frames[0])
    data = manifest.group(path)["data"]
    # Separate stops: rollout's last observation has no outgoing action group.
    action_limit = begin + int(frames[-1]) if sample.get("adaptive_history", False) else end
    raw = _read_frames(
        data, begin + int(frames[0]), begin + int(frames[-1]) + 1, stride,
        action_stop=min(begin + int(action_frames[-1, -1]) + 1, action_limit),
    )
    actions = torch.from_numpy(raw["action"]).float()
    missing = action_frames.numel() - len(actions)
    if missing:
        fill = actions[-1:] if hold_action else torch.zeros_like(actions[-1:])
        actions = torch.cat((actions, fill.expand(missing, -1)))
    real_count = len(raw["rgb"])
    padding = len(frames) - real_count
    if padding:
        for key in ("rgb", "state"):
            raw[key] = np.concatenate((raw[key], np.repeat(raw[key][-1:], padding, axis=0)))
    observation, state = observation_fn(
        raw["rgb"], torch.from_numpy(raw["state"]).float(), image_size, normalization,
    )
    if sample.get("adaptive_history", False):
        observation["valid_frames"] = torch.arange(len(frames)) < real_count
    packed = normalize(actions, normalization["action_mean"], normalization["action_std"])
    return {**sample, "observation": observation, "action": packed.reshape(len(action_frames), -1),
            "state": state}
