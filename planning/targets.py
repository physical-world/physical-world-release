"""Recorded start/goal reads for generated and bound PushT planning targets."""

import json
from pathlib import Path

import torch

from datasets.pusht import _read_frames, pusht_observation
from evaluation.common import evaluation_episode_ids


def read_planning_target(manifest, target_path, target_index, image_size, normalization):
    """Load one bound single-history target without simulation or model inference.

    source_episode is validation-local; offset is an episode-local raw frame.
    Return unbatched CPU observations with a singleton time dimension. These are
    recorded observations, not the simulator-replayed obs_0/obs_g used by paper
    planning. Raw absolute controls are replay metadata, never current inputs.
    The target manifest's hash is provenance, not a verified replay-state hash.
    The supplied data manifest relocates the original eval_data_path explicitly.
    """
    targets = json.loads(Path(target_path).read_text())
    if not 0 <= target_index < len(targets["targets"]):
        raise IndexError("Planning target index is outside the target manifest")
    identity = targets["targets"][target_index]
    episode_id = evaluation_episode_ids(manifest, targets)[target_index]
    path, local_id, begin, end = manifest.episode_info(episode_id)
    start = identity["offset"]
    geometry = targets.get('protocol', targets)
    gap = geometry["goal_H"] * geometry["frameskip"]
    goal = start + gap
    if gap <= 0 or start < 0 or goal >= end - begin:
        raise IndexError("Planning start/goal exceeds the recorded episode")
    raw = _read_frames(manifest.group(path)["data"], begin + start,
                       begin + goal + 1, gap, action_stop=begin + goal)
    observation, state = pusht_observation(
        raw["rgb"], torch.from_numpy(raw["state"]).float(), image_size, normalization,
    )
    return {
        "target_index": target_index, "identity": dict(identity),
        "target_set_hash": targets["target_set_hash"],
        "episode_id": episode_id, "shard_path": path, "local_episode_id": local_id,
        "episode_start": begin, "episode_stop": end,
        "observation_frames": torch.tensor([start, goal]),
        "shard_frames": torch.tensor([begin + start, begin + goal]),
        "action_frames": torch.arange(start, goal),
        "rgb": raw["rgb"], "state": state,
        "recorded_current": {key: value[:1] for key, value in observation.items()},
        "recorded_goal": {key: value[1:] for key, value in observation.items()},
        "replay_absolute_actions": torch.from_numpy(raw["action"]).float(),
    }


def read_simulator_parameters(manifest, target):
    """Read only the initial frame's physical metadata, outside model inputs."""
    data = manifest.group(target["shard_path"])["data"]
    frame = int(target["shard_frames"][0])
    seed = int(data["cog_seed"][frame])
    if seed != target["identity"]["cog_seed"]:
        raise ValueError("Target identity disagrees with recorded CoG seed")
    return {"cog_seed": seed, "cog_local_normed": data["cog_local_normed"][frame]}
