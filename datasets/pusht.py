"""Read raw PushT episodes from a Zarr shard or an ordered shard manifest."""

from bisect import bisect_right
from itertools import accumulate
import json
from pathlib import Path

import numpy as np
import zarr
import torch

from datasets.common import (
    _episode_bounds, _read_frames, _read_window, preprocess_rgb, normalize,
    denormalize, window_starts, WindowDataset, window_dataloader as pusht_dataloader,
)


class PushTManifest:
    """Index all episodes in manifest order, before any train/validation split."""

    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path)
        self.metadata = json.loads(self.path.read_text())
        self.shards = self.metadata["shards"]
        self.action_stats = self.metadata["action_stats_px"]["absolute"]
        self._groups = {}
        self._episode_ends = {}
        counts = [shard["episodes"] for shard in self.shards]
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("Shard episode counts must be nonnegative integers")
        self.episode_stops = list(accumulate(counts))
        self.total = sum(counts)
        if self.total != self.metadata["episodes"]:
            raise ValueError("Manifest total differs from the sum of shard episode counts")

    def __len__(self):
        return self.total

    def locate(self, episode_id: int) -> tuple[Path, int]:
        """Global ID -> shard path and local ID; negative IDs are not supported."""
        if not 0 <= episode_id < self.total:
            raise IndexError(f"episode_id must be in [0, {self.total})")
        shard_id = bisect_right(self.episode_stops, episode_id)
        start = 0 if shard_id == 0 else self.episode_stops[shard_id - 1]
        return self.path.parent / self.shards[shard_id]["path"], episode_id - start

    def group(self, path: Path):
        """Lazily cache a read-only shard root; no image arrays are cached."""
        if path not in self._groups:
            self._groups[path] = zarr.open_group(str(path), mode="r")
        return self._groups[path]

    def episode_info(self, episode_id: int) -> tuple[Path, int, int, int]:
        """Global ID -> shard, local ID and half-open shard frame boundaries."""
        path, local_id = self.locate(episode_id)
        if path not in self._episode_ends:
            self._episode_ends[path] = self.group(path)["meta/episode_ends"][:]
        begin, end = _episode_bounds(self._episode_ends[path], local_id)
        return path, local_id, begin, end

    def shard_paths(self, split: str) -> list[Path]:
        """Resolve paths for one released split without opening any shards."""
        return [self.path.parent / shard["path"] for shard in self.shards if shard["split"] == split]

    def split_episode_ids(self) -> dict[str, list[int]]:
        """Global IDs in manifest order, using released shard split labels."""
        splits = {"train": [], "validation": []}
        start = 0
        for shard, stop in zip(self.shards, self.episode_stops):
            splits[shard["split"]].extend(range(start, stop))
            start = stop
        return splits

    def read_episode(self, episode_id: int) -> dict:
        """Full episode inspection using the same metadata and reader as Dataset."""
        path, local_id, begin, end = self.episode_info(episode_id)
        return {
            **_read_frames(self.group(path)["data"], begin, end),
            "episode_id": local_id, "global_episode_id": episode_id,
            "shard_path": path, "start": begin, "stop": end,
        }


def read_episode(shard_path: str | Path, episode_id: int) -> dict:
    """Inspect a known shard without a manifest, using the canonical reader.

    RGB/state precede the corresponding action; the final action's successor
    observation is not recorded. Raw state has five coordinates.
    """
    root = zarr.open_group(str(shard_path), mode="r")
    begin, end = _episode_bounds(root["meta/episode_ends"], episode_id)
    return {
        **_read_frames(root["data"], begin, end),
        "episode_id": episode_id, "start": begin, "stop": end,
    }


def pusht_normalization(manifest: PushTManifest) -> dict:
    """Paper Vanilla absolute-action statistics and fixed state/proprio constants.

    Actions use training-frame sample statistics already saved in the manifest.
    State/proprio constants are inherited from the bound legacy implementation,
    not fitted on fresh-IID or validation data. Last two state/proprio entries
    describe velocity; the paper's raw zero velocity is not normalized to zero.
    This function does not construct proprio inputs or read physical parameters.
    """
    action = manifest.action_stats
    # stats from original DINOWM: agent_x, agent_y, block_x, block_y, block_angle, velocity_x, velocity_y
    state_mean = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027, 2.54307914])
    state_std = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
    return {
        "action_mean": torch.tensor(action["mean"], dtype=torch.float32),
        "action_std": torch.tensor(action["std"], dtype=torch.float32).clamp_min(1e-6),
        "state_mean": state_mean,
        "state_std": state_std,
        "proprio_mean": state_mean[[0, 1, 5, 6]],
        "proprio_std": state_std[[0, 1, 5, 6]],
        "action_frame_count": action["count"],
    }


def read_split_actions(manifest: PushTManifest, split: str) -> torch.Tensor:
    """Read all raw absolute actions for inspection, without decoding RGB frames.

    Materializes only two action coordinates per frame; no windows are sampled.
    The returned rows follow shard/frame order and include final recorded actions.
    """
    actions = []
    for path in manifest.shard_paths(split):
        actions.append(torch.from_numpy(manifest.group(path)["data/action"][:]))
    return torch.cat(actions).float()


def action_statistics(actions: torch.Tensor) -> dict:
    """Diagnostic frame-weighted mean and sample std, computed in float64.

    Pass training actions to reproduce manifest statistics. Validation summaries
    are descriptive only and must never replace the normalization binding.
    """
    values = actions.double()
    return {"count": len(values), "mean": values.mean(0), "std": values.std(0, correction=1)}


def episode_window(
    episode: dict,
    start: int,
    num_frames: int,
    frameskip: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict:
    """Slice one raw episode and pack normalized absolute actions chronologically.

    For N observations and stride F, require N*F raw steps. Observation j is
    recorded before actions start+j*F through start+(j+1)*F-1. The last action
    group is retained even though its successor observation is outside this
    window. Packed coordinates are [x0, y0, x1, y1, ...], never averaged.
    RGB and five-dimensional state stay raw; preprocessing/proprio are separate.
    Frame indices are episode-local, and no slice crosses an episode boundary.
    """
    if start not in window_starts(len(episode["action"]), num_frames, frameskip):
        raise IndexError("Window start does not fit a complete episode window")
    data = {"img": episode["rgb"], "state": episode["state"], "action": episode["action"]}
    window = _read_window(data, start, num_frames, frameskip, action_mean, action_std)
    window["rgb"] = window["rgb"].copy()
    return window


def pusht_observation(rgb, raw_state, image_size, normalization):
    """Shared image/proprio preparation; raw state remains replay metadata."""
    velocity = torch.zeros_like(raw_state[:, :2])
    state = torch.cat((raw_state, velocity), dim=-1)
    proprio = torch.cat((raw_state[:, :2], velocity), dim=-1)
    observation = {
        "visual": preprocess_rgb(rgb, image_size),
        "proprio": normalize(proprio, normalization["proprio_mean"], normalization["proprio_std"]),
    }
    return observation, state


class PushTDataset(WindowDataset):
    """PushT windows retain normalized proprio and raw seven-coordinate states."""

    observation = staticmethod(pusht_observation)
