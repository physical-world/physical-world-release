"""Shared episode indexing, bounded windows and RGB preparation."""

import json
from pathlib import Path
import zarr

from functools import cached_property
from bisect import bisect_right
from itertools import accumulate

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

def _episode_bounds(episode_ends, episode_id: int) -> tuple[int, int]:
    """Interpret Zarr's cumulative frame stops for either episode entry point."""
    if not 0 <= episode_id < len(episode_ends):
        raise IndexError(f"episode_id must be in [0, {len(episode_ends)})")
    begin = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    return begin, int(episode_ends[episode_id])


def _read_frames(data, start: int, stop: int, frameskip: int = 1, action_stop=None) -> dict:
    """Slice Zarr or in-memory arrays; keep every action between observations."""
    return {
        "rgb": data["img"][start:stop:frameskip],
        "state": data["state"][start:stop:frameskip],
        "action": data["action"][start:stop if action_stop is None else action_stop],
    }


def preprocess_rgb(rgb, image_size: int) -> torch.Tensor:
    """Raw uint8 THWC RGB -> float32 TCHW visual input for the bound visual models.

    Pass image_size from the checkpoint config (224 for paper Vanilla). Resize the
    shorter edge, center crop, then normalize with mean/std 0.5 per RGB channel.
    Tensor bilinear resize uses antialiasing, matching Pixi ada torchvision 0.18.
    The float conversion allocates storage; raw source pixels are never modified.
    """
    visual = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(0, 3, 1, 2) / 255.0 # T, H, W, C -> T, C, H, W
    visual = TF.resize(visual, image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    visual = TF.center_crop(visual, image_size)
    return TF.normalize(visual, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def normalize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply the supplied statistics to the last dimension without mutating inputs."""
    return (values - mean) / std


def denormalize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert normalization with the same statistics."""
    return values * std + mean


def window_starts(episode_length: int, num_frames: int, frameskip: int) -> range:
    """All legal Vanilla starts, including every raw-frame phase; no padding."""
    if num_frames <= 0 or frameskip <= 0:
        raise ValueError("num_frames and frameskip must be positive")
    return range(max(0, episode_length - num_frames * frameskip + 1))


def _read_window(data, start, num_frames, frameskip, action_mean, action_std, episode_stop=None, hold_action=True) -> dict:
    """Bounded temporal reads; optional terminal image/state holds and environment-specific control fill."""
    stop = start + num_frames * frameskip
    frames = np.arange(start, stop, frameskip)
    if episode_stop is None:
        raw = _read_frames(data, start, stop, frameskip)
        valid = np.ones(num_frames, dtype=bool)
    else:
        valid = frames < episode_stop
        sampled = np.minimum(frames, episode_stop - 1)
        raw = {"rgb": data["img"].oindex[sampled],
               "state": data["state"].oindex[sampled],
               "action": data["action"][start:min(stop, episode_stop)]}
        padding = stop - start - len(raw["action"])
        if padding:
            fill = raw["action"][-1:] if hold_action else np.zeros_like(raw["action"][-1:])
            raw["action"] = np.concatenate((raw["action"], np.repeat(fill, padding, axis=0)))
    raw_actions = torch.from_numpy(raw["action"]).float()
    return {
        "start": start, "stop": stop,
        "valid_frames": torch.from_numpy(valid),
        "observation_frames": torch.arange(start, stop, frameskip),
        "action_frames": torch.arange(start, stop).reshape(num_frames, frameskip),
        "rgb": raw["rgb"],
        "state": torch.from_numpy(raw["state"]).float(),
        "raw_actions": raw_actions.reshape(num_frames, frameskip, -1),
        "action": normalize(raw_actions, action_mean, action_std).reshape(num_frames, -1),
    }


def window_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    *, drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Batch windows with native collation and a local sampling RNG.

    Use shuffle=True for training and False for validation. Iterating the same
    training loader again advances its RNG; rebuilding with the same seed
    reproduces its initial order. This does not reproduce legacy permutations.
    Keep the partial final batch by default; callers may drop it for training. Outputs remain
    on CPU; device transfer belongs to the consuming training/inference code.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        drop_last=drop_last,
    )


class WindowDataset(torch.utils.data.Dataset):
    """Shared episode/start indexing and bounded windows for PushT and Reacher.

    Subclasses supply only observation preparation. Sampling and collation remain
    native DataLoader responsibilities; raw states stay replay metadata.
    """

    hold_action = True

    def __init__(self, manifest, episode_ids, num_frames: int,
                 frameskip: int, image_size: int, normalization: dict, pad_last_frame: bool = False):
        self.manifest = manifest
        self.pad_last_frame = pad_last_frame
        self.episode_ids = list(episode_ids)
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.image_size = image_size
        self.normalization = normalization
        self.episodes = []
        counts = []
        for episode_id in self.episode_ids:
            record = manifest.episode_info(episode_id)
            self.episodes.append(record)
            index_frames = 4 if pad_last_frame else num_frames
            counts.append(len(window_starts(record[3] - record[2], index_frames, frameskip)))
        self.window_stops = list(accumulate(counts))
        self.total = sum(counts)

    def __len__(self):
        return self.total

    def locate(self, index: int) -> tuple[int, int]:
        """Sample index -> selected-episode position and episode-local start."""
        if not 0 <= index < self.total:
            raise IndexError(f"sample index must be in [0, {self.total})")
        position = bisect_right(self.window_stops, index)
        offset = 0 if position == 0 else self.window_stops[position - 1]
        return position, index - offset

    def __getitem__(self, index: int):
        position, start = self.locate(index)
        path, _, begin, end = self.episodes[position]
        stats = self.normalization
        window = _read_window(
            self.manifest.group(path)["data"], begin + start, self.num_frames,
            self.frameskip, stats["action_mean"], stats["action_std"],
            episode_stop=end if self.pad_last_frame else None, hold_action=self.hold_action,
        )
        observation, state = self.observation(window["rgb"], window["state"], self.image_size, stats)
        if self.pad_last_frame:
            observation["valid_frames"] = window["valid_frames"]
        return observation, window["action"], state



class ZarrEpisodes:
    """Read a single flat Zarr with shared split and dynamics-table metadata."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".manifest.json").read_text())
        self.root = zarr.open_group(str(self.path), mode="r")
        meta = self.root["meta"]
        self.episode_ends = meta["episode_ends"][:]
        self.split = meta["split"][:]
        self.dynamics = meta["episode_dynamics_id"][:]
        self.seeds = meta["episode_seed"][:]

    def __len__(self):
        return len(self.episode_ends)

    def group(self, path):
        """Reuse the open read-only group for the shared bounded reader."""
        return self.root

    def episode_info(self, episode_id):
        begin, end = _episode_bounds(self.episode_ends, episode_id)
        return self.path, episode_id, begin, end

    def split_episode_ids(self):
        return {name: np.flatnonzero(self.split == label).tolist()
                for name, label in (("train", 0), ("validation", 1))}

    def read_episode(self, episode_id):
        path, _, begin, end = self.episode_info(episode_id)
        dynamics = int(self.dynamics[episode_id])
        return {
            **_read_frames(self.root["data"], begin, end),
            "episode_id": episode_id, "start": begin, "stop": end,
            "dynamics_id": dynamics, "seed": int(self.seeds[episode_id]),
            "physics": self.episode_physics(episode_id),
            "path": path,
        }


    def episode_physics(self, episode_id):
        return self.metadata['physics_params_by_dynamics'][int(self.dynamics[episode_id])]


class ConstantConditionDataset(WindowDataset):
    """Strided images and a constant episode condition, not packed raw controls."""

    @cached_property
    def conditions(self):
        raw = self.raw_conditions()
        return (raw - self.normalization['action_mean']) / self.normalization['action_std']

    def __getitem__(self, index):
        position, start = self.locate(index)
        path, _, begin, end = self.episodes[position]
        frames = np.arange(self.num_frames) * self.frameskip + begin + start
        valid = frames < end
        frames = np.minimum(frames, end - 1)
        data = self.manifest.group(path)['data']
        observation = {'visual': preprocess_rgb(data['img'].oindex[frames], self.image_size)}
        if self.pad_last_frame:
            observation['valid_frames'] = torch.from_numpy(valid)
        state = torch.from_numpy(data['state'].oindex[frames]).float()
        return observation, self.conditions[position].expand(self.num_frames, -1), state
