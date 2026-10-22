"""Read the bound FetchSlide three-physics XY collection and raw controls."""

from datasets.common import ZarrEpisodes as FetchSlideEpisodes

import numpy as np
import torch
from datasets.common import ConstantConditionDataset


def fetch_episode_ids(collection):
    """Keep the bound minimum length in both training and validation splits."""
    lengths = np.diff(np.r_[0, collection.episode_ends])
    return {name: [i for i in ids if lengths[i] >= 20]
            for name, ids in collection.split_episode_ids().items()}


def fetch_conditions(collection, episode_ids):
    """Two recorded 4D hit rows, flattened once per episode; no retraction rows."""
    starts = np.r_[0, collection.episode_ends[:-1]][episode_ids]
    rows = (starts[:, None] + np.arange(2)).reshape(-1)
    return torch.from_numpy(collection.root['data/action'].oindex[rows]).float().reshape(-1, 8)


def fetch_normalization(collection):
    conditions = fetch_conditions(collection, fetch_episode_ids(collection)['train'])
    return dict(action_mean=conditions.mean(0), action_std=conditions.std(0).clamp_min(1e-6),
                action_episode_count=len(conditions))


class FetchSlideDataset(ConstantConditionDataset):
    def raw_conditions(self):
        return fetch_conditions(self.manifest, self.episode_ids)
