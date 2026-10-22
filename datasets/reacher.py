"""Read the paper's damping-only Reacher collection without loading all images."""

import numpy as np
import torch

from datasets.common import ZarrEpisodes as ReacherEpisodes, preprocess_rgb, WindowDataset



def reacher_normalization(collection):
    """Float32 sample statistics over training controls, matching the binding."""
    lengths = np.diff(np.r_[0, collection.episode_ends])
    training = np.repeat(collection.split == 0, lengths)
    actions = torch.from_numpy(collection.root["data/action"][:][training]).float()
    return {"action_mean": actions.mean(0), "action_std": actions.std(0).clamp_min(1e-6),
            "action_frame_count": len(actions)}


def reacher_observation(rgb, raw_state, image_size, normalization):
    """Visual-only observations; preserve qpos/qvel solely for replay."""
    return {"visual": preprocess_rgb(rgb, image_size)}, raw_state


class ReacherDataset(WindowDataset):
    """Reacher controls pack five consecutive 2D actions; state is not an input."""

    observation = staticmethod(reacher_observation)
    hold_action = False
