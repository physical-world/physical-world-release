"""Read the paper-bound aligned-v2 BeerPong collection."""
from datasets.common import ZarrEpisodes


class BeerPongEpisodes(ZarrEpisodes):
    def episode_physics(self, episode_id):
        return {name: float(self.root[f'meta/episode_{name}'][episode_id])
                for name in ('contact_damping', 'cup_body_y')}


import torch
from datasets.common import ConstantConditionDataset


def beerpong_conditions(collection, episode_ids):
    return torch.from_numpy(collection.root['meta/episode_action'].oindex[episode_ids]).float()


def beerpong_normalization(collection):
    actions = beerpong_conditions(collection, collection.split_episode_ids()['train'])
    return dict(action_mean=actions.mean(0), action_std=actions.std(0).clamp_min(1e-6),
                action_episode_count=len(actions))


class BeerPongDataset(ConstantConditionDataset):
    def raw_conditions(self):
        return beerpong_conditions(self.manifest, self.episode_ids)
