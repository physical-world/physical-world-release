"""Environment-specific physical targets and training-owned statistics."""
import numpy as np
import torch


def reacher_physics(collection):
    names = ['joint1_damping','joint2_damping']
    table = torch.tensor([[row[name] for name in names]
                          for row in collection.metadata['physics_params_by_dynamics']],dtype=torch.float32)
    raw = table[torch.from_numpy(collection.dynamics)]
    lengths = np.diff(np.r_[0,collection.episode_ends])
    mean,std = _frame_statistics(raw,lengths,collection.split==0)
    return raw,mean,std


def _frame_statistics(raw,lengths,training):
    values = torch.repeat_interleave(raw[training],torch.from_numpy(lengths[training]),dim=0)
    return values.mean(0),values.std(0).clamp_min(1e-6)


def fetch_slide_physics(collection):
    starts = np.r_[0,collection.episode_ends[:-1]]
    raw = torch.from_numpy(np.column_stack([collection.root['data'][key].oindex[starts]
                                           for key in ('mass','friction','fluid_density')])).float()
    lengths = np.diff(np.r_[0,collection.episode_ends])
    mean,std = _frame_statistics(raw,lengths,(collection.split==0)&(lengths>=20))
    return raw,mean,std


def beerpong_physics(collection):
    raw = torch.from_numpy(collection.root['meta/episode_contact_damping'][:]).float().reshape(-1,1)
    training = raw[collection.split==0]
    return raw,training.mean(0),training.std(0).clamp_min(1e-6)
