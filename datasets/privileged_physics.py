"""Attach known normalized physics to the bound observation input."""
import torch
from torch.utils.data import Dataset


def physics_observation(observation,physics):
    """Append physical coordinates after base proprio, or use physics alone."""
    condition = physics.to(observation['visual']).expand(*observation['visual'].shape[:-3],-1)
    proprio = torch.cat((observation['proprio'],condition),dim=-1) if 'proprio' in observation else condition
    return {**observation,'proprio':proprio}


class PrivilegedPhysicsDataset(Dataset):
    """Reuse the existing windows and align one physical label with each episode."""
    def __init__(self,windows,physics):
        self.windows = windows
        self.physics = physics

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        position,_ = self.windows.locate(index)
        observation,action,state = self.windows[index]
        return physics_observation(observation,self.physics[position]),action,state


def privileged_physics(collection,environment):
    """Return episode-aligned labels and their training-owned normalization."""
    if environment == 'pusht':
        from datasets.context import pusht_physics_targets
        values = pusht_physics_targets(collection,list(range(len(collection))))
        mean,std = torch.zeros(2),torch.ones(2)
    elif environment == 'fetch_slide':
        from datasets.physics import fetch_slide_physics
        raw,mean,std = fetch_slide_physics(collection)
        values = (raw-mean)/std
    elif environment == 'beerpong':
        from datasets.physics import beerpong_physics
        raw,mean,std = beerpong_physics(collection)
        values = (raw-mean)/std
    else:
        from datasets.physics import reacher_physics
        raw,mean,std = reacher_physics(collection)
        values = (raw-mean)/std
    return values,{'physics_mean':mean,'physics_std':std}
