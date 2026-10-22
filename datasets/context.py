"""Split-local donor selection and bounded temporal input reads."""
from collections import defaultdict

import numpy as np
import torch

from datasets.common import _read_window
from datasets.pusht import pusht_observation


class ContextSampler:
    """Select another split-local episode with the same physical condition."""

    def __init__(self,manifest,episode_ids,conditions,length=60,frameskip=5,fixed_start=None,min_episode_length=0):
        self.manifest = manifest
        self.episode_ids = list(episode_ids)
        self.length = length
        self.frameskip = frameskip
        self.fixed_start = fixed_start
        self.conditions = dict(zip(self.episode_ids,map(int,conditions)))
        self.groups = defaultdict(list)
        for episode,condition in self.conditions.items():
            _,_,begin,end = manifest.episode_info(episode)
            if end-begin >= min_episode_length:
                self.groups[condition].append(episode)

    def sample(self,target,rng=None):
        rng = np.random if rng is None else rng
        candidates = [ep for ep in self.groups[self.conditions[target]] if ep != target]
        donor = candidates[int(rng.randint(len(candidates)))]
        _,_,begin,end = self.manifest.episode_info(donor)
        start = self.fixed_start if self.fixed_start is not None else int(rng.randint(max(0,end-begin-self.length)+1))
        frames = np.minimum(start+np.arange(0,self.length,self.frameskip),end-begin-1)
        actions = np.minimum(start+np.arange(self.length),end-begin-1)
        return {'target':target,'donor':donor,'condition':self.conditions[target],
                'start':start,'frames':frames,'actions':actions}


class PushTContexts(ContextSampler):
    def __init__(self,manifest,episode_ids,length=60,frameskip=5):
        ids = list(episode_ids)
        seeds = pusht_episode_values(manifest,ids,'cog_seed')
        super().__init__(manifest,ids,seeds,length,frameskip)


def read_pusht_context(manifest, record, length, frameskip, image_size, normalization):
    """Read only donor frames; hold terminal RGB/state and absolute controls."""
    path, _, begin, end = manifest.episode_info(record['donor'])
    window = _read_window(manifest.group(path)['data'], begin + record['start'],
                          length // frameskip, frameskip,
                          normalization['action_mean'], normalization['action_std'],
                          episode_stop=end)
    observation, _ = pusht_observation(window['rgb'], window['state'], image_size, normalization)
    return {'ctx_visual': observation['visual'], 'ctx_proprio': observation['proprio'],
            'ctx_act': window['action']}


def pusht_episode_values(manifest,episode_ids,key):
    """Read one metadata row per episode, grouping selections by shard."""
    selections = defaultdict(list)
    for position,episode in enumerate(episode_ids):
        path,_,begin,_ = manifest.episode_info(episode)
        selections[path].append((position,begin))
    values = [None]*len(episode_ids)
    for path,rows in selections.items():
        selected = manifest.group(path)['data'][key].oindex[[begin for _,begin in rows]]
        for (position,_),value in zip(rows,selected):
            values[position] = value
    return np.asarray(values)


def pusht_physics_targets(manifest,episode_ids):
    return torch.from_numpy(pusht_episode_values(manifest,episode_ids,'cog_local_normed')).float()


def read_reacher_context(manifest,record,length,frameskip,image_size,normalization):
    from datasets.common import preprocess_rgb
    path,_,begin,_ = manifest.episode_info(record['donor'])
    window = _read_window(manifest.group(path)['data'],begin+record['start'],length//frameskip,
                          frameskip,normalization['action_mean'],normalization['action_std'])
    return {'ctx_visual':preprocess_rgb(window['rgb'],image_size),'ctx_act':window['action']}


def read_condition_context(manifest,record,length,frameskip,image_size,normalization,raw_condition):
    from datasets.common import preprocess_rgb
    _,_,begin,end = manifest.episode_info(record['donor'])
    frames = np.minimum(np.arange(0,length,frameskip),end-begin-1)+begin
    images = preprocess_rgb(manifest.root['data/img'].oindex[frames],image_size)
    condition = (raw_condition-normalization['action_mean'])/normalization['action_std']
    return {'ctx_visual':images,'ctx_act':condition.expand(len(frames),-1)}


def read_fetch_context(manifest,record,length,frameskip,image_size,normalization):
    from datasets.fetch_slide import fetch_conditions
    raw = fetch_conditions(manifest,[record['donor']])[0]
    return read_condition_context(manifest,record,length,frameskip,image_size,normalization,raw)


def read_beerpong_context(manifest,record,length,frameskip,image_size,normalization):
    from datasets.beerpong import beerpong_conditions
    raw = beerpong_conditions(manifest,[record['donor']])[0]
    return read_condition_context(manifest,record,length,frameskip,image_size,normalization,raw)


class ContextDataset(torch.utils.data.Dataset):
    """Add a separate same-physics donor to each canonical main window."""
    def __init__(self, windows, sampler, read_context, image_size, normalization):
        self.windows = windows
        self.sampler = sampler
        self.read_context = read_context
        self.image_size = image_size
        self.normalization = normalization

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        position, _ = self.windows.locate(index)
        target = self.windows.episode_ids[position]
        record = self.sampler.sample(target)
        context = self.read_context(self.windows.manifest, record, self.sampler.length,
                                    self.sampler.frameskip, self.image_size, self.normalization)
        observation, action, state = self.windows[index]
        return {**observation, **context}, action, state


class FixedContexts:
    """Encode each recorded donor once, preserving protocol-specific assignments."""
    def __init__(self, model, collection, records, read_context, geometry, image_size, normalization):
        self.model = model
        self.collection = collection
        self.records = records
        self.read_context = read_context
        self.geometry = geometry
        self.image_size = image_size
        self.normalization = normalization
        self.cache = {}

    @torch.inference_mode()
    def __getitem__(self, episode):
        if episode not in self.cache:
            context = self.read_context(self.collection, self.records[episode],
                                        self.geometry['ctx_length'], self.geometry['ctx_frameskip'],
                                        self.image_size, self.normalization)
            device = next(self.model.parameters()).device
            self.cache[episode] = {'ctx_emb': self.model.encode_context(
                {key:value[None].to(device) for key,value in context.items()})[0].cpu()}
        return self.cache[episode]
