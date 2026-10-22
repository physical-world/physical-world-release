"""Epoch-boundary recovery shared by the training entry points."""
import json
import random
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
import torch


def validate_mode(config):
    if config.get('init_from') and config.get('resume_from'):
        raise ValueError('init_from and resume_from are mutually exclusive')
    if config.get('resume_from') and not (Path(config.resume_from) / 'training.json').is_file():
        raise ValueError('resume_from requires a complete state_epoch_N directory; use init_from for weight-only checkpoints')


def model_config(config, asset=None):
    validate_mode(config)
    asset = config if asset is None else asset
    path = Path(config.resume_from)/'model_config.yaml' if config.resume_from else Path(asset.model_config)
    return OmegaConf.load(path)


def _signature(config):
    values = OmegaConf.to_container(config, resolve=True)
    for key in ('init_from', 'resume_from', 'logging', 'logging_mode', 'project', 'epochs'):
        values.pop(key, None)
    if 'training' in values:
        values['training'].pop('epochs', None)
    return values


def restore_training(config, accelerator, loaders):
    validate_mode(config)
    if not config.resume_from:
        return [], 0
    path = Path(config.resume_from)
    record = json.loads((path/'training.json').read_text())
    if record['world_size'] != accelerator.num_processes or record['settings'] != _signature(config):
        raise ValueError('Full resume requires the same training settings and world size (epoch limit and logging may change)')
    accelerator.load_state(str(path))
    state = torch.load(path/f'loaders_{accelerator.process_index}.pt', map_location='cpu', weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    accelerator.step = state['accelerator_step']
    for loader, saved in zip(loaders, state['loaders']):
        if saved['generator'] is not None:
            loader.generator.set_state(saved['generator'])
        if hasattr(loader, 'iteration'):
            loader.iteration = saved['iteration']
    return record['history'], record['step']


def save_training(run, config, bound, accelerator, loaders, history, step):
    """All ranks participate; the completion marker is written last."""
    path = Path(run)/f"state_epoch_{history[-1]['epoch']}"
    accelerator.save_state(str(path), safe_serialization=False)
    states = [dict(generator=loader.generator.get_state() if loader.generator is not None else None,
                   iteration=getattr(loader, 'iteration', 0)) for loader in loaders]
    state = dict(loaders=states, python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 accelerator_step=accelerator.step)
    torch.save(state, path/f'loaders_{accelerator.process_index}.pt')
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(bound, path/'model_config.yaml', resolve=True)
        record = dict(history=history, step=step, world_size=accelerator.num_processes, settings=_signature(config))
        (path/'training.json').write_text(json.dumps(record, indent=2)+'\n')
    accelerator.wait_for_everyone()
