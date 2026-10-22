"""Strict loading and raw parameter-dictionary saving, preserving paper payloads."""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.lewm.construction import build_model


def load_jepa(checkpoint, configuration, device='cpu'):
    bound = OmegaConf.load(configuration)
    model = build_model(bound)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), bound


def save_weights(path, model):
    path = Path(path).resolve()
    from runtime.checkpoints import ROOT
    path.relative_to((ROOT / 'logs').resolve())
    with path.open('xb') as stream:
        torch.save(model.state_dict(), stream)
