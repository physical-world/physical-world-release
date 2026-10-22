"""Convert canonical RGB in [-1,1] to the paper's ImageNet visual inputs."""
import torch


def prepare_info(observation, action=None):
    visual = observation['visual']
    mean = visual.new_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std = visual.new_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
    info = {key:value for key,value in observation.items() if key != 'visual'}
    info['pixels'] = ((visual + 1) / 2 - mean) / std
    if action is not None:
        info['action'] = torch.nan_to_num(action.float(), nan=0.0)
    return info
