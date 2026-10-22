"""Standalone supervised physical estimation and original-object restoration."""
from pathlib import Path

import torch
from torch import nn

from runtime.checkpoints import ROOT

def load_estimator(path):
    """Restore trusted paper checkpoints with their original component classes."""
    return torch.load(path,map_location='cpu',weights_only=False)


class PhysicsEstimator(nn.Module):
    def __init__(self,ctx_encoder,output_dim):
        super().__init__()
        self.ctx_encoder = ctx_encoder
        self.regressor = nn.Sequential(nn.Linear(ctx_encoder.output_dim,64),nn.GELU(),nn.Linear(64,output_dim))

    def forward(self,context):
        return self.regressor(self.ctx_encoder(context))


def build_context_encoder(metadata, *, action_dim=None):
    """Explicit constructor from the bound encoder dimensions and settings."""
    from models.pusht_context.dp_resnet import DiffusionPolicyResNet
    from models.pusht_context.ctx_encoder import Ctx_Encoder
    from models.pusht_context.vit_ctx import ViT_Ctx
    from models.proprio import ProprioceptiveEmbedding
    image_config = metadata['encoder_config']['ctx_img_model']
    pool_config = metadata['encoder_config']['ctx_pool_model']
    image = DiffusionPolicyResNet(input_shape=(3,metadata['img_size'],metadata['img_size']),
                                 num_kp=image_config['num_kp'],gn_num_groups=image_config['gn_num_groups'])
    dataset = metadata['dataset']
    frames = dataset['ctx_length']//dataset['ctx_frameskip']
    if action_dim is None:
        action_dim = len(metadata['input_normalization']['action_mean'])
        if metadata['environment'] in ('pusht','reacher'):
            action_dim *= dataset['ctx_frameskip']
    action = ProprioceptiveEmbedding(num_frames=1,in_chans=action_dim,emb_dim=metadata['action_emb_dim'])
    prop_dim = metadata['proprio_emb_dim']
    proprio = ProprioceptiveEmbedding(num_frames=1,in_chans=4,emb_dim=prop_dim) if prop_dim else None
    pool = ViT_Ctx(in_dim=image.output_dim+prop_dim+metadata['action_emb_dim'],out_dim=metadata['ctx_token_dim'],
                   hidden_dim=pool_config['hidden_dim'],depth=pool_config['depth'],n_heads=pool_config['n_heads'],
                   mlp_hidden_dim=pool_config['mlp_hidden_dim'],head_dim=pool_config['head_dim'],max_seq_len=frames,
                   use_rope=pool_config['use_rope'],dropout=pool_config['dropout'],emb_dropout=pool_config['emb_dropout'])
    return Ctx_Encoder(image,proprio,action,pool)


def build_estimator(metadata):
    return PhysicsEstimator(build_context_encoder(metadata), len(metadata['names']))


def save_estimator(path,payload):
    """Save original component objects without changing their class identities."""
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'logs').resolve()):
        raise ValueError('New checkpoints must be written under logs/')
    with path.open('xb') as stream:
        torch.save(payload,stream)


def load_physics_estimator(path,device='cpu'):
    """Restore the original encoder/regressor objects for frozen inference."""
    payload = load_estimator(path)
    metadata = payload['physics_estimator']
    model = PhysicsEstimator(payload['ctx_encoder'],len(metadata['names']))
    model.regressor.load_state_dict(payload['regressor'].state_dict(),strict=True)
    return model.to(device).eval().requires_grad_(False),metadata,payload


def convert_physics(value,estimator_mean,estimator_std,physics_mean,physics_std):
    """Undo estimator output scaling, then apply Privileged Physics training statistics."""
    return (value*torch.as_tensor(estimator_std,device=value.device)+
            torch.as_tensor(estimator_mean,device=value.device)-
            torch.as_tensor(physics_mean,device=value.device))/torch.as_tensor(physics_std,device=value.device)
