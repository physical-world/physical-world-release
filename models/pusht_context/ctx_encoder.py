"""Image-only spatial encoding followed by separate proprio/action fusion."""
import torch
from torch import nn


class Ctx_Encoder(nn.Module):
    def __init__(self,img_model,prop_model,act_model,pool_model,
                 l2_normalize=False,use_layer_norm=False):
        super().__init__()
        self.img_model = img_model
        self.prop_model = prop_model
        self.act_model = act_model
        self.pool_model = pool_model
        self.output_dim = pool_model.output_dim
        self.l2_normalize = l2_normalize
        self.use_layer_norm = use_layer_norm
        self.ctx_layer_norm = None

    def frame_tokens(self,obs):
        images = obs['ctx_visual']
        batch,time = images.shape[:2]
        pieces = [self.img_model(images.flatten(0,1)).reshape(batch,time,-1)]
        if self.prop_model is not None:
            pieces.append(self.prop_model(obs['ctx_proprio']))
        pieces.append(self.act_model(obs['ctx_act']))
        return torch.cat(pieces,dim=-1)

    def forward(self,obs):
        return self.pool_model(self.frame_tokens(obs))
