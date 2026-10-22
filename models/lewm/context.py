"""Parallel image/action Context reuse with method-specific LeWM parameters."""
import torch
from models.lewm.jepa import JEPA
from models.pusht_context.dp_resnet import DiffusionPolicyResNet
from models.pusht_context.ctx_encoder import Ctx_Encoder
from models.pusht_context.vit_ctx import ViT_Ctx
from models.proprio import ProprioceptiveEmbedding


def build_context(image_size=224, output_dim=64, proprio_dim=4, action_dim=10):
    image = DiffusionPolicyResNet(input_shape=(3, image_size, image_size), num_kp=64, gn_num_groups=8)
    proprio = ProprioceptiveEmbedding(num_frames=1, in_chans=proprio_dim, emb_dim=10) if proprio_dim else None
    action = ProprioceptiveEmbedding(num_frames=1, in_chans=action_dim, emb_dim=10)
    width = image.output_dim + 10 + (10 if proprio is not None else 0)
    pool = ViT_Ctx(in_dim=width, out_dim=output_dim, hidden_dim=512, depth=4,
                   n_heads=8, mlp_hidden_dim=1024, head_dim=64, max_seq_len=12, use_rope=False)
    return Ctx_Encoder(image, proprio, action, pool)


class ContextJEPA(JEPA):
    def __init__(self, *args, ctx_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx_encoder = ctx_encoder

    def encode_context(self, info):
        return self.ctx_encoder(info)

    def encode(self, info):
        output = super().encode(info)
        if 'ctx_visual' in info:
            output['ctx_emb'] = self.encode_context(info)
        elif 'ctx_emb' in info:
            output['ctx_emb'] = info['ctx_emb']
        return output

    def predict(self, embedding, action_embedding, ctx_emb=None):
        context = ctx_emb[:, None].expand(-1, embedding.shape[1], -1)
        prediction = self.predictor(torch.cat((embedding, context), dim=-1), action_embedding)
        return self.pred_proj(prediction.flatten(0, 1)).reshape(*prediction.shape[:2], -1)
