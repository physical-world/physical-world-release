"""Context model at its original serialized import path."""
import torch

from models.visual_world_model import VWorldModel


class VWorldModel_Context(VWorldModel):
    def __init__(self, *args, ctx_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx_encoder = ctx_encoder
        self.condition_dim = ctx_encoder.output_dim
        self.ctx_token_dim = self.condition_dim
        self.ctx_detach_for_predictor = False
        self.extra_losses = {}
        self.emb_dim += self.condition_dim

    def encode_context(self, obs):
        return obs['ctx_emb'] if 'ctx_emb' in obs else self.ctx_encoder(obs)

    def encode_precomputed_obs(self, z_dct, act, ctx_emb):
        main = super().encode_precomputed_obs(z_dct, act)
        condition = ctx_emb[:, None, None].expand(*main.shape[:3], -1)
        return torch.cat((main, condition), dim=-1)

    def encode(self, obs, act, ctx_emb=None):
        context = self.encode_context(obs) if ctx_emb is None else ctx_emb
        return self.encode_precomputed_obs(self.encode_obs(obs), act, context)

    def _rollout_condition(self, obs):
        return obs['ctx_emb'] if 'ctx_emb' in obs else self.encode_context(obs)

    def _rollout_encode(self, obs, act, condition):
        return self.encode(obs, act, ctx_emb=condition)

    def _rollout_prediction(self, z, condition):
        prediction = self.predict(z)[:, -1:]
        fixed = condition[:, None, None].expand(*prediction.shape[:3], -1)
        return torch.cat((prediction[..., :-self.condition_dim], fixed), dim=-1)
