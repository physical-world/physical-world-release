"""Learnable visual representation and action-conditioned next-state prediction."""
import torch
from torch import nn


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj,
                 proprio_encoder=None, visual_dim=192, proprio_loss_weight=0.5):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj
        self.proprio_encoder = proprio_encoder
        self.visual_dim = visual_dim
        self.proprio_loss_weight = proprio_loss_weight

    @property
    def has_proprio(self):
        return self.proprio_encoder is not None

    def split_embedding(self, embedding):
        pieces = {'visual': embedding[..., :self.visual_dim]}
        if self.has_proprio:
            pieces['proprio'] = embedding[..., self.visual_dim:]
        return pieces

    def encode(self, info):
        pixels = info['pixels'].float()
        batch, time = pixels.shape[:2]
        cls = self.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
        visual = self.projector(cls).reshape(batch, time, -1)
        output = dict(info, visual_emb=visual, emb=visual)
        if self.has_proprio:
            proprio = self.proprio_encoder(info['proprio'].float())
            output.update(proprio_emb=proprio, emb=torch.cat((visual, proprio), dim=-1))
        if 'action' in info:
            output['act_emb'] = self.action_encoder(info['action'])
        return output

    def predict(self, embedding, action_embedding, ctx_emb=None):
        predicted = self.predictor(embedding, action_embedding)
        return self.pred_proj(predicted.flatten(0, 1)).reshape(*predicted.shape[:2], -1)

    def prediction_losses(self, predicted_emb, target_emb):
        predicted, target = self.split_embedding(predicted_emb), self.split_embedding(target_emb)
        visual_loss = (predicted['visual'] - target['visual']).square().mean()
        losses = dict(visual_loss=visual_loss, pred_loss=visual_loss)
        if self.has_proprio:
            proprio_loss = (predicted['proprio'] - target['proprio'].detach()).square().mean()
            losses.update(proprio_loss=proprio_loss, weighted_proprio_loss=self.proprio_loss_weight * proprio_loss)
            losses['pred_loss'] = visual_loss + losses['weighted_proprio_loss']
        return losses
