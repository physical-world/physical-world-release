"""LeWM execution on canonical observations for shared evaluation and simulation."""
import torch
from torch import nn
from models.lewm.preprocessing import prepare_info


class LeWorldModel(nn.Module):
    def __init__(self, jepa, decoder=None, num_hist=3, num_pred=1):
        super().__init__()
        self.jepa = jepa
        self.decoder = decoder
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.proprio_loss_weight = jepa.proprio_loss_weight

    def encode_context(self, observation):
        return self.jepa.encode_context(observation)

    def encode_obs(self, observation):
        main = {key:value for key,value in observation.items() if not key.startswith('ctx_')}
        return self.jepa.split_embedding(self.jepa.encode(prepare_info(main))['emb'])

    def decode_obs(self, latent):
        return {'visual': self.decoder(latent['visual'])}, latent['visual'].new_zeros(())

    def rollout(self, observation, actions):
        info = self.jepa.encode(prepare_info(observation))
        embedding = info['emb']
        context = info.get('ctx_emb', observation.get('ctx_emb'))
        initial = embedding.shape[1]
        for end in range(initial + 1, actions.shape[1] + 2):
            start = max(0, end - 1 - self.num_hist)
            history = embedding[:, start:end-1]
            act = self.jepa.action_encoder(actions[:, start:end-1])
            predicted = self.jepa.predict(history, act, context)[:, -1:]
            embedding = torch.cat((embedding, predicted), dim=1)
        return self.jepa.split_embedding(embedding), embedding

    def forward(self, observation, actions, decode_images=False):
        info = self.jepa.encode(prepare_info(observation, actions))
        predicted = self.jepa.predict(info['emb'][:, :self.num_hist],
                                       info['act_emb'][:, :self.num_hist], info.get('ctx_emb'))
        losses = self.jepa.prediction_losses(predicted, info['emb'][:, self.num_pred:])
        images = reconstructed = None
        if decode_images:
            images = self.decoder(predicted[..., :self.jepa.visual_dim])
            reconstructed = self.decoder(info['visual_emb'])
        components = dict(z_loss=losses['pred_loss'], z_visual_loss=losses['visual_loss'])
        if 'proprio_loss' in losses:
            components['z_proprio_loss'] = losses['proprio_loss']
        if decode_images:
            for name, output, target in [('pred', images, observation['visual'][:, self.num_pred:]),
                                         ('reconstructed', reconstructed, observation['visual'])]:
                error = (output - target).square().mean()
                components.update({f'decoder_recon_loss_{name}':error,
                                   f'decoder_vq_loss_{name}':error.new_zeros(()),
                                   f'decoder_loss_{name}':error})
        return self.jepa.split_embedding(info['emb']), images, reconstructed, losses['pred_loss'], components
