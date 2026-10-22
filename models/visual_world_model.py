"""DINO world model: encoding, prediction and rollout for bound conditions."""

import torch
from torch import nn
from torchvision import transforms


def masked_frame_mse(pred, target, valid=None):
    """Average features per frame, real frames per window, then windows."""
    if valid is None:
        return nn.functional.mse_loss(pred, target)
    if valid.dtype != torch.bool or valid.shape != pred.shape[:2]:
        raise ValueError("valid_frames must be boolean [batch, time]")
    counts = valid.sum(1)
    if not torch.all(counts > 0):
        raise ValueError("Each window must contain a real target")
    error = (pred - target).square().flatten(2).mean(2)
    return (error.masked_fill(~valid, 0).sum(1) / counts).mean()


class VWorldModel(nn.Module):
    """Assemble feature-concatenated tokens for bound Vanilla and Privileged Physics checkpoints.

    Each row-major image patch receives the same frame's proprio and action
    embeddings. Forward defaults to latent loss; validation can request images.
    The bound model uses one copy of each embedding and a 16x decoder scale.
    """

    def __init__(self, image_size, encoder, proprio_encoder, action_encoder, *,
                 predictor=None, decoder=None, num_hist=3, num_pred=1, proprio_loss_weight=0.5):
        super().__init__()
        self.predictor = predictor
        self.decoder = decoder
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.proprio_loss_weight = proprio_loss_weight
        self.emb_criterion = nn.MSELoss()
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.proprio_dim = 0 if proprio_encoder is None else proprio_encoder.emb_dim
        self.action_dim = action_encoder.emb_dim
        self.emb_dim = encoder.emb_dim + self.proprio_dim + self.action_dim
        self.concat_dim = 1
        self.num_proprio_repeat = 1
        self.num_action_repeat = 1
        self.condition_dim = 0
        self.encoder_image_size = (image_size // 16) * encoder.patch_size
        self.encoder_transform = transforms.Compose([
            transforms.Resize(self.encoder_image_size, antialias=True),
        ])

    def encode_act(self, act):
        return self.action_encoder(act)

    def encode_proprio(self, proprio):
        return self.proprio_encoder(proprio)

    def encode_obs(self, obs):
        batch, time = obs["visual"].shape[:2]
        # Match the reference NCHW layout used by DINO's patch convolution.
        images = self.encoder_transform(obs["visual"].flatten(0, 1)).contiguous()
        visual = self.encoder(images).unflatten(0, (batch, time))
        result = {"visual": visual}
        if self.proprio_encoder is not None:
            result["proprio"] = self.encode_proprio(obs["proprio"])
        return result

    def encode_precomputed_obs(self, z_dct, act):
        visual = z_dct["visual"]
        patches = visual.shape[2]
        pieces = [visual]
        if self.proprio_dim:
            pieces.append(z_dct["proprio"].unsqueeze(2).expand(-1, -1, patches, -1))
        pieces.append(self.encode_act(act).unsqueeze(2).expand(-1, -1, patches, -1))
        return torch.cat(pieces, dim=-1)

    def encode(self, obs, act):
        return self.encode_precomputed_obs(self.encode_obs(obs), act)

    def separate_emb(self, z):
        visual, proprio, action, _ = z.split(
            (self.encoder.emb_dim, self.proprio_dim, self.action_dim, self.condition_dim), dim=-1
        )
        observation = {"visual": visual}
        if self.proprio_dim:
            observation["proprio"] = proprio[:, :, 0]
        return observation, action[:, :, 0]

    def predict(self, z):
        """Predict the next latent at each frame, preserving patch order."""
        return self.predictor(z.flatten(1, 2)).unflatten(1, z.shape[1:3])

    def decode(self, z):
        z_obs, _ = self.separate_emb(z)
        return self.decode_obs(z_obs)

    def decode_obs(self, z_obs):
        visual, difference = self.decoder(z_obs["visual"])
        obs = {"visual": visual.unflatten(0, z_obs["visual"].shape[:2])}
        if "proprio" in z_obs:
            obs["proprio"] = z_obs["proprio"]
        return obs, difference

    def _training_latents(self, obs, act):
        z = self.encode(obs, act)
        return z, z[:, :self.num_hist], z[:, self.num_pred:]

    def _decode_training(self, z, valid):
        """Decode real frames only, with the bound non-quantized decoder."""
        if valid is None:
            return self.decode(z)
        decoded, difference = self.decode(z[valid].unsqueeze(0))
        visual = decoded["visual"].new_zeros((*valid.shape, *decoded["visual"].shape[2:]))
        visual[valid] = decoded["visual"][0]
        return {"visual": visual}, difference

    def latent_loss(self, pred, target, valid=None):
        """Detach targets; supervise visual/proprio at every patch, not actions."""
        target = target.detach()
        visual_end = self.encoder.emb_dim
        proprio_end = visual_end + self.proprio_dim
        visual_loss = masked_frame_mse(pred[..., :visual_end], target[..., :visual_end], valid)
        if not self.proprio_dim:
            return {"z_visual_loss": visual_loss, "z_loss": visual_loss}
        proprio_loss = masked_frame_mse(
            pred[..., visual_end:proprio_end], target[..., visual_end:proprio_end], valid
        )
        if self.proprio_loss_weight is None:
            loss = masked_frame_mse(pred[..., :proprio_end], target[..., :proprio_end], valid)
        else:
            loss = visual_loss + self.proprio_loss_weight * proprio_loss
        return {"z_visual_loss": visual_loss, "z_proprio_loss": proprio_loss, "z_loss": loss}

    def _latent_regularization(self, encoded, valid):
        return 0, {}

    def forward(self, obs, act, decode_images=False):
        """Latent objective; optionally include the bound validation decoder terms."""
        z, source, target = self._training_latents(obs, act)
        pred = self.predict(source)[:, -target.shape[1]:]
        valid = obs.get("valid_frames")
        target_valid = None if valid is None else valid[:, self.num_pred:]
        losses = self.latent_loss(pred, target, target_valid)
        regularization, auxiliary = self._latent_regularization(z, valid)
        losses.update(auxiliary)
        losses["loss"] = losses["z_loss"] + regularization
        images = [None, None]
        if decode_images and self.decoder is not None:
            for index, (name, tokens, visual, mask) in enumerate([
                ("pred", pred, obs["visual"][:, self.num_pred:], target_valid),
                ("reconstructed", z, obs["visual"], valid),
            ]):
                decoded, difference = self._decode_training(tokens.detach(), mask)
                images[index] = decoded["visual"]
                recon_loss = masked_frame_mse(images[index], visual, mask)
                losses[f"decoder_recon_loss_{name}"] = recon_loss
                losses[f"decoder_vq_loss_{name}"] = difference
                losses[f"decoder_loss_{name}"] = recon_loss + 0.25 * difference
            losses["loss"] = losses["loss"] + losses["decoder_loss_reconstructed"]
        return pred, images[0], images[1], losses["loss"], losses

    def replace_actions_from_z(self, z, act):
        """Replace predicted action features with supplied controls, in place."""
        start = self.encoder.emb_dim + self.proprio_dim
        z[..., start:start + self.action_dim] = self.encode_act(act).unsqueeze(2)
        return z

    def _rollout_condition(self, obs):
        return None

    def _rollout_encode(self, obs, act, condition):
        return self.encode(obs, act)

    def _rollout_prediction(self, z, condition):
        return self.predict(z)[:, -1:]

    def rollout(self, obs_0, act, rollout_history_size=None):
        """Return initial and imagined observations, then their full tokens.

        Actions include controls aligned with the initial observations. With A
        action groups, return A+1 frames; only the last frame keeps predicted
        action features, which are unused. Gradients through actions are kept.
        """
        history_size = self.num_hist if rollout_history_size is None else rollout_history_size
        if isinstance(history_size, bool) or not isinstance(history_size, int) or not 1 <= history_size <= self.num_hist:
            raise ValueError("rollout_history_size must be an integer between 1 and checkpoint num_hist")
        num_obs_init = obs_0["visual"].shape[1]
        condition = self._rollout_condition(obs_0)
        z = self._rollout_encode(obs_0, act[:, :num_obs_init], condition)
        for step in range(num_obs_init, act.shape[1]):
            z_new = self._rollout_prediction(z[:, -history_size:], condition).clone()
            self.replace_actions_from_z(z_new, act[:, step:step + 1])
            z = torch.cat((z, z_new), dim=1)
        z_final = self._rollout_prediction(z[:, -history_size:], condition)
        z = torch.cat((z, z_final), dim=1)
        z_obs, _ = self.separate_emb(z)
        return z_obs, z


class VWorldModel_PrefixContext(VWorldModel):
    """Full main-encoder patch tokens from a fixed, separate donor episode."""
    def __init__(self, *args, prefix_context_frames=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_context_frames = prefix_context_frames
        self.direct_context_frames = prefix_context_frames

    def encode_context(self, obs):
        if 'ctx_emb' in obs:
            return obs['ctx_emb']
        context = {'visual': obs['ctx_visual']}
        if self.proprio_encoder is not None:
            context['proprio'] = obs['ctx_proprio']
        return super().encode(context, obs['ctx_act'])

    def predict_with_context(self, z, context):
        return self.predict(torch.cat((context, z), dim=1))[:, -z.shape[1]:]

    def _training_latents(self, obs, act):
        z = super().encode(obs, act)
        source = torch.cat((self.encode_context(obs), z[:, :self.num_hist]), dim=1)
        return z, source, z[:, self.num_pred:]

    def _rollout_condition(self, obs):
        return self.encode_context(obs)

    def _rollout_prediction(self, z, condition):
        return self.predict_with_context(z, condition)[:, -1:]
