"""TSWM adds aggregation straightening to the shared prediction/decoder flow."""
from models.visual_world_model import VWorldModel as PredictiveModel
from models.tswm.curvature import aggregation_curvature


class VWorldModel(PredictiveModel):
    def __init__(self, image_size, encoder, proprio_encoder, action_encoder, *,
                 predictor, decoder=None, num_hist=3, num_pred=1, straighten_scale=0.1):
        super().__init__(image_size, encoder, proprio_encoder, action_encoder, predictor=predictor,
                         decoder=decoder, num_hist=num_hist, num_pred=num_pred, proprio_loss_weight=None)
        self.straighten_scale = straighten_scale

    def _latent_regularization(self, encoded, valid):
        curvature = aggregation_curvature(self.encoder, encoded[..., :self.encoder.emb_dim], valid)
        return curvature * self.straighten_scale, {'curvature_loss_used_for_training': curvature}
