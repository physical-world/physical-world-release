"""Explicit assembly of the paper's LeWM modules from the bound model arguments."""
from transformers import ViTConfig, ViTModel
from models.lewm.embedding import Embedder
from models.lewm.predictor import ARPredictor
from models.lewm.projector import MLP
from models.lewm.jepa import JEPA
from models.proprio import ProprioceptiveEmbedding


def build_model(config):
    arguments = config.model
    visual_dim = config.embed_dim
    encoder_config = ViTConfig(
        image_size=config.img_size, patch_size=arguments.encoder.patch_size,
        num_channels=3, hidden_size=visual_dim, num_hidden_layers=12,
        num_attention_heads=3, intermediate_size=4 * visual_dim,
        hidden_act='gelu', hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        initializer_range=0.02, layer_norm_eps=1e-12, qkv_bias=True)
    encoder_config.interpolate_pos_encoding = True
    encoder = ViTModel(encoder_config, add_pooling_layer=False, use_mask_token=False)
    proprio = None
    if 'proprio_encoder' in arguments:
        prop = arguments.proprio_encoder
        proprio = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1,
                                          in_chans=prop.input_dim, emb_dim=prop.output_dim)
    predictor = arguments.predictor
    model_class = JEPA
    options = {}
    if 'ctx_encoder' in arguments:
        from models.lewm.context import ContextJEPA, build_context
        context = arguments.ctx_encoder
        model_class = ContextJEPA
        options['ctx_encoder'] = build_context(image_size=context.image_size, output_dim=context.output_dim,
                                              proprio_dim=context.get('proprio_dim', 4),
                                              action_dim=context.get('action_dim', 10))
    model = model_class(
        encoder=encoder,
        predictor=ARPredictor(num_frames=predictor.num_frames, input_dim=predictor.input_dim,
                              hidden_dim=predictor.hidden_dim, output_dim=predictor.output_dim,
                              depth=predictor.depth, heads=predictor.heads, dim_head=predictor.dim_head,
                              mlp_dim=predictor.mlp_dim, dropout=predictor.dropout,
                              emb_dropout=predictor.emb_dropout,
                              condition_dim=predictor.get('condition_dim', predictor.input_dim)),
        action_encoder=Embedder(input_dim=arguments.action_encoder.input_dim,
                                emb_dim=arguments.action_encoder.emb_dim),
        projector=MLP(arguments.projector.input_dim, arguments.projector.hidden_dim,
                      arguments.projector.output_dim),
        pred_proj=MLP(arguments.pred_proj.input_dim, arguments.pred_proj.hidden_dim,
                      arguments.pred_proj.output_dim),
        proprio_encoder=proprio, visual_dim=visual_dim,
        proprio_loss_weight=arguments.get('proprio_loss_weight', 0.5), **options)
    return model
