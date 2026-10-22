"""Construct trainable world-model components from configuration and sample shapes."""
from omegaconf import OmegaConf

from models.dino import FrozenDino, DinoV2Encoder, ChannelProjector
from models.proprio import ProprioceptiveEmbedding
from models.visual_world_model import VWorldModel, VWorldModel_PrefixContext
from models.vit import ViTPredictor
from models.vqvae import VQVAE


def arguments(config):
    return {k:v for k,v in config.items() if not k.startswith('_')}


def build_world_model(bound, dino_weights, sample):
    obs, action, _ = sample
    encoder = FrozenDino(dino_weights)
    tswm = bound.encoder.get('projector') == 'channel'
    embedding, decoder_class, model_class = ProprioceptiveEmbedding, VQVAE, VWorldModel
    options = dict(proprio_loss_weight=bound.model.get('proprio_loss_weight', 0.5))
    decoder_options = {}
    if tswm:
        from models.tswm.proprio import ProprioceptiveEmbedding as embedding
        from models.tswm.vqvae import VQVAE as decoder_class
        from models.tswm.world_model import VWorldModel as model_class
        encoder = DinoV2Encoder(base_model=encoder.base_model,
            projector_config=ChannelProjector(**arguments(bound.encoder.projector_config)),
            agg_out_dim=bound.encoder.agg_out_dim, agg_mlp_hidden_dim=bound.encoder.agg_mlp_hidden_dim)
        decoder_options['projector_cfg'] = bound.encoder.projector_config
        options = dict(straighten_scale=float(bound.training.straighten.removeprefix('aggcos')))
    prop = embedding(in_chans=obs['proprio'].shape[-1], emb_dim=bound.proprio_emb_dim,
                     **arguments(bound.proprio_encoder)) if bound.proprio_emb_dim and 'proprio' in obs else None
    act = embedding(in_chans=action.shape[-1], emb_dim=bound.action_emb_dim, **arguments(bound.action_encoder))
    context_dim = 0
    if bound.env.dataset.get('use_ctx', False) and 'ctx_encoder' in bound:
        from models.physics.estimator import build_context_encoder
        from models.pusht_context.visual_wm_ctx import VWorldModel_Context
        metadata = dict(environment=bound.env.name, img_size=bound.img_size,
                        dataset=OmegaConf.to_container(bound.env.dataset),
                        encoder_config={k:OmegaConf.to_container(bound[k]) for k in ('ctx_img_model','ctx_pool_model')},
                        action_emb_dim=bound.action_emb_dim, proprio_emb_dim=bound.proprio_emb_dim,
                        ctx_token_dim=bound.ctx_token_dim)
        context = build_context_encoder(metadata, action_dim=obs['ctx_act'].shape[-1])
        model_class = VWorldModel_Context
        options['ctx_encoder'] = context
        context_dim = context.output_dim
    elif 'prefix_context_frames' in bound:
        model_class = VWorldModel_PrefixContext
        options['prefix_context_frames'] = bound.prefix_context_frames
    dim = encoder.emb_dim + (prop.emb_dim if prop is not None else 0) + bound.action_emb_dim + context_dim
    predictor = ViTPredictor(num_patches=(bound.img_size//16)**2,
                             num_frames=bound.get('predictor_num_frames',bound.num_hist),
                             dim=dim, **arguments(bound.predictor))
    decoder = decoder_class(emb_dim=encoder.emb_dim, **arguments(bound.decoder), **decoder_options)
    model = model_class(bound.img_size, encoder, prop, act, predictor=predictor, decoder=decoder,
                        num_hist=bound.num_hist, num_pred=bound.num_pred, **options)
    if tswm:
        model.data_provenance = {'source':'training configuration'}
    return model


def initialize_components(model, payload):
    for name in ('predictor','action_encoder','proprio_encoder','decoder','ctx_encoder'):
        target, source = getattr(model,name,None), payload.get(name)
        if (target is None) != (source is None):
            raise ValueError(f'Incompatible initialization component: {name}')
        if target is not None:
            target.load_state_dict(source.state_dict(), strict=True)
    if hasattr(model.encoder,'projector'):
        model.encoder.load_state_dict(payload['encoder'].state_dict(), strict=True)
