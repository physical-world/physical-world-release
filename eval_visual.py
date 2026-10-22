"""Paper checkpoint evaluation for Reacher, FetchSlide and BeerPong."""
import json
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from runtime.checkpoints import ROOT, create_run, source_revision
from models.dino import FrozenDino
from models.visual_world_model import VWorldModel
from datasets.evaluation_targets import prepare_targets, check_target_geometry


@hydra.main(version_base='1.3', config_path='conf', config_name='eval_reacher')
def main(config):
    if config.phase not in ('nonplanning', 'planning'):
        raise ValueError('phase must be nonplanning or planning')
    if config.environment == 'reacher':
        from datasets.reacher import ReacherEpisodes as Episodes, reacher_normalization as normalize
        from evaluation.reacher_evaluation import export_reacher as export_nonplanning
        from planning.reacher_export import export_reacher_planning as export_planning
    elif config.environment == 'fetch_slide':
        from datasets.fetch_slide import FetchSlideEpisodes as Episodes, fetch_normalization as normalize
        from evaluation.fetch_slide_evaluation import export_fetch as export_nonplanning
        from planning.fetch_slide_export import export_fetch_planning as export_planning
    elif config.environment == 'beerpong':
        from datasets.beerpong import BeerPongEpisodes as Episodes, beerpong_normalization as normalize
        from evaluation.beerpong_evaluation import export_beerpong as export_nonplanning
        from planning.beerpong_export import export_beerpong_planning as export_planning
    else:
        raise ValueError(config.environment)
    paper = OmegaConf.load(ROOT / config.paper_eval_config)
    paper.eval.seed, paper.plan.seed = config.eval_seed, config.plan_seed
    paper.plan = OmegaConf.merge(paper.plan, config.planning_overrides)
    run = create_run(config)
    shutil.copy2(ROOT / config.paper_eval_config, run / 'paper_source.yaml')
    OmegaConf.save(paper, run / 'paper_eval.yaml')
    (run / 'source_revision.txt').write_text(f"{source_revision() or 'unavailable'}\n")
    for source in (ROOT / 'eval_visual.py', ROOT / 'evaluation/visual_evaluation.py', ROOT / f'evaluation/{config.environment}_evaluation.py',
                   *sorted((ROOT / 'planning').glob('*.py'))):
        relative = source.relative_to(ROOT)
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    collection = Episodes(ROOT / config.manifest)
    _, manifest, sampling = prepare_targets(config, collection, config.environment, config.phase, run)
    targets = manifest['targets']
    bound = OmegaConf.load(ROOT / config.checkpoint_config)
    check_target_geometry(manifest,bound.frameskip)
    torch.manual_seed(config.eval_seed)
    np.random.seed(config.eval_seed)
    payload = torch.load(ROOT / config.checkpoint, map_location='cpu', weights_only=False)
    model = VWorldModel(bound.img_size, FrozenDino(config.get('dino_weights')), None, payload['action_encoder'],
                        predictor=payload['predictor'], decoder=payload['decoder'], num_hist=bound.num_hist,
                        num_pred=bound.num_pred, proprio_loss_weight=None).to(config.device).eval()
    normalization = normalize(collection)
    torch.save(normalization, run / 'normalization.pt')
    inspection = config.num_targets is not None or config.max_targets is not None or config.max_batches is not None or bool(config.planning_overrides)
    metadata = dict(ckpt_path=str((ROOT / config.checkpoint).resolve()), ckpt_epoch=payload['epoch'],
                    eval_seed=config.eval_seed, target_set_hash=manifest['target_set_hash'],
                    selected_target_count=len(targets), inspection=inspection)
    print(f'Run directory: {run}', flush=True)
    if config.phase == 'nonplanning':
        from evaluation.image_metrics import LPIPS
        from evaluation.segmentation import load_segmentation_model
        segmentation, _ = load_segmentation_model(ROOT / config.segmentation, config.device)
        masks = zarr.open_group(str(ROOT / config.masks), mode='r')
        export_nonplanning(model, collection, targets, normalization, masks, run / 'nonplanning', metadata,
                           LPIPS(net_type="vgg", version="0.1").to(config.device).eval().requires_grad_(False), segmentation, batch_size=paper.eval.batch_size,
                           max_batches=config.max_batches, num_visualizations=paper.eval.num_viz)
    else:
        search = paper.plan.planner.sub_planner if config.environment == 'reacher' else paper.plan.cem
        settings = {key: search[key] for key in ('num_samples', 'topk', 'opt_steps', 'var_scale')}
        export_planning(model, collection, targets, normalization, run / 'planning', metadata, settings,
                        seed=config.plan_seed, inspection=inspection, num_visualizations=paper.eval.num_viz,
                        candidate_chunk_size=paper.plan.get('wm_candidate_chunk_size'))


if __name__ == '__main__':
    main()
