"""Independent LeWM non-planning and latent-only planning on bound paper targets."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from runtime.checkpoints import ROOT, source_revision
from models.lewm.checkpoint import load_jepa
from models.lewm.decoder import load_decoder
from models.lewm.world_model import LeWorldModel
from datasets.lewm import build_datasets
from datasets.evaluation_targets import add_sampling_arguments, load_collection, prepare_targets, check_target_geometry
from datasets.evaluation_contexts import prepare_contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['pusht', 'reacher', 'fetch_slide', 'beerpong'], default='pusht')
    parser.add_argument('--method', default='lewm')
    parser.add_argument('--workflow', choices=['nonplanning', 'planning'], required=True)
    parser.add_argument('--inspection', action='store_true')
    parser.add_argument('--eval-seed', type=int, default=42)
    parser.add_argument('--plan-seed', type=int)
    add_sampling_arguments(parser)
    args = parser.parse_args()
    if args.plan_seed is None:
        args.plan_seed = 99 if args.environment == 'pusht' else 42
    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    asset = OmegaConf.load(f'conf/assets/{args.environment}_{args.method}.yaml')
    run = ROOT/'logs'/args.method/args.environment/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    collection = load_collection(args.environment, asset.manifest)
    targets_path, selection, sampling = prepare_targets(args, collection, args.environment, args.workflow, run)
    configuration = OmegaConf.load(asset.checkpoint_config)
    check_target_geometry(selection,15 if args.environment=='beerpong' else 5)
    ctx_path, ctx_selection = prepare_contexts(args, collection, selection, sampling, args.workflow, run) if configuration.data.get('context',False) else (None,None)
    jepa, bound = load_jepa(asset.checkpoint, asset.checkpoint_config, 'cuda')
    decoder = load_decoder(asset.decoder, 'cuda')[0] if args.workflow == 'nonplanning' else None
    model = LeWorldModel(jepa, decoder, bound.history_size, bound.num_preds).eval().requires_grad_(False)
    collection, normalization, _ = build_datasets(asset, bound, inspection=True)
    metadata = dict(environment=args.environment, method=args.method, world_model_kind='lewm',
                    candidate_evaluation='wm_lewm', planning_latent_cost='lewm_latent',
                    checkpoint=str(Path(asset.checkpoint).resolve()),
                    configuration=str(Path(asset.checkpoint_config).resolve()), manifest=str(Path(asset.manifest).resolve()),
                    decoder=None if decoder is None else str(Path(asset.decoder).resolve()), targets=str(targets_path.resolve()),
                    job_id=os.environ.get('SLURM_JOB_ID'), torch=str(torch.__version__), inspection=args.inspection,
                    cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
                    precision=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                   matmul_precision=torch.get_float32_matmul_precision(),
                                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                                   cudnn_deterministic=torch.backends.cudnn.deterministic),
                    workflow=args.workflow, eval_seed=args.eval_seed, plan_seed=args.plan_seed,
                    target_set_hash=selection['target_set_hash'],selected_target_count=len(selection['targets']),
                    array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                    code_revision=source_revision())
    (run/'arguments.json').write_text(json.dumps(vars(args), indent=2, default=str))
    OmegaConf.save(bound, run/'model_config.yaml')
    torch.save(normalization, run/'normalization.pt')
    contexts = None
    if bound.data.get('context', False):
        from dinowm_based_evaluate import fixed_conditions
        geometry = {'env':{'dataset':{'ctx_length':180 if args.environment == 'beerpong' else 60,
                                    'ctx_frameskip':15 if args.environment == 'beerpong' else 5}},
                    'img_size':bound.img_size}
        contexts = fixed_conditions(model, OmegaConf.create(geometry), collection, normalization,
                                    ctx_selection, ctx_path)
        metadata['context_references'] = {key:str(Path(value.reference_path).resolve()) for key,value in contexts.items()}
    (run/'bindings.json').write_text(json.dumps(metadata, indent=2))
    if args.environment == 'pusht':
        if args.workflow == 'nonplanning':
            from evaluation.common import evaluation_episode_ids
            from evaluation.result_export import export_nonplanning
            from evaluation.image_metrics import LPIPS
            from evaluation.segmentation import load_segmentation_model
            ids = evaluation_episode_ids(collection, json.loads(targets_path.read_text()))
            segmentation, _ = load_segmentation_model('checkpoints/pusht/segmentation.pth', 'cuda')
            export_nonplanning(model, collection, ids, normalization, run/'nonplanning', metadata,
                                LPIPS(net_type="vgg", version="0.1").cuda().eval().requires_grad_(False), segmentation, max_batches=1 if args.inspection else None,
                                batch_size=1 if args.inspection else 8, seed=args.eval_seed, contexts_by_protocol=contexts)
        else:
            from planning.pusht_export import export_pusht_planning
            settings = dict(horizon=5, action_dim=10, num_samples=4 if args.inspection else 300,
                            topk=2 if args.inspection else 30, opt_steps=1 if args.inspection else 30, var_scale=1.)
            export_pusht_planning(model, collection, targets_path, normalization, run/'planning', metadata,
                                  settings, alpha=1.0,
                                  seed=args.plan_seed, inspection=args.inspection,
                                  contexts=None if contexts is None else contexts['planning'])
    else:
        from dinowm_based_evaluate import run_environment
        settings = argparse.Namespace(environment=args.environment, workflow=args.workflow,
                                      inspection=args.inspection, conditioning=args.method, plan_seed=args.plan_seed)
        run_environment(settings, asset, model, collection, targets_path, normalization,
                        None, run, metadata, contexts=contexts)
    if contexts is not None:
        from dinowm_based_evaluate import save_contexts
        save_contexts(run, contexts)
    if args.workflow == 'planning':
        from planning.lewm_planning_video import render_saved_planning
        model.decoder = load_decoder(asset.decoder, 'cuda')[0]
        visualization = render_saved_planning(model, collection, normalization, args.environment, args.method,
                                              run, run/'planning_decoded.mp4',
                                              contexts=None if contexts is None else contexts['planning'])
        visualization['decoder'] = str(Path(asset.decoder).resolve())
        (run/'planning_visualization.json').write_text(json.dumps(visualization, indent=2))
    print(run, flush=True)


if __name__ == '__main__':
    main()
