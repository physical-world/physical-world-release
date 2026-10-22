"""Independently runnable bound DINO-WM non-planning and planning workflows."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from runtime.checkpoints import load_world_model
from models.dino import dino_weights_path
from datasets.pusht import pusht_normalization
from datasets.evaluation_targets import add_sampling_arguments, load_collection, prepare_targets, check_target_geometry
from datasets.evaluation_contexts import (prepare_contexts, context_reader, context_records,
                                         check_geometry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow',choices=['nonplanning','planning'],required=True)
    parser.add_argument('--inspection',action='store_true')
    parser.add_argument('--inspection-inputs',type=Path)
    parser.add_argument('--conditioning',choices=['vanilla','tswm','privileged_physics','estimated','context','prefix_context'],default='privileged_physics')
    parser.add_argument('--environment',choices=['pusht','reacher','fetch_slide','beerpong'],default='pusht')
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--plan-seed", type=int, default=None)
    add_sampling_arguments(parser)
    args = parser.parse_args()
    if args.plan_seed is None:
        args.plan_seed = 99 if args.environment == "pusht" else 42
    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    method = 'estimated_physics' if args.conditioning=='estimated' else args.conditioning
    asset = OmegaConf.load(f'conf/assets/{args.environment}_{method}.yaml')
    run = Path(f'logs/{args.conditioning}')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    manifest = load_collection(args.environment, asset.manifest)
    targets, selection, sampling = prepare_targets(args, manifest, args.environment, args.workflow, run)
    configuration = OmegaConf.load(asset.checkpoint_config)
    check_target_geometry(selection,configuration.frameskip)
    needs_context = args.conditioning in ('estimated','context','prefix_context')
    ctx_path, ctx_selection = prepare_contexts(args, manifest, selection, sampling, args.workflow, run) if needs_context else (None,None)
    if args.inspection_inputs is not None and not args.inspection:
        parser.error('--inspection-inputs requires --inspection')
    preset = torch.load(args.inspection_inputs,map_location='cpu',weights_only=True) if args.inspection_inputs else None
    if preset is not None and (preset['environment']!=args.environment or
                               Path(preset['manifest']).resolve()!=Path(asset.manifest).resolve()):
        raise ValueError('Inspection statistics and evaluation assets differ')
    model,_,bound = load_world_model(asset.checkpoint,asset.checkpoint_config,asset.get('dino_weights'),device='cuda',
                                    load_decoder=args.workflow=='nonplanning')
    model.eval().requires_grad_(False)
    if args.environment=='reacher':
        from datasets.reacher import reacher_normalization
        normalization = reacher_normalization(manifest) if preset is None else preset['normalization']
    elif args.environment=='fetch_slide':
        from datasets.fetch_slide import fetch_normalization
        normalization = fetch_normalization(manifest) if preset is None else preset['normalization']
    elif args.environment=='beerpong':
        from datasets.beerpong import beerpong_normalization
        normalization = beerpong_normalization(manifest) if preset is None else preset['normalization']
    else:
        normalization = pusht_normalization(manifest) if preset is None else preset['normalization']
    from datasets.privileged_physics import privileged_physics
    physics = None
    if args.conditioning in ('privileged_physics','estimated') and preset is None:
        physics,physical_normalization = privileged_physics(manifest,args.environment)
        normalization.update(physical_normalization)
    elif args.conditioning in ('privileged_physics','estimated'):
        physics = preset['values']
    providers = None
    if args.conditioning=='estimated':
        providers = estimated_conditions(asset,manifest,normalization,ctx_selection,ctx_path)
        if args.workflow=='planning':
            physics = providers['planning']
    contexts = fixed_conditions(model,bound,manifest,normalization,ctx_selection,ctx_path) if needs_context and args.conditioning!='estimated' else None
    metadata = dict(checkpoint=str(Path(asset.checkpoint).resolve()),configuration=str(Path(asset.checkpoint_config).resolve()),
                    targets=str(targets.resolve()),manifest=str(Path(asset.manifest).resolve()),
                    environment=args.environment,workflow=args.workflow,inspection=args.inspection,
                    eval_seed=args.eval_seed,plan_seed=args.plan_seed,
                    target_set_hash=selection['target_set_hash'],selected_target_count=len(selection['targets']),
                    array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                    array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                    job_id=os.environ.get('SLURM_JOB_ID'),torch=str(torch.__version__),privileged_physics=args.conditioning=='privileged_physics',conditioning=args.conditioning,
                    cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
                    precision=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                   matmul_precision=torch.get_float32_matmul_precision(),
                                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                                   cudnn_deterministic=torch.backends.cudnn.deterministic),
                    dino_weights=str(dino_weights_path(asset.get('dino_weights')).resolve()),
                    dino_revision='b8931f7bf91576930313be2c6d6af376033b35f0',
                    candidate_evaluation='wm_ts' if method=='tswm' else 'wm_dino',
                    planning_latent_cost='straightened_dino_latent' if method=='tswm' else 'dino_latent')
    if args.inspection_inputs is not None:
        metadata['inspection_inputs'] = str(args.inspection_inputs.resolve())
    if providers is not None:
        metadata.update(estimator=str(Path(asset.estimator).resolve()),
                        context_references={key:str(Path(value.reference_path).resolve()) for key,value in providers.items()})
    if contexts is not None:
        metadata['context_references'] = {key:str(Path(value.reference_path).resolve()) for key,value in contexts.items()}
    (run/'bindings.json').write_text(json.dumps(metadata,indent=2))
    (run/'arguments.json').write_text(json.dumps(vars(args),indent=2,default=str))
    OmegaConf.save(bound,run/'model_config.yaml')
    torch.save(normalization,run/'normalization.pt')
    if args.environment!='pusht':
        run_environment(args,asset,model,manifest,targets,normalization,physics,run,metadata,providers,contexts)
        save_estimates(run,providers)
        save_contexts(run,contexts)
        print(run,flush=True)
        return
    if args.workflow=='nonplanning':
        from evaluation.common import evaluation_episode_ids
        from evaluation.result_export import export_nonplanning
        from evaluation.image_metrics import LPIPS
        from evaluation.segmentation import load_segmentation_model
        selection = json.loads(targets.read_text())
        ids = evaluation_episode_ids(manifest,selection)
        segmentation,_ = load_segmentation_model('checkpoints/pusht/segmentation.pth','cuda')
        evaluation = OmegaConf.load(asset.get('evaluation_config',f'checkpoints/pusht/{method}/next_frame_eval_config.yaml')).eval
        export_nonplanning(model,manifest,ids,normalization,run/'nonplanning',metadata,LPIPS(net_type="vgg", version="0.1").cuda().eval().requires_grad_(False),segmentation,
                            batch_size=1 if args.inspection else 8,max_batches=1 if args.inspection else None,
                            physics=physics,
                            windows_per_episode=evaluation.fixed_horizon_windows_per_episode,
                            seed=args.eval_seed,init_context=evaluation.rollout_init_ctx,physics_by_protocol=providers,contexts_by_protocol=contexts)
    else:
        from planning.pusht_export import export_pusht_planning
        plan = OmegaConf.load(asset.get('planning_config',f'checkpoints/pusht/{method}/planning_eval_config.yaml')).plan
        settings = dict(horizon=5,action_dim=10,num_samples=plan.planner.sub_planner.num_samples,
                        topk=plan.planner.sub_planner.topk,opt_steps=plan.planner.sub_planner.opt_steps,var_scale=1.0)
        if args.inspection:
            settings.update(num_samples=4,topk=2,opt_steps=1)
        export_pusht_planning(model,manifest,targets,normalization,run/'planning',metadata,settings,
                              alpha=plan.objective.alpha,seed=args.plan_seed,inspection=args.inspection,condition_on_physics=args.conditioning in ('privileged_physics','estimated'),
                              physics=physics if providers is not None else None,
                              contexts=None if contexts is None else contexts['planning'])
    save_estimates(run,providers)
    save_contexts(run,contexts)
    print(run,flush=True)


def run_environment(args,asset,model,collection,target_path,normalization,physics,run,metadata,providers=None,contexts=None):
    if args.environment=='reacher':
        from evaluation.reacher_evaluation import export_reacher as export_nonplanning
        from planning.reacher_export import export_reacher_planning as export_planning
    elif args.environment=='beerpong':
        from evaluation.beerpong_evaluation import export_beerpong as export_nonplanning
        from planning.beerpong_export import export_beerpong_planning as export_planning
    else:
        from evaluation.fetch_slide_evaluation import export_fetch as export_nonplanning
        from planning.fetch_slide_export import export_fetch_planning as export_planning
    method = 'estimated_physics' if args.conditioning=='estimated' else args.conditioning
    targets = json.loads(target_path.read_text())['targets']
    if args.workflow=='nonplanning':
        import zarr
        from evaluation.image_metrics import LPIPS
        from evaluation.segmentation import load_segmentation_model
        masks = zarr.open_group(str(Path(asset.manifest).with_name(Path(asset.manifest).stem+'_mask.zarr')),mode='r')
        segmentation,_ = load_segmentation_model(f'checkpoints/{args.environment}/segmentation.pth','cuda')
        export_nonplanning(model,collection,targets,normalization,masks,run/'nonplanning',metadata,
                           LPIPS(net_type="vgg", version="0.1").cuda().eval().requires_grad_(False),segmentation,batch_size=1 if args.inspection else 8,
                           max_batches=1 if args.inspection else None,physics=physics,physics_by_protocol=providers,contexts_by_protocol=contexts)
    else:
        settings = OmegaConf.to_container(OmegaConf.load(f'conf/planning/{args.environment}.yaml'))
        settings.pop('seed')
        if args.inspection:
            settings.update(num_samples=4,topk=2,opt_steps=1)
        export_planning(model,collection,targets,normalization,run/'planning',metadata,settings,
                        seed=args.plan_seed,inspection=args.inspection,physics=physics,
                        contexts=None if contexts is None else contexts['planning'])


def estimated_conditions(asset,collection,normalization,assignments,reference_path):
    from models.physics.estimator import load_physics_estimator
    from models.physics.estimated_physics import EstimatedPhysics
    estimator,metadata,_ = load_physics_estimator(asset.estimator,'cuda')
    check_geometry(assignments['geometry'],metadata['dataset'])
    read_context = context_reader(assignments['environment'])
    providers = {}
    for protocol in assignments['protocols']:
        records = context_records(assignments,protocol)
        provider = EstimatedPhysics(estimator,metadata,collection,records,read_context,
                                    normalization['physics_mean'],normalization['physics_std'])
        provider.reference_path = reference_path
        providers[protocol] = provider
    return providers


def save_estimates(run,providers):
    if providers is not None:
        with (run/'estimated_physics.json').open('x') as stream:
            json.dump({key:value.estimates for key,value in providers.items()},stream,indent=2)



def fixed_conditions(model,bound,collection,normalization,assignments,reference_path):
    from datasets.context import FixedContexts
    geometry = assignments['geometry']
    check_geometry(geometry,bound.env.dataset)
    reader = context_reader(assignments['environment'])
    providers = {}
    for protocol in assignments['protocols']:
        records = context_records(assignments,protocol)
        provider = FixedContexts(model,collection,records,reader,geometry,bound.img_size,normalization)
        provider.reference_path = reference_path
        providers[protocol] = provider
    return providers


def save_contexts(run,providers):
    if providers is not None:
        with (run/'contexts.json').open('x') as stream:
            json.dump({key:{'assignments':value.records,'encoded_shapes':{episode:list(row['ctx_emb'].shape)
                       for episode,row in value.cache.items()}} for key,value in providers.items()},stream,indent=2)


if __name__=='__main__':
    main()
