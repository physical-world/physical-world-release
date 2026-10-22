"""Hydra/Accelerate training entry point for the bound standalone estimator."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list,set_seed
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader,Subset

from datasets.pusht import PushTManifest,PushTDataset,pusht_normalization
from datasets.context import PushTContexts,pusht_physics_targets,read_pusht_context
from models.physics.estimator import build_estimator,load_estimator,save_estimator
from training.estimator_training import PhysicsContextDataset,estimator_epoch
from runtime.run_logging import RunLogger
from training.training_state import model_config, restore_training, save_training


def pusht_data(path,metadata,seed):
    manifest = PushTManifest(path)
    normalization = pusht_normalization(manifest)
    datasets = []
    for split,ids in manifest.split_episode_ids().items():
        windows = PushTDataset(manifest,ids,4,metadata['frameskip'],metadata['img_size'],normalization)
        contexts = PushTContexts(manifest,ids,metadata['dataset']['ctx_length'],metadata['dataset']['ctx_frameskip'])
        datasets.append(PhysicsContextDataset(windows,contexts,pusht_physics_targets(manifest,ids),
                        validation_seed=seed if split == 'validation' else None))
    return datasets


def reacher_data(path,metadata,seed):
    from datasets.reacher import ReacherEpisodes,ReacherDataset,reacher_normalization
    from datasets.context import ContextSampler,read_reacher_context
    from datasets.physics import reacher_physics
    collection = ReacherEpisodes(path)
    normalization = reacher_normalization(collection)
    raw,mean,std = reacher_physics(collection)
    datasets = []
    for split,ids in collection.split_episode_ids().items():
        windows = ReacherDataset(collection,ids,4,metadata['frameskip'],metadata['img_size'],normalization)
        contexts = ContextSampler(collection,ids,collection.dynamics[ids],metadata['dataset']['ctx_length'],metadata['dataset']['ctx_frameskip'])
        datasets.append(PhysicsContextDataset(windows,contexts,((raw-mean)/std)[ids],read_reacher_context,
                        validation_seed=seed if split=='validation' else None))
    return datasets


def fetch_slide_data(path,metadata,seed):
    from datasets.fetch_slide import FetchSlideEpisodes,FetchSlideDataset,fetch_normalization,fetch_episode_ids
    from datasets.context import ContextSampler,read_fetch_context
    from datasets.physics import fetch_slide_physics
    collection = FetchSlideEpisodes(path)
    normalization = fetch_normalization(collection)
    raw,mean,std = fetch_slide_physics(collection)
    datasets = []
    for split,ids in fetch_episode_ids(collection).items():
        windows = FetchSlideDataset(collection,ids,4,metadata['frameskip'],metadata['img_size'],normalization)
        contexts = ContextSampler(collection,ids,collection.dynamics[ids],metadata['dataset']['ctx_length'],
                                  metadata['dataset']['ctx_frameskip'],fixed_start=0,min_episode_length=40)
        datasets.append(PhysicsContextDataset(windows,contexts,((raw-mean)/std)[ids],read_fetch_context,
                        validation_seed=seed if split=='validation' else None))
    return datasets


def beerpong_data(path,metadata,seed):
    from datasets.beerpong import BeerPongEpisodes,BeerPongDataset,beerpong_normalization
    from datasets.context import ContextSampler,read_beerpong_context
    from datasets.physics import beerpong_physics
    collection = BeerPongEpisodes(path)
    normalization = beerpong_normalization(collection)
    raw,mean,std = beerpong_physics(collection)
    datasets = []
    for split,ids in collection.split_episode_ids().items():
        windows = BeerPongDataset(collection,ids,4,metadata['frameskip'],metadata['img_size'],normalization)
        contexts = ContextSampler(collection,ids,collection.dynamics[ids],metadata['dataset']['ctx_length'],
                                  metadata['dataset']['ctx_frameskip'],fixed_start=0)
        datasets.append(PhysicsContextDataset(windows,contexts,((raw-mean)/std)[ids],read_beerpong_context,
                        validation_seed=seed if split=='validation' else None))
    return datasets


def training_metadata(config, bound):
    """Derive normalization and target geometry from the selected training collection."""
    from datasets.privileged_physics import privileged_physics
    from datasets.reacher import ReacherEpisodes, reacher_normalization
    from datasets.fetch_slide import FetchSlideEpisodes, fetch_normalization
    from datasets.beerpong import BeerPongEpisodes, beerpong_normalization
    factories = dict(pusht=(PushTManifest, pusht_normalization),
                     reacher=(ReacherEpisodes, reacher_normalization),
                     fetch_slide=(FetchSlideEpisodes, fetch_normalization),
                     beerpong=(BeerPongEpisodes, beerpong_normalization))
    collection_type, normalize = factories[config.environment]
    collection = collection_type(config.manifest)
    normalization = normalize(collection)
    _, physics = privileged_physics(collection, config.environment)
    names = dict(pusht=['cog_local_normed_x','cog_local_normed_y'], reacher=['joint1_damping','joint2_damping'],
                 fetch_slide=['mass','friction','fluid_density'], beerpong=['contact_damping'])
    return dict(environment=config.environment, names=names[config.environment],
                mean=physics['physics_mean'].tolist(), std=physics['physics_std'].tolist(),
                input_normalization={k:v.tolist() for k,v in normalization.items() if k.endswith(('_mean','_std'))},
                dataset=OmegaConf.to_container(bound.env.dataset, resolve=True),
                encoder_config={k:OmegaConf.to_container(bound[k], resolve=True)
                                for k in ('ctx_img_model','ctx_pool_model')},
                img_size=bound.img_size, frameskip=bound.frameskip,
                action_emb_dim=bound.action_emb_dim, proprio_emb_dim=bound.proprio_emb_dim,
                ctx_token_dim=bound.ctx_token_dim)


@hydra.main(version_base='1.3',config_path='conf',config_name='estimator')
def main(config):
    bound = model_config(config)
    accelerator = Accelerator(mixed_precision='no',split_batches=True)
    if config.batch_size % accelerator.num_processes:
        raise ValueError('Global batch size must be divisible by the number of ranks')
    set_seed(config.seed)
    metadata = training_metadata(config, bound)
    if metadata['environment'] == 'reacher':
        datasets = reacher_data(config.manifest,metadata,config.seed)
    elif metadata['environment'] == 'fetch_slide':
        datasets = fetch_slide_data(config.manifest,metadata,config.seed)
    elif metadata['environment'] == 'beerpong':
        datasets = beerpong_data(config.manifest,metadata,config.seed)
    else:
        datasets = pusht_data(config.manifest,metadata,config.seed)
    if config.inspection:
        datasets = [Subset(d,[0,*d.windows.window_stops[:-1]][:config.batch_size]) for d in datasets]
    loaders = [DataLoader(d,batch_size=config.batch_size,shuffle=i==0,num_workers=config.num_workers,
                          generator=torch.Generator().manual_seed(config.seed))
               for i,d in enumerate(datasets)]
    model = build_estimator(metadata)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    epoch = step = 0
    history = []
    best_mse = float('inf')
    best_checkpoint = None
    if config.init_from:
        payload = load_estimator(config.init_from)
        if payload['physics_estimator']['environment'] != metadata['environment']:
            raise ValueError('Estimator initialization requires the same physical targets/environment')
        model.ctx_encoder.load_state_dict(payload['ctx_encoder'].state_dict(), strict=True)
        model.regressor.load_state_dict(payload['regressor'].state_dict(), strict=True)
    directory = [str(Path('logs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')) if accelerator.is_main_process else None]
    broadcast_object_list(directory)
    run = Path(directory[0])
    logger = None
    if accelerator.is_main_process:
        run.mkdir(parents=True,exist_ok=False)
        OmegaConf.save(config,run/'resolved.yaml',resolve=True)
        (run/'bindings.json').write_text(json.dumps({'source':config.init_from or config.resume_from,
            'manifest':str(Path(config.manifest).resolve()),'job_id':os.environ.get('SLURM_JOB_ID'),'torch':str(torch.__version__),'metadata':metadata},indent=2))
        logger = RunLogger(run,project=config.project,config=OmegaConf.to_container(config,resolve=True),mode=config.logging_mode)
    model,optimizer,train_loader,valid_loader = accelerator.prepare(model,optimizer,*loaders)
    history, step = restore_training(config, accelerator, [train_loader, valid_loader])
    epoch = history[-1]['epoch'] if history else 0
    best_mse = min((row['val_mse'] for row in history), default=float('inf'))
    if config.resume_from:
        best_checkpoint = history[-1]['best_checkpoint']
    if not config.inspection and config.epochs <= epoch:
        raise ValueError('Epoch limit must exceed the completed epoch')
    try:
        for epoch in range(epoch+1,epoch+2 if config.inspection else config.epochs+1):
            training,step = estimator_epoch(model,train_loader,accelerator.device,optimizer,config.gradient_clip,
                                             accelerator,1 if config.inspection else None,step,logger)
            validation,_ = estimator_epoch(model,valid_loader,accelerator.device,accelerator=accelerator,
                                            max_batches=1 if config.inspection else None)
            metrics = {'epoch':epoch,**{f'train_{k}':v for k,v in training.items()},**{f'val_{k}':v for k,v in validation.items()}}
            history.append(metrics)
            improved = validation['mse'] < best_mse
            best_mse = min(best_mse,validation['mse'])
            path = run/f'model_{epoch}.pth'
            if improved:
                best_checkpoint = str(path.resolve())
            metrics['best_checkpoint'] = best_checkpoint
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                saved = dict(epoch=epoch,global_cur_train_steps=step,ctx_encoder=unwrapped.ctx_encoder,
                             regressor=unwrapped.regressor,ctx_encoder_optimizer=optimizer.state_dict(),
                             wandb_run_id=logger.run.id,physics_estimator=metadata,best_val_mse=best_mse,
                             history=history,best_checkpoint=best_checkpoint)
                save_estimator(path,saved)
                logger.log({f'validation/{k}':v for k,v in validation.items()},step)
                (run/'result.json').write_text(json.dumps({'history':history,'best_checkpoint':best_checkpoint},indent=2))
                print(run,metrics,flush=True)
            save_training(run, config, bound, accelerator, [train_loader, valid_loader], history, step)
    finally:
        if logger is not None:
            logger.close()
        accelerator.end_training()


if __name__ == '__main__':
    main()
