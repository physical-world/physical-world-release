"""Single-node Accelerate entry point for bound visual world model training."""

import json
import sys
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import Subset

from runtime.checkpoints import ROOT, create_run, save_checkpoint
from datasets.beerpong import BeerPongEpisodes, BeerPongDataset, beerpong_normalization
from datasets.fetch_slide import FetchSlideEpisodes, FetchSlideDataset, fetch_normalization, fetch_episode_ids
from datasets.reacher import ReacherEpisodes, ReacherDataset, reacher_normalization
from datasets.pusht import PushTManifest, PushTDataset, pusht_normalization, pusht_dataloader
from runtime.run_logging import RunLogger
from training.training_state import model_config, restore_training, save_training
from models.construction import build_world_model, initialize_components
from training.dinowm import fit, vanilla_optimizers, tswm_optimizers


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(config):
    bound = model_config(config)
    settings = config.training
    accumulation = settings.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(cpu=config.device == "cpu", mixed_precision=settings.get("mixed_precision", "no"), split_batches=True,
                              gradient_accumulation_steps=accumulation)
    if settings.batch_size % accelerator.num_processes:
        raise ValueError("Global batch size must be divisible by the number of ranks")
    set_seed(settings.seed)
    directory = [str(create_run(config, bound)) if accelerator.is_main_process else None]
    broadcast_object_list(directory)
    run = Path(directory[0])
    runtime = {"rank": accelerator.process_index, "local_rank": accelerator.local_process_index,
               "world_size": accelerator.num_processes, "device": str(accelerator.device),
               "global_batch_size": settings.batch_size,
               "per_rank_batch_size": settings.batch_size // accelerator.num_processes,
               "effective_batch_size": settings.batch_size * accumulation, "mixed_precision": accelerator.mixed_precision,
               "gradient_accumulation_steps": accumulation, "split_batches": True, "even_batches": True,
               "torch": str(torch.__version__)}
    (run / f"rank-{accelerator.process_index}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"Run directory: {run}", file=sys.stderr, flush=True)
    print(json.dumps(runtime), flush=True)
    if bound.env.name in ("reacher_damping", "reacher"):
        manifest = ReacherEpisodes(ROOT / config.manifest)
        normalization = reacher_normalization(manifest)
        dataset_class = ReacherDataset
        environment = 'reacher'
    elif bound.env.name in ("fetch_slide_3phys", "fetch_slide"):
        manifest = FetchSlideEpisodes(ROOT / config.manifest)
        normalization = fetch_normalization(manifest)
        dataset_class = FetchSlideDataset
        environment = 'fetch_slide'
    elif bound.env.name in ("beerpong_damping", "beerpong"):
        manifest = BeerPongEpisodes(ROOT / config.manifest)
        normalization = beerpong_normalization(manifest)
        dataset_class = BeerPongDataset
        environment = 'beerpong'
    else:
        manifest = PushTManifest(ROOT / config.manifest)
        normalization = pusht_normalization(manifest)
        dataset_class = PushTDataset
        environment = 'pusht'
    splits = fetch_episode_ids(manifest) if dataset_class is FetchSlideDataset else manifest.split_episode_ids()
    physics = None
    if bound.env.dataset.get('use_cog_local_normed',False) or bound.env.dataset.get('proprio_mode') in ('damping','physics'):
        from datasets.privileged_physics import privileged_physics
        physics,physical_normalization = privileged_physics(manifest,environment)
        normalization.update(physical_normalization)
    loaders = []
    for phase in ("train", "validation"):
        episode_ids = splits[phase][:settings.batch_size] if config.inspection else splits[phase]
        dataset = dataset_class(manifest, episode_ids, bound.num_hist + bound.num_pred,
                               bound.frameskip, bound.img_size, normalization,
                               pad_last_frame=bound.env.dataset.get("pad_last_frame", False))
        windows = dataset
        if physics is not None:
            from datasets.privileged_physics import PrivilegedPhysicsDataset
            dataset = PrivilegedPhysicsDataset(windows,physics[episode_ids])
        if bound.env.dataset.get('use_ctx',False):
            from datasets.context import (ContextDataset,ContextSampler,PushTContexts,
                                          read_pusht_context,read_reacher_context,read_fetch_context,read_beerpong_context)
            geometry = bound.env.dataset
            if environment=='pusht':
                sampler = PushTContexts(manifest,splits[phase],geometry.ctx_length,geometry.ctx_frameskip)
            else:
                ids = splits[phase]
                sampler = ContextSampler(manifest,ids,manifest.dynamics[ids],geometry.ctx_length,
                                         geometry.ctx_frameskip,fixed_start=None if environment=='reacher' else 0,
                                         min_episode_length=geometry.get('ctx_min_episode_length',geometry.ctx_length))
            readers = dict(pusht=read_pusht_context,reacher=read_reacher_context,
                           fetch_slide=read_fetch_context,beerpong=read_beerpong_context)
            dataset = ContextDataset(dataset,sampler,readers[environment],bound.img_size,normalization)
        if config.inspection:
            dataset = Subset(dataset, [0, *windows.window_stops[:-1]])
        loaders.append(pusht_dataloader(dataset, settings.batch_size, phase == "train",
                                        settings.seed, settings.num_workers))
    model = build_world_model(bound, config.get('dino_weights'), loaders[0].dataset[0])
    if config.init_from:
        if bound.encoder.get('projector') == 'channel':
            from models.tswm.checkpoint import load_payload
            payload = load_payload(config.init_from)
        else:
            from models.physics.estimator import load_estimator
            payload = load_estimator(config.init_from)
        initialize_components(model, payload)
    if hasattr(model.encoder,'projector'):
        optimizers = tswm_optimizers(model,settings.predictor_lr,settings.action_encoder_lr,
                                    settings.encoder_lr,settings.decoder_lr)
    else:
        optimizers = vanilla_optimizers(model, settings.predictor_lr, settings.action_encoder_lr,
                                    settings.decoder_lr if bound.model.train_decoder else None,
                                    settings.ctx_encoder_lr if hasattr(model,'ctx_encoder') else None)
    model, train_loader, validation_loader, *prepared = accelerator.prepare(model, *loaders, *optimizers.values())
    optimizers = dict(zip(optimizers, prepared))
    logger = None
    try:
        if accelerator.is_main_process:
            logger = RunLogger(run, project=config.logging.project,
                               config=OmegaConf.to_container(config, resolve=True), mode=config.logging.mode)
        history, step = restore_training(config, accelerator, [train_loader, validation_loader])

        def completed(history, step):
            if accelerator.is_main_process:
                (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                save_checkpoint(run / f"model_{history[-1]['epoch']}.pth", accelerator.unwrap_model(model), optimizers,
                                epoch=history[-1]['epoch'], global_step=step, normalization=normalization)
            save_training(run, config, bound, accelerator, [train_loader, validation_loader], history, step)

        history = fit(model, train_loader, validation_loader, optimizers, device=accelerator.device,
                      epochs=(history[-1]["epoch"] if history else 0) + 1 if config.inspection else settings.epochs, gradient_clip=settings.gradient_clip,
                      logger=logger, accelerator=accelerator, history=history, global_step=step, on_epoch=completed)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            save_checkpoint(run / "final.pth", accelerator.unwrap_model(model), optimizers,
                            epoch=history[-1]["epoch"], global_step=history[-1]["train"]["global_step"],
                            normalization=normalization)
    finally:
        if logger is not None:
            logger.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
