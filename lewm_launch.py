"""Explicit Accelerate entry point for learnable LeWM training and raw weight saving."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
import hydra
from omegaconf import OmegaConf
import torch

from runtime.checkpoints import ROOT
from datasets.pusht import pusht_dataloader
from datasets.lewm import build_datasets
from training.lewm_training import LeWMObjective, run_epoch
from models.lewm.checkpoint import save_weights
from models.lewm.schedule import warmup_cosine
from runtime.run_logging import RunLogger
from models.lewm.construction import build_model
from training.training_state import model_config, restore_training, save_training


@hydra.main(version_base='1.3', config_path='conf', config_name='train_lewm')
def main(config):
    bound = model_config(config, config.assets)
    accelerator = Accelerator(mixed_precision='bf16', split_batches=True)
    set_seed(config.seed)
    asset = config.assets
    model = build_model(bound)
    if config.init_from:
        model.load_state_dict(torch.load(config.init_from, map_location="cpu", weights_only=True), strict=True)
    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    collection, normalization, datasets = build_datasets(asset, bound, config.inspection)
    objective = LeWMObjective(model, history_size=bound.history_size, num_preds=bound.num_preds,
                              sigreg_weight=bound.loss.sigreg.weight, **bound.loss.sigreg.kwargs,
                              temporally_centered=bound.loss.sigreg.get('temporally_centered', False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=bound.optimizer.lr, weight_decay=bound.optimizer.weight_decay)
    paper_batch_size = bound.loader.batch_size * bound.trainer.devices
    accumulation_steps = bound.trainer.get('accumulate_grad_batches', 1)
    batch_size = paper_batch_size if config.batch_size is None else config.batch_size
    loaders = [pusht_dataloader(datasets[phase], batch_size, phase == 'train', config.seed,
                                config.num_workers, drop_last=phase == 'train') for phase in ('train', 'validation')]
    objective, optimizer, train, validation = accelerator.prepare(objective, optimizer, *loaders)
    total_updates = ((len(train) + accumulation_steps - 1) // accumulation_steps) * bound.trainer.max_epochs
    scheduler = warmup_cosine(optimizer.optimizer, total_updates)
    accelerator.register_for_checkpointing(scheduler)
    directory = [str(ROOT/'logs'/'lewm_training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                 if accelerator.is_main_process else None]
    broadcast_object_list(directory)
    run = Path(directory[0])
    if accelerator.is_main_process:
        run.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, run/'resolved.yaml', resolve=True)
        OmegaConf.save(bound, run/'model_config.yaml')
        torch.save(normalization, run/'normalization.pt')
        bindings = {key:str(Path(value).resolve()) for key,value in asset.items()}
        bindings.update(job_id=os.environ.get('SLURM_JOB_ID'), world_size=accelerator.num_processes,
                        global_batch_size=batch_size, paper_global_batch_size=paper_batch_size,
                        optimizer_update_frequency=accumulation_steps,
                        effective_batch_size=batch_size * accumulation_steps,
                        paper_effective_batch_size=paper_batch_size * accumulation_steps,
                        scheduler='native LambdaLR; paper update-based warmup/cosine',
                        scheduler_total_updates=total_updates, scheduler_warmup_updates=max(1, int(.01*total_updates)),
                        mixed_precision='bf16', torch=str(torch.__version__))
        (run/'bindings.json').write_text(json.dumps(bindings, indent=2))
    accelerator.wait_for_everyone()
    logger = RunLogger(run, project=config.logging.project,
                       config=OmegaConf.to_container(config, resolve=True), mode=config.logging.mode) if accelerator.is_main_process else None
    history, step = restore_training(config, accelerator, [train, validation])
    try:
        start = history[-1]["epoch"] if history else 0
        epochs = start + 2 if config.inspection else bound.trainer.max_epochs
        if epochs <= start:
            raise ValueError("Epoch limit must exceed the completed epoch")
        for epoch in range(start + 1, epochs + 1):
            training = run_epoch(objective, train, accelerator.device, optimizer=optimizer, accelerator=accelerator,
                                  gradient_clip=bound.trainer.gradient_clip_val, logger=logger, global_step=step,
                                  scheduler=scheduler, accumulation_steps=accumulation_steps)
            step = training['global_step']
            evaluation = run_epoch(objective, validation, accelerator.device, accelerator=accelerator,
                                   logger=logger, global_step=step)
            history.append(dict(epoch=epoch, train=training, validation=evaluation))
            if accelerator.is_main_process:
                save_weights(run/f'weights_epoch_{epoch}.pt', accelerator.unwrap_model(objective).model)
                with (run/f'optimizer_epoch_{epoch}.pt').open('xb') as stream:
                    torch.save(dict(optimizer=optimizer.state_dict(), epoch=epoch, global_step=step, scheduler=scheduler.state_dict()), stream)
                (run/'history.json').write_text(json.dumps(history, indent=2))
            save_training(run, config, bound, accelerator, [train, validation], history, step)
    finally:
        if logger is not None:
            logger.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(run, flush=True)


if __name__ == '__main__':
    main()
