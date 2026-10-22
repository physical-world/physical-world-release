"""Independent CLS decoder fine-tuning with a frozen paper JEPA."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
import hydra
from omegaconf import OmegaConf
import torch

from runtime.checkpoints import ROOT
from datasets.pusht import pusht_dataloader
from datasets.lewm import build_datasets
from training.lewm_decoder_training import decoder_epoch
from models.lewm.checkpoint import load_jepa
from models.lewm.decoder import CLSImageDecoder, load_decoder, save_decoder
from training.training_state import validate_mode, restore_training, save_training
from runtime.run_logging import RunLogger


@hydra.main(version_base='1.3', config_path='conf', config_name='train_lewm_decoder')
def main(config):
    validate_mode(config)
    accelerator = Accelerator(cpu=config.device == "cpu", mixed_precision="no")
    if accelerator.num_processes != 1:
        raise ValueError("Decoder training currently supports one process")
    set_seed(config.seed)
    model, bound = load_jepa(config.assets.checkpoint, config.assets.checkpoint_config, accelerator.device)
    bound.data.data_path = str(Path(config.assets.manifest).resolve())
    decoder = CLSImageDecoder(**config.decoder).to(accelerator.device)
    payload = {'decoder_config': OmegaConf.to_container(config.decoder, resolve=True)}
    if config.init_from:
        loaded, _ = load_decoder(config.init_from)
        decoder.load_state_dict(loaded.state_dict(), strict=True)
    decoder_data = OmegaConf.create(OmegaConf.to_container(bound, resolve=False))
    decoder_data.data.context = False
    _, normalization, datasets = build_datasets(config.assets, decoder_data, config.inspection)
    loaders = [pusht_dataloader(datasets[phase], config.batch_size, phase == 'train',
                                config.seed, config.num_workers) for phase in ('train', 'validation')]
    run = ROOT/'logs'/'lewm_decoder'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, run/'resolved.yaml', resolve=True)
    OmegaConf.save(bound, run/'model_config.yaml')
    torch.save(normalization, run/'normalization.pt')
    bindings = {key:str(Path(value).resolve()) for key,value in config.assets.items()}
    bindings.update(job_id=os.environ.get('SLURM_JOB_ID'), torch=str(torch.__version__),
                    frozen_jepa=True, initialization='resume' if config.resume_from else ('weights' if config.init_from else 'random'))
    (run/'bindings.json').write_text(json.dumps(bindings, indent=2))
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.lr)
    logger = RunLogger(run, project=config.logging.project,
                       config=OmegaConf.to_container(config, resolve=True), mode=config.logging.mode)
    decoder, optimizer, *loaders = accelerator.prepare(decoder, optimizer, *loaders)
    history, step = restore_training(config, accelerator, loaders)
    try:
        start = history[-1]["epoch"] if history else 0
        epochs = start + 1 if config.inspection else config.epochs
        if epochs <= start:
            raise ValueError("Epoch limit must exceed the completed epoch")
        for epoch in range(start + 1, epochs + 1):
            training = decoder_epoch(model, decoder, loaders[0], accelerator.device, optimizer=optimizer,
                                     logger=logger, global_step=step)
            step = training['global_step']
            validation = decoder_epoch(model, decoder, loaders[1], accelerator.device, logger=logger, global_step=step)
            history.append(dict(epoch=epoch, train=training, validation=validation))
            record = dict(payload, epoch=epoch, train_mse=training['mse'], val_mse=validation['mse'],
                          lr=config.lr, batch_size=config.batch_size)
            save_decoder(run/f'decoder_epoch_{epoch}.pt', decoder, record)
            with (run/f'optimizer_epoch_{epoch}.pt').open('xb') as stream:
                torch.save(dict(optimizer=optimizer.state_dict(), epoch=epoch, global_step=step), stream)
            (run/'history.json').write_text(json.dumps(history, indent=2))
            save_training(run, config, bound, accelerator, loaders, history, step)
    finally:
        logger.close()
    print(run, flush=True)


if __name__ == '__main__':
    main()
