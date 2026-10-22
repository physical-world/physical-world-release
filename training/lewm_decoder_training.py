"""A posteriori RGB decoder updates with the complete JEPA frozen."""
import torch
from models.lewm.preprocessing import prepare_info


def decoder_epoch(jepa, decoder, loader, device, *, optimizer=None, logger=None, global_step=0):
    jepa.eval().requires_grad_(False)
    decoder.train(optimizer is not None)
    records = []
    for observation, action, _ in loader:
        observation = {key:value.to(device) for key,value in observation.items()}
        with torch.no_grad():
            visual = jepa.encode(prepare_info(observation))['visual_emb']
        with torch.set_grad_enabled(optimizer is not None):
            reconstruction = decoder(visual)
            loss = (reconstruction - observation['visual']).square().mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
        if logger is not None:
            logger.log({'decoder/mse':float(loss.detach())}, global_step)
        records.append(dict(batch_size=len(visual), mse=float(loss.detach())))
    count = sum(row['batch_size'] for row in records)
    return dict(mse=sum(row['mse']*row['batch_size'] for row in records)/count,
                batches=records, global_step=global_step)
