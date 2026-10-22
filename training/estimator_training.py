"""Window-weighted Context supervision with deterministic validation sampling."""
from itertools import islice

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.context import read_pusht_context


class PhysicsContextDataset(Dataset):
    def __init__(self,windows,contexts,targets,reader=read_pusht_context,validation_seed=None):
        self.windows = windows
        self.contexts = contexts
        self.targets = targets
        self.reader = reader
        self.validation_seed = validation_seed

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        position,_ = self.windows.locate(index)
        target = self.windows.episode_ids[position]
        rng = None if self.validation_seed is None else np.random.RandomState((self.validation_seed+index)%(2**32))
        record = self.contexts.sample(target,rng)
        context = self.reader(self.windows.manifest,record,self.contexts.length,
                              self.contexts.frameskip,self.windows.image_size,self.windows.normalization)
        return context,self.targets[position]


def estimator_epoch(model,loader,device,optimizer=None,gradient_clip=1.0,
                    accelerator=None,max_batches=None,global_step=0,logger=None):
    model.train(optimizer is not None)
    squared = l2 = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for context,target in islice(loader,max_batches):
            context = {k:v.to(device) for k,v in context.items()}
            target = target.to(device)
            prediction = model(context)
            loss = torch.nn.functional.mse_loss(prediction,target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward() if accelerator is None else accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                if logger is not None:
                    logger.log({'train/mse':loss.detach()},global_step)
            prediction,target = (prediction.detach(),target.detach()) if accelerator is None else accelerator.gather_for_metrics((prediction.detach(),target.detach()))
            delta = prediction-target
            squared += float(delta.square().sum())
            l2 += float(delta.norm(dim=1).sum())
            count += len(target)
    return {'mse':squared/(count*target.shape[-1]),'mean_l2':l2/count},global_step
