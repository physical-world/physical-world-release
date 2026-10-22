"""LeWM objectives and bounded or full training with differentiable rank statistics."""
import torch
from torch import nn
from torch.distributed.nn.functional import all_gather

from models.lewm.preprocessing import prepare_info
from models.lewm.sigreg import SIGReg


class LeWMObjective(nn.Module):
    def __init__(self, model, *, history_size=3, num_preds=1, sigreg_weight=0.09,
                 knots=17, num_proj=1024, temporally_centered=False):
        super().__init__()
        self.model = model
        self.sigreg = SIGReg(knots, num_proj)
        self.history_size = history_size
        self.num_preds = num_preds
        self.sigreg_weight = sigreg_weight
        self.temporally_centered = temporally_centered

    def forward(self, observation, action):
        info = self.model.encode(prepare_info(observation, action))
        predicted = self.model.predict(info['emb'][:, :self.history_size],
                                       info['act_emb'][:, :self.history_size], info.get('ctx_emb'))
        losses = self.model.prediction_losses(predicted, info['emb'][:, self.num_preds:])
        visual = info['visual_emb']
        if self.temporally_centered:
            from models.lewm.temporal import temporal_residual, temporal_statistics
            losses.update(temporal_statistics(visual))
            visual = temporal_residual(visual)
        if torch.distributed.is_initialized():
            visual = torch.cat(all_gather(visual), dim=0)
        losses['sigreg_loss'] = self.sigreg(visual.transpose(0, 1))
        losses['loss'] = losses['pred_loss'] + self.sigreg_weight * losses['sigreg_loss']
        return losses


def run_epoch(objective, loader, device, *, optimizer=None, accelerator=None,
              gradient_clip=1.0, logger=None, global_step=0, scheduler=None, accumulation_steps=1):
    training = optimizer is not None
    objective.train(training)
    records = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (observation, action, _) in enumerate(loader):
        observation = {key:value.to(device) for key,value in observation.items()}
        action = action.to(device)
        with torch.set_grad_enabled(training):
            losses = objective(observation, action)
            if training:
                if accelerator is None:
                    losses['loss'].backward()
                else:
                    accelerator.backward(losses['loss'])
                used_lr = optimizer.param_groups[0]['lr']
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                    if accumulation_steps > 1:
                        for parameter in objective.parameters():
                            if parameter.grad is not None:
                                group_size = min(accumulation_steps, len(loader) - (batch_index // accumulation_steps) * accumulation_steps)
                                parameter.grad.div_(group_size)
                    norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), gradient_clip,
                                                          error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                    losses['grad_norm'] = norm
                losses['learning_rate'] = losses['loss'].new_tensor(used_lr)
                losses['next_learning_rate'] = losses['loss'].new_tensor(optimizer.param_groups[0]['lr'])
        if accelerator is not None:
            losses = {key:accelerator.reduce(value.detach(), reduction='mean') for key,value in losses.items()}
        metrics = {key:float(value.detach()) for key,value in losses.items()}
        if logger is not None:
            phase = 'train' if training else 'validation'
            logger.log({f'{phase}/batch/{key}':value for key,value in metrics.items()}, global_step)
        records.append(dict(batch_size=len(action), global_step=global_step, **metrics))
    summary = {key:sum(record[key] * record['batch_size'] for record in records if key in record)
               /sum(record['batch_size'] for record in records if key in record)
               for key in metrics}
    return dict(metrics=summary, batches=records, global_step=global_step)
