"""Batch and epoch loops for the bound Vanilla and TSWM objectives."""

from contextlib import nullcontext
from itertools import chain

import torch


def vanilla_optimizers(model, predictor_lr, action_encoder_lr, decoder_lr=None, ctx_encoder_lr=None):
    """Two AdamW groups and, when trained, the bound decoder Adam group."""
    model.encoder.requires_grad_(False)
    if model.decoder is not None:
        model.decoder.requires_grad_(decoder_lr is not None)
    optimizers = {
        "predictor_optimizer": torch.optim.AdamW(model.predictor.parameters(), lr=predictor_lr),
        "action_encoder_optimizer": torch.optim.AdamW(
            chain(model.action_encoder.parameters(),
                  () if model.proprio_encoder is None else model.proprio_encoder.parameters()),
            lr=action_encoder_lr,
        ),
    }
    if ctx_encoder_lr is not None:
        optimizers['ctx_encoder_optimizer'] = torch.optim.AdamW(
            model.ctx_encoder.parameters(), lr=ctx_encoder_lr)
    if decoder_lr is not None:
        optimizers["decoder_optimizer"] = torch.optim.Adam(model.decoder.parameters(), lr=decoder_lr)
    return optimizers


def tswm_optimizers(model, predictor_lr, action_encoder_lr, encoder_lr, decoder_lr):
    """Train the learned heads and decoder, with the official DINO backbone frozen."""
    optimizers = vanilla_optimizers(model, predictor_lr, action_encoder_lr, decoder_lr)
    model.encoder.requires_grad_(True)
    model.encoder.base_model.requires_grad_(False)
    optimizers['encoder_optimizer'] = torch.optim.Adam(model.encoder.parameters(), lr=encoder_lr)
    return optimizers


def optimization_step(model, optimizers, obs, action, gradient_clip, accelerator=None):
    """Take one update; retain clipped gradients for notebook inspection.

    Plain calls clear first and retain clipped gradients for inspection.
    Accelerated calls belong inside accumulate(); prepared optimizers clear
    after synchronized updates. Norm is reported only at an update boundary.
    """
    model.train()
    unwrapped = model if accelerator is None else accelerator.unwrap_model(model)
    unwrapped.encoder.train("encoder_optimizer" in optimizers)
    train_decoder = "decoder_optimizer" in optimizers
    if unwrapped.decoder is not None:
        unwrapped.decoder.train(train_decoder)
    if accelerator is None:
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
    _, _, _, loss, losses = model(obs, action, decode_images=train_decoder)
    if accelerator is None:
        loss.backward()
    else:
        accelerator.backward(loss)
    metrics = {key: value.detach() for key, value in losses.items()}
    if accelerator is None or accelerator.sync_gradients:
        if accelerator is not None:
            accelerator.unscale_gradients()
        metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True).detach()
    for optimizer in optimizers.values():
        optimizer.step()
        if accelerator is not None:
            optimizer.zero_grad(set_to_none=True)
    return metrics


def inspect_parameter_update(model, before):
    """Summarize each component's clipped gradients and actual parameter changes."""
    rows = {}
    for name in ("encoder", "decoder", "predictor", "action_encoder", "proprio_encoder", "ctx_encoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        parameters = list(module.named_parameters())
        gradients = [parameter.grad.detach().norm().cpu() for _, parameter in parameters if parameter.grad is not None]
        differences = [(parameter.detach().cpu() - before[f"{name}.{key}"]).abs()
                       for key, parameter in parameters]
        rows[name] = {
            "training": module.training,
            "trainable_numel": sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad),
            "gradient_tensors": len(gradients),
            "gradient_norm_after_clip": torch.stack(gradients).norm().item() if gradients else 0.0,
            "changed_numel": sum((difference != 0).sum().item() for difference in differences),
            "max_abs_update": max(difference.max().item() for difference in differences),
        }
    return rows


def batch_to_device(batch, device):
    """Move model inputs, including any valid-frame mask; raw state stays on CPU."""
    obs, action, _ = batch
    return {key: value.to(device) for key, value in obs.items()}, action.to(device)


def run_epoch(model, loader, device, *, optimizers=None, gradient_clip=1.0, global_step=0, logger=None, accelerator=None):
    """Train when optimizers are supplied, otherwise validate without gradients.

    Match the reference log's equal weighting of batch means, including the
    partial last batch. Accelerate reduces rank means; its prepared loaders
    repeat tail samples to keep ranks even. Keep scalar records for inspection.
    """
    training = optimizers is not None
    if not training:
        model.eval()
    if training:
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
    batches = []
    totals, counts = {}, {}
    for batch in loader:
        obs, action = batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            if training:
                context = nullcontext() if accelerator is None else accelerator.accumulate(model)
                with context:
                    metrics = optimization_step(model, optimizers, obs, action, gradient_clip, accelerator)
                if accelerator is None or accelerator.sync_gradients:
                    global_step += 1
            else:
                _, prediction, _, _, metrics = model(obs, action, decode_images=True)
        if accelerator is not None:
            metrics = {key: accelerator.reduce(value.detach(), reduction="mean") for key, value in metrics.items()}
        values = {key: value.item() for key, value in metrics.items()}
        if logger is not None:
            if training and (accelerator is None or accelerator.sync_gradients):
                logger.log({f"train/batch/{key}": value for key, value in values.items()}, global_step)
            elif not training and not batches and prediction is not None:
                image = torch.cat((obs["visual"][0, -1], prediction[0, -1]), dim=-1)
                logger.log({}, global_step, {"validation/GT_prediction": image})
        batches.append({"global_step": global_step, "batch_size": len(action), **values})
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {"metrics": {key: value / counts[key] for key, value in totals.items()},
            "batches": batches, "global_step": global_step}


def fit(model, train_loader, validation_loader, optimizers, *, device, epochs,
        gradient_clip, global_step=0, logger=None, accelerator=None, history=None, on_epoch=None):
    """Alternate complete training and validation epochs over the supplied loaders."""
    history = [] if history is None else history
    start = history[-1]["epoch"] if history else 0
    if epochs <= start:
        raise ValueError("Epoch limit must exceed the completed epoch")
    for epoch in range(start + 1, epochs + 1):
        training = run_epoch(model, train_loader, device, optimizers=optimizers,
                             gradient_clip=gradient_clip, global_step=global_step, logger=logger, accelerator=accelerator)
        global_step = training["global_step"]
        validation = run_epoch(model, validation_loader, device, global_step=global_step, logger=logger, accelerator=accelerator)
        if logger is not None:
            metrics = {f"{phase}/epoch/{key}": value for phase, result in
                       (("train", training), ("validation", validation)) for key, value in result["metrics"].items()}
            logger.log({"epoch": epoch, **metrics}, global_step)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        if on_epoch is not None:
            on_epoch(history, global_step)
    return history
