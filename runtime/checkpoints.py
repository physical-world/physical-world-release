"""Run provenance and the bound visual world model full-component save formats."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from omegaconf import OmegaConf
import torch

from models.dino import DINOV2_REPO, WEIGHTS_URL, dino_weights_path


ROOT = Path(__file__).resolve().parents[1]


def source_revision():
    """Git commit of this checkout, or None for a source download without git metadata."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def create_run(config, bound=None):
    """Create a new run under logs; record resolved settings and source bindings."""
    run = ROOT / "logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, run / "resolved.yaml", resolve=True)
    assets = {}
    for key in ("checkpoint", "checkpoint_config", "manifest", "dino_weights"):
        if bound is not None and key in ("checkpoint", "checkpoint_config"):
            continue
        if key == "dino_weights":
            path = (ROOT / dino_weights_path(config.get(key))).resolve()
            if config.get(key) is None:
                # The pretrained entry point fills this cache after run creation.
                assets[key] = {"path": str(path), "repository": DINOV2_REPO, "url": WEIGHTS_URL}
                continue
        else:
            path = (ROOT / config[key]).resolve()
        stat = path.stat()
        assets[key] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if bound is None:
        shutil.copy2(assets["checkpoint_config"]["path"], run / "model_config.yaml")
    else:
        OmegaConf.save(bound, run / "model_config.yaml", resolve=True)
    record = {"assets": assets, "job_id": os.environ.get("SLURM_JOB_ID"),
              "host": platform.node(), "python": platform.python_version(), "torch": str(torch.__version__),
              "dino_revision": DINOV2_REPO}
    (run / "bindings.json").write_text(json.dumps(record, indent=2) + "\n")
    return run


def save_checkpoint(path, model, optimizers, *, epoch, global_step, normalization):
    """Write a new trusted full-object checkpoint, never replacing existing files.

    Epoch is the number of completed epochs; global_step counts optimizer updates.
    Keep normalization outside the bound paper payload. Frozen DINO is bound
    separately, as in the original file. Caller passes an unwrapped world model.
    The checkpoint lives directly in the directory returned by create_run.
    """
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "logs").resolve()):
        raise ValueError("New checkpoints must be written under logs/")
    payload = {"epoch": epoch, "global_cur_train_steps": global_step,
               "action_encoder": model.action_encoder, "proprio_encoder": model.proprio_encoder,
               "predictor": model.predictor,
               "predictor_optimizer": optimizers["predictor_optimizer"].state_dict(),
               "action_encoder_optimizer": optimizers["action_encoder_optimizer"].state_dict(),
               "decoder": model.decoder}
    if "decoder_optimizer" in optimizers:
        payload["decoder_optimizer"] = optimizers["decoder_optimizer"].state_dict()
    if hasattr(model.encoder, 'projector'):
        from models.tswm.checkpoint import save_payload
        payload = {'epoch':epoch, 'data_provenance':model.data_provenance,
                   'encoder':model.encoder, 'predictor':model.predictor, 'decoder':model.decoder,
                   'action_encoder':model.action_encoder, 'proprio_encoder':model.proprio_encoder}
        for name in ('encoder_optimizer','predictor_optimizer','decoder_optimizer','action_encoder_optimizer'):
            payload[name] = optimizers[name].state_dict()
        payload['global_cur_train_steps'] = global_step
        with path.open('xb') as stream:
            save_payload(stream, payload)
    elif hasattr(model, 'ctx_encoder'):
        from models.physics.estimator import save_estimator
        payload['ctx_encoder'] = model.ctx_encoder
        payload['ctx_encoder_optimizer'] = optimizers['ctx_encoder_optimizer'].state_dict()
        save_estimator(path, payload)
    else:
        with path.open("xb") as stream:
            torch.save(payload, stream)
    with path.with_suffix(".normalization.pt").open("xb") as stream:
        torch.save(normalization, stream)
    binding = {"checkpoint": path.name, "normalization": path.with_suffix(".normalization.pt").name,
               "model_config": "model_config.yaml", "run_config": "resolved.yaml", "provenance": "bindings.json"}
    with path.with_suffix(".json").open("x") as stream:
        json.dump(binding, stream, indent=2)


def restore_optimizers(optimizers, payload):
    """Restore the saved optimizer groups onto their already-constructed parameters."""
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload[name])


def load_world_model(checkpoint, model_config, dino_weights=None, *, device='cpu', load_decoder=True):
    """Restore bound DINO-WM or TSWM components and their explicit model."""
    from models.dino import FrozenDino
    from models.visual_world_model import VWorldModel
    bound = OmegaConf.load(model_config)
    if bound.encoder.get('projector') == 'channel':
        from models.tswm.checkpoint import load_payload
        payload = load_payload(checkpoint)
        from models.tswm.world_model import VWorldModel as TSWM
        model = TSWM(bound.img_size,payload['encoder'],payload['proprio_encoder'],
                            payload['action_encoder'],predictor=payload['predictor'],
                            decoder=payload['decoder'] if load_decoder else None,
                            num_hist=bound.num_hist,num_pred=bound.num_pred,
                            straighten_scale=float(bound.training.straighten.removeprefix('aggcos')))
        model.encoder.base_model.requires_grad_(False)
        model.data_provenance = payload['data_provenance']
        return model.to(device),payload,bound
    from models.physics.estimator import load_estimator
    payload = load_estimator(checkpoint)
    model_class = VWorldModel
    options = {}
    if 'ctx_encoder' in payload:
        from models.pusht_context.visual_wm_ctx import VWorldModel_Context
        model_class = VWorldModel_Context
        options['ctx_encoder'] = payload['ctx_encoder']
    elif 'prefix_context_frames' in bound:
        from models.visual_world_model import VWorldModel_PrefixContext
        model_class = VWorldModel_PrefixContext
        options['prefix_context_frames'] = bound.prefix_context_frames
    model = model_class(bound.img_size,FrozenDino(dino_weights),payload['proprio_encoder'],
                        payload['action_encoder'],predictor=payload['predictor'],
                        decoder=payload['decoder'] if load_decoder else None,
                        num_hist=bound.num_hist,num_pred=bound.num_pred,
                        proprio_loss_weight=bound.model.proprio_loss_weight, **options)
    return model.to(device),payload,bound
