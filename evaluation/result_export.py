"""PushT non-planning scores and original paper result exports."""

import csv
import json
from pathlib import Path
from statistics import mean

from runtime.checkpoints import ROOT

import torch

from evaluation.common import evaluation_samples, read_evaluation_sample
from evaluation.image_metrics import image_metrics
from evaluation.segmentation import iou_counts, iou_from_counts, read_evaluation_masks, remap_masks, segment_images


def segmentation_scores(predicted, target):
    return {f"segm_iou_{'all' if key == 'overall' else key}": value
            for key, value in iou_from_counts(iou_counts(predicted, target)).items()}


@torch.no_grad()
def predict_batch(model, samples):
    """Shared model execution; environment-specific metric reductions follow."""
    device = next(model.parameters()).device
    obs = {key: torch.stack([s["observation"][key] for s in samples]).to(device)
           for key in samples[0]["observation"]}
    actions = torch.stack([s["action"] for s in samples]).to(device)
    sample = samples[0]
    indices = sample["target_indices"]
    target = torch.stack([obs["visual"][i, s["target_indices"]] for i, s in enumerate(samples)])
    latent = truth_z = reconstructed = losses = None
    if sample["protocol"] == "teacher_forced":
        _, predicted, reconstructed, _, losses = model(obs, actions, decode_images=True)
        predicted = torch.stack([predicted[i, s["target_indices"] - 1] for i, s in enumerate(samples)])
    else:
        initial = len(sample["initial_indices"])
        latent, _ = model.rollout({key: value if key.startswith('ctx_') else value[:, :initial] for key, value in obs.items()}, actions)
        target_chunks = [model.encode_obs({key: value[:, start:start + 8] for key, value in obs.items()})
                         for start in range(0, obs["visual"].shape[1], 8)]
        truth_z = {key: torch.cat([chunk[key] for chunk in target_chunks], 1)[:, indices] for key in latent}
        latent = {key: value[:, indices] for key, value in latent.items()}
        predicted = torch.cat([model.decode_obs({key: value[:, start:start + 8] for key, value in latent.items()})[0]["visual"]
                               for start in range(0, len(indices), 8)], 1)
    return dict(observation=obs, target=target, predicted=predicted, reconstructed=reconstructed,
                losses=losses, latent=latent, target_latent=truth_z)


def evaluation_images(data, samples):
    obs = data["observation"]["visual"]
    if samples[0]["protocol"] == "teacher_forced":
        initial = torch.stack([obs[i, s["target_indices"][:1] - 1] for i, s in enumerate(samples)])
    else:
        initial = obs[:, :len(samples[0]["initial_indices"])]
    return dict(predicted=data["predicted"].cpu(), target=data["target"].cpu(), initial=initial.cpu())


@torch.no_grad()
def score_batch(model, samples, lpips, segmentation):
    """PushT teacher batches pool frames; rollouts score one step at a time."""
    data = predict_batch(model, samples)
    predicted, target = data["predicted"], data["target"]
    masks = remap_masks(torch.stack([s["raw_masks"] for s in samples]).to(predicted.device))
    sample = samples[0]
    if sample["protocol"] == "teacher_forced":
        values = {key: float(value) for key, value in data["losses"].items()}
        for name, images, truth in (("pred", predicted, target), ("reconstructed", data["reconstructed"], data["observation"]["visual"])):
            values.update({f"img_{key}_{name}": value for key, value in image_metrics(images, truth, lpips).items()})
        values.update(segmentation_scores(segment_images(segmentation, predicted), masks))
    else:
        latent, truth_z = data["latent"], data["target_latent"]
        predicted_masks = torch.cat([segment_images(segmentation, predicted[:, start:start + 8])
                                     for start in range(0, predicted.shape[1], 8)], 1)
        steps = []
        for step in range(predicted.shape[1]):
            row = {f"z_{key}_err": float((latent[key][:, step] - truth_z[key][:, step]).square().mean()) for key in latent}
            row.update({f"img_{key}": value for key, value in image_metrics(predicted[:, step], target[:, step], lpips).items()})
            row.update(segmentation_scores(predicted_masks[:, step], masks[:, step]))
            steps.append(row)
        values = steps
        if sample["protocol"] == "open_loop":
            tag = f"rollout_ic{len(sample['initial_indices'])}"
            values = {f"{tag}_{key}_pred": mean(row[key] for row in steps) for key in steps[0] if key.startswith("img_")}
            for key in latent:
                error = (latent[key] - truth_z[key]).square()
                values[f"{tag}_z_{key}_err_pred"] = float(error.mean())
                values[f"{tag}_z_{key}_err_last"] = float(error[:, -1].mean())
            values.update({f"{tag}_{key}": value for key, value in segmentation_scores(predicted_masks, masks).items()})
    return values, {**evaluation_images(data, samples), "masks": masks.cpu()}


def summarize_results(protocol, records, metadata, *, batch_size=8, num_hist=3,
                      frameskip=5, fixed_horizon=5, windows_per_episode=2, seed=42, init_context=3, physics_by_protocol=None):
    """Preserve equal-batch / equal-window / equal-episode paper reductions."""
    summary = dict(metadata)
    if protocol == "teacher_forced":
        identity = {"batch", "episode", "source_episode", "variable_prefix", "input_frames", "prediction_frames"}
        summary.update(val_batch_size=batch_size, n_batches=len(records), per_batch=records)
    elif protocol == "fixed_horizon":
        summary.update(horizon_wm_tokens=fixed_horizon, raw_action_horizon=fixed_horizon * frameskip,
                       initial_observation_frames=1, n_windows=len(records), n_batches=len(records),
                       windows_per_episode=1 if windows_per_episode is None else windows_per_episode,
                       window_sampling_seed=None if windows_per_episode is None else seed, per_window=records)
        for key in records[0]["per_step"][0]:
            values = [mean(row["per_step"][step][key] for row in records) for step in range(fixed_horizon)]
            prefix = f"fixed_h{fixed_horizon}_{key}"
            summary.update({f"{prefix}_step{step}": value for step, value in enumerate(values, 1)})
            summary.update({f"{prefix}_mean": mean(values), f"{prefix}_terminal": values[-1]})
        return summary
    elif protocol == "open_loop":
        identity = {"episode", "source_episode", "horizon", "sampled_rows"}
        tag = f"rollout_ic{init_context}"
        summary.update(n_episodes=len(records), per_episode=records)
        summary.update({f"{tag}_init_ctx": init_context, f"{tag}_num_hist": num_hist, f"{tag}_use_teacher_force": False})
    else:
        raise ValueError(protocol)
    summary.update({key: mean(row[key] for row in records) for key in records[0] if key not in identity})
    return summary


def write_results(output, protocol, summary):
    """Exclusive writes protect existing results, including symlinked sources."""
    names = {"teacher_forced": "0_result", "fixed_horizon": "0_result_fixed_horizon_rollout", "open_loop": "0_result_rollout"}
    stem = Path(output) / names[protocol]
    with stem.with_suffix(".json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    with stem.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows((key, value) for key, value in summary.items() if isinstance(value, (int, float)))


@torch.no_grad()
def export_nonplanning(model, manifest, episode_ids, normalization, output, metadata, lpips, segmentation,
                       *, batch_size=8, max_batches=None, num_visualizations=1, adaptive_history=False, physics=None,
                       windows_per_episode=2, seed=42, init_context=3, physics_by_protocol=None, contexts_by_protocol=None):
    """Run the bound three protocols; max_batches bounds inspection per protocol.

    Callers bind checkpoint/configuration/targets in metadata and must use a new
    logs directory. The model and metric networks must already be in eval mode.
    """
    from evaluation.evaluation_viz import save_sample_grid
    output = Path(output).resolve()
    output.relative_to((ROOT / "logs").resolve())
    output.mkdir(parents=True, exist_ok=False)
    validation = manifest.split_episode_ids()["validation"]
    local_ids = {global_id: local_id for local_id, global_id in enumerate(validation)}
    summaries = {}
    for protocol in ("teacher_forced", "fixed_horizon", "open_loop"):
        conditions = physics_by_protocol[protocol] if physics_by_protocol is not None else physics
        records, slices, episodes = [], 0, 0
        # One generator retains the fixed-window RNG across episode boundaries.
        samples = iter(evaluation_samples(manifest, episode_ids, protocol, num_hist=model.num_hist,
                                           adaptive_history=adaptive_history,windows_per_episode=windows_per_episode,
                                           seed=seed,init_context=init_context))
        pending = next(samples, None)
        while pending is not None and (max_batches is None or len(records) < max_batches):
            episode_id = pending["episode_id"]
            batch = [pending]
            pending = next(samples, None)
            if protocol == "teacher_forced":
                while (pending is not None and pending["episode_id"] == episode_id and len(batch) < batch_size
                       and pending["baseline_start"] > batch[-1]["baseline_start"]):
                    batch.append(pending)
                    pending = next(samples, None)
            data = [{**read_evaluation_sample(manifest, sample, 224, normalization),
                     "raw_masks": read_evaluation_masks(manifest, sample)} for sample in batch]
            if conditions is not None:
                from datasets.privileged_physics import physics_observation
                data = [{**row,'observation':physics_observation(row['observation'],conditions[row['episode_id']])}
                        for row in data]
            if contexts_by_protocol is not None:
                contexts = contexts_by_protocol[protocol]
                data = [{**row,'observation':{**row['observation'],**contexts[row['episode_id']]}} for row in data]
            values, images = score_batch(model, data, lpips, segmentation)
            row = dict(episode=local_ids[episode_id], source_episode=local_ids[episode_id])
            if protocol == "teacher_forced":
                count = len(batch[0]["observation_frames"])
                row.update(batch=len(records), variable_prefix=batch[0]["variable_prefix"], input_frames=count, prediction_frames=count - 1)
                row.update(values)
            elif protocol == "fixed_horizon":
                row.update(start=int(batch[0]["observation_frames"][0]), per_step=values)
            else:
                row.update(horizon=len(batch[0]["observation_frames"]) - 1, sampled_rows=batch[0]["observation_frames"].tolist())
                row.update(values)
            if protocol == "teacher_forced" and batch[0]["baseline_start"] == 0:
                episodes += 1
            records.append(row)
            slices += len(batch)
            limit = num_visualizations[protocol] if isinstance(num_visualizations, dict) else num_visualizations
            if len(records) <= limit:
                save_sample_grid(output / f"{protocol}_{len(records) - 1}.png", **images)
        provenance = dict(metadata, inspection=max_batches is not None, max_batches=max_batches)
        if protocol == "teacher_forced":
            provenance.update(n_val_episodes=episodes, n_val_slices=slices,
                              n_variable_prefix_slices=sum(row["variable_prefix"] for row in records), eval_mask_zarr_path=None)
        summary = summarize_results(protocol, records, provenance,batch_size=batch_size,num_hist=model.num_hist,
                                    windows_per_episode=windows_per_episode,seed=seed,init_context=init_context)
        write_results(output, protocol, summary)
        summaries[protocol] = summary
    return summaries
