"""Shared image metrics with explicit environment labels and optional Privileged Physics inputs."""

import torch
from runtime.checkpoints import ROOT
from datasets.common import preprocess_rgb

@torch.no_grad()
def score_visual_batch(model, samples, lpips, segmentation, count_masks, score_counts):
    """Preserve visual-only whole-sequence image scores and pooled pixel counts."""
    from evaluation.image_metrics import image_metrics
    from evaluation.result_export import predict_batch, evaluation_images
    from evaluation.segmentation import segment_images
    data = predict_batch(model, samples)
    predicted, target = data['predicted'], data['target']
    masks = torch.stack([s['raw_masks'] for s in samples]).to(predicted.device)
    predicted_masks = torch.cat([segment_images(segmentation, predicted[:, i:i + 8])
                                 for i in range(0, predicted.shape[1], 8)], 1)
    protocol = samples[0]['protocol']
    if protocol == 'teacher_forced':
        values = {key: float(value) for key, value in data['losses'].items()}
        for name, images, truth in [('pred', predicted, target), ('reconstructed', data['reconstructed'], data['observation']['visual'])]:
            values.update({f'img_{key}_{name}': value for key, value in image_metrics(images, truth, lpips).items()})
        counts = count_masks(predicted_masks, masks)
        values.update(score_counts(counts))
    else:
        error = (data['latent']['visual'] - data['target_latent']['visual']).square()
        if protocol == 'fixed_horizon':
            values, counts = [], []
            for step in range(predicted.shape[1]):
                count = count_masks(predicted_masks[:, step], masks[:, step])
                row = {'z_visual_err': float(error[:, step].mean())}
                row.update({f'img_{key}': value for key, value in image_metrics(predicted[:, step], target[:, step], lpips).items()})
                row.update(score_counts(count))
                values.append(row)
                counts.append(count)
        else:
            values = dict(z_visual_err_pred=float(error.mean()), z_visual_err_last=float(error[:, -1].mean()))
            values.update({f'img_{key}_pred': value for key, value in image_metrics(predicted, target, lpips).items()})
            counts = count_masks(predicted_masks, masks)
            values.update({f'{key}_pred': value for key, value in score_counts(counts).items()})
    return values, counts, {**evaluation_images(data, samples), 'masks': masks.cpu()}



def pool_counts(counts):
    return {label: {key: sum(row[label][key] for row in counts) for key in counts[0][label]}
            for label in counts[0]}


def summarize_visual(protocol, records, counts, metadata, num_hist, score_counts):
    """Weight teacher batches by samples and sum IoU counts before division."""
    from statistics import mean
    result = dict(metadata)
    if protocol == 'teacher_forced':
        weight = sum(row['batch_size'] for row in records)
        result.update(n_val_slices=weight, n_batches=len(records))
        result.update({key: sum(row['values'][key] * row['batch_size'] for row in records) / weight
                       for key in records[0]['values'] if not key.startswith('segm_')})
        result.update(score_counts(pool_counts(counts)))
    elif protocol == 'fixed_horizon':
        result.update(horizon_wm_tokens=5, raw_action_horizon=25, per_target=records)
        steps = []
        for step in range(5):
            values = {key: mean(row['per_step'][step][key] for row in records)
                      for key in records[0]['per_step'][step] if not key.startswith('segm_')}
            values.update(score_counts(pool_counts([row[step] for row in counts])))
            steps.append(values)
        for key in steps[0]:
            values = [row[key] for row in steps]
            prefix = f'fixed_h5_{key}'
            result.update({f'{prefix}_step{i}': value for i, value in enumerate(values, 1)})
            result.update({f'{prefix}_mean': mean(values), f'{prefix}_terminal': values[-1]})
    else:
        result.update(rollout_history_size=None, training_num_hist=num_hist, rollout_init_ctx=3, per_target=records)
        result.update({f'rollout_ic3_{key}': mean(row[key] for row in records)
                       for key in records[0] if key.startswith(('img_', 'z_visual_'))})
        result.update({f'rollout_ic3_{key}_pred': value for key, value in
                       score_counts(pool_counts(counts)).items()})
    return result


@torch.no_grad()
def export_visual(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                   *, sample_indices, read_sample, score_batch, summarize, batch_size=8, max_batches=None, num_visualizations=1, physics=None, physics_by_protocol=None, contexts_by_protocol=None):
    """Execute the three bound protocols into a fresh logs directory."""
    import json
    from pathlib import Path
    from evaluation.result_export import write_results
    from evaluation.evaluation_viz import save_sample_grid
    output = Path(output).resolve()
    output.relative_to((ROOT / 'logs').resolve())
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for protocol in ('teacher_forced', 'fixed_horizon', 'open_loop'):
        conditions = physics_by_protocol[protocol] if physics_by_protocol is not None else physics
        records, counts = [], []
        samples = iter(sample_indices(collection, targets, protocol, model.num_hist))
        pending = next(samples, None)
        while pending is not None and (max_batches is None or len(records) < max_batches):
            batch = [pending]
            pending = next(samples, None)
            if protocol == 'teacher_forced':
                while pending is not None and pending['episode_id'] == batch[0]['episode_id'] and len(batch) < batch_size:
                    batch.append(pending)
                    pending = next(samples, None)
            data = [read_sample(collection, sample, normalization, masks) for sample in batch]
            if conditions is not None:
                from datasets.privileged_physics import physics_observation
                data = [{**row,'observation':physics_observation(row['observation'],conditions[row['episode_id']])}
                        for row in data]
            if contexts_by_protocol is not None:
                contexts = contexts_by_protocol[protocol]
                data = [{**row,'observation':{**row['observation'],**contexts[row['episode_id']]}} for row in data]
            values, count, images = score_batch(model, data, lpips, segmentation)
            if protocol == 'teacher_forced':
                row = dict(batch_size=len(batch), values=values)
            elif protocol == 'fixed_horizon':
                row = dict(batch[0]['target'], rows=batch[0]['observation_frames'].tolist(), per_step=values)
            else:
                row = dict(batch[0]['target'], sampled_rows=batch[0]['observation_frames'].tolist(), **values)
            records.append(row)
            counts.append(count)
            if len(records) <= num_visualizations:
                save_sample_grid(output / f'{protocol}_{len(records)-1}.png', **images)
        provenance = dict(metadata, inspection=metadata.get("inspection", False) or max_batches is not None, max_batches=max_batches)
        summary = summarize(protocol, records, counts, provenance, model.num_hist)
        write_results(output, protocol, summary)
        with (output / f'{protocol}_reductions.json').open('x') as stream:
            json.dump(dict(records=records, counts=counts), stream, indent=2)
        summaries[protocol] = summary
    return summaries


def read_condition_sample(collection, sample, normalization, masks, condition):
    """Keep the episode condition across every real/padded token."""
    _, _, begin, end = collection.episode_info(sample['episode_id'])
    frames = sample['observation_frames']
    indices = (frames + begin).numpy()
    data = collection.root['data']
    observation = {'visual': preprocess_rgb(data['img'].oindex[indices], 224)}
    if sample['adaptive_history']:
        observation['valid_frames'] = torch.arange(len(frames)) <= len(sample['initial_indices'])
    condition = (condition - normalization['action_mean']) / normalization['action_std']
    return {**sample, 'observation': observation,
            'action': condition.expand(len(sample['action_frames']), -1),
            'state': torch.from_numpy(data['state'].oindex[indices]).float(),
            'raw_masks': torch.from_numpy(masks['data/img_mask'].oindex[(sample['target_frames'] + begin).numpy()])}

