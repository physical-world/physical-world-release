"""Aligned-v2 BeerPong: fixed H5 starts from three observations at stride 15."""
import torch
from datasets.beerpong import beerpong_conditions
from evaluation.common import evaluation_samples
from evaluation.segmentation import iou_counts, iou_from_counts
from evaluation.visual_evaluation import read_condition_sample, score_visual_batch, summarize_visual, export_visual


def beerpong_samples(collection, targets, protocol, num_hist=3):
    if protocol == 'teacher_forced':
        targets = [{'source_episode': i} for i in sorted({t['source_episode'] for t in targets})]
    for target in targets:
        for sample in evaluation_samples(collection, [target['source_episode']], protocol,
                                         frameskip=15, num_hist=num_hist, adaptive_history=num_hist > 3,
                                         fixed_horizon=7, windows_per_episode=None, fixed_start=0):
            if protocol == 'fixed_horizon':
                sample['initial_indices'] = torch.arange(3)
                sample['target_indices'] = torch.arange(3, 8)
                sample['target_frames'] = sample['observation_frames'][3:]
            yield {**sample, 'target': dict(target)}


def read_beerpong_sample(collection, sample, normalization, masks):
    return read_condition_sample(collection, sample, normalization, masks,
                                 beerpong_conditions(collection, [sample['episode_id']])[0])


def beerpong_iou_counts(predicted, target):
    counts = iou_counts(predicted, target, ((0, 'background'), (1, 'ball')))
    return {name: counts[name] for name in ('background', 'ball')}


def beerpong_iou_values(counts):
    scores = iou_from_counts(counts)
    return {**{f'segm_iou_{key}': value for key, value in scores.items()},
            'segm_miou': sum(scores.values()) / 2}


def score_beerpong_batch(model, samples, lpips, segmentation):
    return score_visual_batch(model, samples, lpips, segmentation, beerpong_iou_counts, beerpong_iou_values)


def summarize_beerpong(protocol, records, counts, metadata, num_hist):
    result = summarize_visual(protocol, records, counts, metadata, num_hist, beerpong_iou_values)
    if protocol == 'fixed_horizon':
        result.pop('raw_action_horizon')
        result.update(raw_timestep_horizon=75, initial_observation_frames=3)
    return result


def export_beerpong(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                    *, batch_size=8, max_batches=None, num_visualizations=1, physics=None, physics_by_protocol=None, contexts_by_protocol=None):
    return export_visual(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                         sample_indices=beerpong_samples, read_sample=read_beerpong_sample,
                         score_batch=score_beerpong_batch, summarize=summarize_beerpong,
                         batch_size=batch_size, max_batches=max_batches, num_visualizations=num_visualizations, physics=physics,
                         physics_by_protocol=physics_by_protocol,contexts_by_protocol=contexts_by_protocol)
