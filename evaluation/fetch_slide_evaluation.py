"""Bound FetchSlide variable timelines and fixed episode conditions."""

import torch
from datasets.fetch_slide import fetch_conditions
from evaluation.common import evaluation_samples
from evaluation.segmentation import iou_counts, iou_from_counts


def fetch_samples(collection, targets, protocol, num_hist=3):
    """Global source IDs; unique teacher episodes, target-ordered rollouts."""
    if protocol == 'teacher_forced':
        targets = [{'source_episode': i} for i in sorted({t['source_episode'] for t in targets})]
    for target in targets:
        for sample in evaluation_samples(collection, [target['source_episode']], protocol,
                                         num_hist=num_hist, adaptive_history=num_hist > 3,
                                         windows_per_episode=None, fixed_start=target.get('offset', 0)):
            yield {**sample, 'target': dict(target)}


def read_fetch_sample(collection, sample, normalization, masks):
    from evaluation.visual_evaluation import read_condition_sample
    return read_condition_sample(collection, sample, normalization, masks,
                                 fetch_conditions(collection, [sample['episode_id']])[0])


def fetch_iou_counts(predicted, target):
    counts = iou_counts(predicted, target, ((0, 'background'), (1, 'puck')))
    return {name: counts[name] for name in ('background', 'puck')}


def fetch_iou_values(counts):
    scores = iou_from_counts(counts)
    return {**{f'segm_iou_{key}': value for key, value in scores.items()},
            'segm_miou': sum(scores.values()) / 2}


def score_fetch_batch(model, samples, lpips, segmentation):
    from evaluation.visual_evaluation import score_visual_batch
    return score_visual_batch(model, samples, lpips, segmentation, fetch_iou_counts, fetch_iou_values)


def summarize_fetch(protocol, records, counts, metadata, num_hist):
    from evaluation.visual_evaluation import summarize_visual
    result = summarize_visual(protocol, records, counts, metadata, num_hist, fetch_iou_values)
    if protocol == 'teacher_forced':
        result.update(n_terminal_padded_slices=0, n_terminal_padding_tokens=0)
    elif protocol == 'fixed_horizon':
        result['raw_timestep_horizon'] = result.pop('raw_action_horizon')
        result['initial_observation_frames'] = 1
    return result


def export_fetch(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                 *, batch_size=8, max_batches=None, num_visualizations=1, physics=None, physics_by_protocol=None, contexts_by_protocol=None):
    from evaluation.visual_evaluation import export_visual
    return export_visual(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                         sample_indices=fetch_samples, read_sample=read_fetch_sample,
                         score_batch=score_fetch_batch, summarize=summarize_fetch,
                         batch_size=batch_size, max_batches=max_batches, num_visualizations=num_visualizations, physics=physics,
                         physics_by_protocol=physics_by_protocol,contexts_by_protocol=contexts_by_protocol)
