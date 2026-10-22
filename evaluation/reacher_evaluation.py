"""Bound Reacher target timelines and segmentation semantics."""

import torch

from datasets.reacher import reacher_observation
from evaluation.common import evaluation_samples, read_evaluation_sample
from evaluation.segmentation import iou_counts, iou_from_counts


def reacher_samples(collection, targets, protocol, num_hist=3):
    """Teacher forcing deduplicates episodes; rollouts retain target order/offsets."""
    validation = collection.split_episode_ids()['validation']
    if protocol == 'teacher_forced':
        selected = sorted({target['validation_episode'] for target in targets})
        for local in selected:
            for sample in evaluation_samples(collection, [validation[local]], protocol, num_hist=num_hist,
                                             adaptive_history=num_hist > 3):
                yield {**sample, 'target': {'validation_episode': local, 'source_episode': validation[local]}}
    else:
        for target in targets:
            samples = evaluation_samples(collection, [validation[target['validation_episode']]], protocol,
                                         num_hist=num_hist, windows_per_episode=None,
                                         fixed_start=target['offset'], init_context=3)
            for sample in samples:
                yield {**sample, 'target': dict(target)}


def read_reacher_sample(collection, sample, normalization, masks):
    """Read bounded RGB/actions and directly aligned 0/1/2 segmentation labels."""
    result = read_evaluation_sample(collection, sample, 224, normalization, reacher_observation, hold_action=False)
    _, _, begin, _ = collection.episode_info(sample['episode_id'])
    indices = (sample['target_frames'] + begin).numpy()
    result['raw_masks'] = torch.from_numpy(masks['data/img_mask'].oindex[indices])
    return result


def reacher_iou_counts(predicted, target):
    return iou_counts(predicted, target, ((0, 'background'), (1, 'link1'), (2, 'link2')))


def reacher_iou_values(counts):
    scores = iou_from_counts(counts)
    result = {f'segm_iou_{key}': value for key, value in scores.items()}
    result['segm_miou'] = sum(scores[name] for name in ('background', 'link1', 'link2')) / 3
    return result


def score_reacher_batch(model, samples, lpips, segmentation):
    from evaluation.visual_evaluation import score_visual_batch
    return score_visual_batch(model, samples, lpips, segmentation, reacher_iou_counts, reacher_iou_values)


def summarize_reacher(protocol, records, counts, metadata, num_hist):
    from evaluation.visual_evaluation import summarize_visual
    return summarize_visual(protocol, records, counts, metadata, num_hist, reacher_iou_values)


def export_reacher(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                   *, batch_size=8, max_batches=None, num_visualizations=1, physics=None, physics_by_protocol=None, contexts_by_protocol=None):
    from evaluation.visual_evaluation import export_visual
    return export_visual(model, collection, targets, normalization, masks, output, metadata, lpips, segmentation,
                         sample_indices=reacher_samples, read_sample=read_reacher_sample,
                         score_batch=score_reacher_batch, summarize=summarize_reacher,
                         batch_size=batch_size, max_batches=max_batches, num_visualizations=num_visualizations,physics=physics,
                         physics_by_protocol=physics_by_protocol,contexts_by_protocol=contexts_by_protocol)
