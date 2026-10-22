"""Reacher IC3 extrapolation with bound targets and training-owned statistics."""
import json
from pathlib import Path

import torch
import zarr

from datasets.reacher import ReacherEpisodes, reacher_normalization
from datasets.physics import reacher_physics
from datasets.context import read_reacher_context
from datasets.privileged_physics import physics_observation
from evaluation.reacher_evaluation import read_reacher_sample


def training_normalization(config):
    training = ReacherEpisodes(config['training_collection'])
    stats = reacher_normalization(training)
    _, mean, std = reacher_physics(training)
    return {**stats, 'physics_mean':mean, 'physics_std':std}


def bound_sample(target, init_context=3):
    frames = torch.tensor(target['sampled_rows'], dtype=torch.long)
    stride = int(frames[1]-frames[0])
    actions = torch.arange((len(frames)-1)*stride).reshape(-1, stride)+int(frames[0])
    indices = torch.arange(init_context, len(frames))
    return {'protocol':'open_loop', 'episode_id':target['source_episode'],
            'observation_frames':frames, 'action_frames':actions, 'action_valid':torch.ones_like(actions,dtype=torch.bool),
            'initial_indices':torch.arange(init_context), 'target_indices':indices, 'target_frames':frames[indices],
            'adaptive_history':False, 'target':target}


def read_input(config, target, normalization, method, *, collection=None, masks=None):
    collection = ReacherEpisodes(config['collection']) if collection is None else collection
    masks = zarr.open_group(config['masks'], mode='r') if masks is None else masks
    sample = bound_sample(target, config['init_context'])
    data = read_reacher_sample(collection, sample, normalization, masks)
    episode = target['source_episode']
    if method == 'privileged_physics':
        physical = collection.episode_physics(episode)
        raw = torch.tensor([physical[k] for k in ('joint1_damping','joint2_damping')])
        physics = (raw-normalization['physics_mean'])/normalization['physics_std']
        data['observation'] = physics_observation(data['observation'], physics)
    elif method == 'context':
        assignments = json.loads(Path(config['context_reference']).read_text())['assignments']
        row = assignments[str(episode)]
        donor = {'donor':row['source_episode'], 'start':row['start']}
        context = read_reacher_context(collection, donor, 60, 5, 224, normalization)
        data['observation'].update(context)
        data['condition'] = donor
    return data


@torch.inference_mode()
def evaluate_bin(config, method, bin_id, output, *, limit=None, normalization=None, target_indices=None):
    from analysis.artifacts import create_output
    from analysis.inference_efficiency import load_method
    from evaluation.image_metrics import LPIPS
    from evaluation.segmentation import load_segmentation_model
    from evaluation.reacher_evaluation import score_reacher_batch, summarize_reacher
    from evaluation.evaluation_viz import save_sample_grid
    output = create_output(output, config)
    collection = ReacherEpisodes(config['collection'])
    masks = zarr.open_group(config['masks'], mode='r')
    stats = training_normalization(config) if normalization is None else normalization
    model = load_method(config, method, load_decoder=True)
    segmenter, _ = load_segmentation_model(config['methods'][method]['segmentation'], 'cuda')
    lpips = LPIPS(net_type='vgg', version='0.1').cuda().eval().requires_grad_(False)
    targets = config['targets'][str(bin_id)]
    indices = list(range(len(targets))) if target_indices is None else target_indices
    records, counts = [], []
    for index in indices[:limit]:
        target = targets[index]
        data = read_input(config, target, stats, method, collection=collection, masks=masks)
        values, count, images = score_reacher_batch(model, [data], lpips, segmenter)
        records.append({**target, **values})
        counts.append(count)
        if len(records) <= 4:
            save_sample_grid(output/f'target_{index:03d}.png', **images)
            torch.save(images, output/f'target_{index:03d}_images.pt')
        print(method, 'bin', bin_id, 'target', index, 'PSNR', values['img_psnr_pred'], 'IoU', values['segm_iou_overall_pred'], flush=True)
    metadata = {'method':method, 'bin':bin_id, 'epoch':20, 'inspection':limit is not None or target_indices is not None,
                'asset_binding':config['methods'][method], 'seed':config['seed'],
                'normalization':{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in stats.items()}}
    summary = summarize_reacher('open_loop', records, counts, metadata, model.num_hist)
    (output/'0_result_rollout.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary



def summarize_bins(config, results):
    import numpy as np
    rows = []
    for method, asset in config['methods'].items():
        for bin_id, bounds in enumerate(config['bins']):
            path = Path(results)/method/f'bin_{bin_id:02d}'/'0_result_rollout.json'
            result = json.loads(path.read_text())
            records = result['per_target']
            row = {'method':method, 'bin':bin_id, 'lower':bounds[0], 'upper':bounds[1],
                   'center':sum(bounds)/2, 'count':len(records), 'in_domain':bin_id in config['id_bins'],
                   'source':str(path), 'inspection':result.get('inspection', False)}
            # IoU is the arm foreground IoU, matching the paper's Reacher tables.
            for name,key,scale in [('psnr','img_psnr_pred',1), ('iou_percent','segm_iou_overall_pred',100)]:
                values = np.asarray([r[key] for r in records])*scale
                row[name] = float(values.mean())
                row[name+'_sem'] = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
            rows.append(row)
    return rows


def plot_qualitative(privileged_physics, context):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    sequences = [('DINO-WM + Privileged Physics', torch.cat((privileged_physics['initial'],privileged_physics['predicted']),1)[0]),
                 ('DINO-WM + PhysiCal (Ours)', torch.cat((context['initial'],context['predicted']),1)[0]),
                 ('GT', torch.cat((context['initial'],context['target']),1)[0])]
    fig, axes = plt.subplots(9, 7, figsize=(12, 16), gridspec_kw={'wspace':.02,'hspace':.22})
    for method,(label,images) in enumerate(sequences):
        for time in range(21):
            ax = axes[method*3+time//7,time%7]
            ax.axis('off')
            if time < len(images):
                artist = ax.imshow(((images[time]+1)/2).clamp(0,1).permute(1,2,0), interpolation='none')
                border = FancyBboxPatch((-.5,-.5),224,224,boxstyle='round,pad=0,rounding_size=4',
                                        facecolor='none',edgecolor='#D2D2D2',linewidth=.8,clip_on=False)
                ax.add_patch(border)
                artist.set_clip_path(border)
                ax.set_title(f't = {time}', fontsize=10, fontfamily='serif')
        axes[method*3+1,0].text(-.15,.5,label.replace(' +','\n+'),rotation=90,ha='right',va='center',
                                    transform=axes[method*3+1,0].transAxes,fontsize=13,fontfamily='serif')
    fig.subplots_adjust(left=.12,right=.99,top=.98,bottom=.02)
    return fig

def main():
    import argparse
    from analysis.artifacts import create_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('conf/analysis/reacher_extrapolation.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=['vanilla','privileged_physics','context'])
    parser.add_argument('--bins', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--limit', type=int, help='Mark a bounded inspection; omit for the complete bound targets')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--results', type=Path, help='Evaluation directory to summarize (required with --summarize)')
    parser.add_argument('--qualitative', nargs=2, type=Path, metavar=('PRIVILEGED_PHYSICS_IMAGES','CONTEXT_IMAGES'))
    args = parser.parse_args()
    if args.summarize and args.results is None:
        parser.error('--summarize requires --results')
    if not args.summarize and not torch.cuda.is_available():
        raise RuntimeError('Extrapolation evaluation requires a CUDA GPU')
    config = json.loads(args.config.read_text())
    output = create_output(args.output, config)
    if args.summarize:
        rows = summarize_bins(config, args.results)
        (output/'bin_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
        if args.qualitative:
            images = [torch.load(p,map_location='cpu',weights_only=True) for p in args.qualitative]
            plot_qualitative(*images).savefig(output/'qualitative.pdf')
        return
    torch.manual_seed(config['seed'])
    stats = training_normalization(config)
    for method in args.methods or config['methods']:
        for bin_id in args.bins:
            evaluate_bin(config, method, bin_id, output/method/f'bin_{bin_id:02d}', limit=args.limit, normalization=stats)


if __name__ == '__main__':
    main()
