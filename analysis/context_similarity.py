"""Extract the exact paper clips with the canonical Context encoder."""
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from datasets.common import _episode_bounds, _read_window
from datasets.pusht import pusht_observation
from models.physics.estimator import load_estimator


def read_context(config, sample):
    collection = Path(config['collection'])
    manifest = json.loads((collection/'manifest.json').read_text())
    slot = sample['slot']
    for field, key in [('reset_seeds', 'reset_seed'), ('noise_seeds', 'noise_seed'), ('sigmas_px', 'sigma_px')]:
        if not 0 <= slot < len(manifest[field]) or manifest[field][slot] != sample[key]:
            raise ValueError(f'Collected episode {slot} differs from the configured {key}')
    cog = manifest['cog_seeds'].index(sample['cog_seed']) + 1
    group = zarr.open_group(str(collection/'rollouts'/f'shard_{(cog-1)//10:02d}.zarr'/f'cog_{cog:03d}'), mode='r')
    begin, end = _episode_bounds(group['meta/episode_ends'], sample['slot'])
    stats = {k: torch.tensor(v) for k, v in config['normalization'].items()}
    window = _read_window(group['data'], begin+sample['start'], config['frames'], config['frameskip'],
                          stats['action_mean'], stats['action_std'], episode_stop=end)
    observation, _ = pusht_observation(window['rgb'], window['state'], config['image_size'], stats)
    context = {'ctx_visual': observation['visual'], 'ctx_proprio': observation['proprio'],
               'ctx_act': window['action']}
    return context, window


@torch.inference_mode()
def extract_tokens(config, samples=None, device='cuda'):
    encoder = load_estimator(config['checkpoint'])['ctx_encoder']
    while hasattr(encoder, 'module'):
        encoder = encoder.module
    encoder = encoder.to(device).eval()
    tokens = []
    for sample in config['samples'] if samples is None else samples:
        context, _ = read_context(config, sample)
        batch = {k: v.unsqueeze(0).to(device) for k, v in context.items()}
        tokens.append(encoder(batch).cpu().numpy()[0])
    return np.stack(tokens)


def similarity_matrix(tokens, groups=8, clips=10):
    unit = tokens / np.linalg.norm(tokens, axis=-1, keepdims=True)
    pairs = (unit @ unit.T).reshape(groups, clips, groups, clips)
    matrix = np.empty((groups, groups))
    for i in range(groups):
        for j in range(groups):
            block = pairs[i, :, j, :]
            matrix[i, j] = block[~np.eye(clips, dtype=bool)].mean() if i == j else block.mean()
    return matrix


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('conf/analysis/context_similarity.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, help='Aggregate existing tokens without GPU extraction')
    parser.add_argument('--collection', type=Path, help='Use a newly collected PushT Context directory')
    args = parser.parse_args()
    if not args.tokens and not torch.cuda.is_available():
        raise RuntimeError('Context token extraction requires a CUDA GPU')
    config = json.loads(args.config.read_text())
    if args.collection is not None:
        config['collection'] = str(args.collection.resolve())
    output = args.output.resolve()
    if not output.is_relative_to(Path('logs').resolve()):
        raise ValueError('Analysis output must be under logs/')
    output.mkdir(parents=True, exist_ok=False)
    if args.tokens:
        with np.load(args.tokens) as archive:
            tokens = archive['embeddings']
    else:
        tokens = extract_tokens(config)
    matrix = similarity_matrix(tokens)
    np.savez_compressed(output/'tokens.npz', embeddings=tokens)
    np.savetxt(output/'similarity.csv', matrix, delimiter=',')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'result.json').write_text(json.dumps({'matrix':matrix.tolist(), 'self_pairs_excluded':True,
        'cross_pairs':100, 'within_pairs':90, 'input_tokens':str(args.tokens) if args.tokens else None}, indent=2)+'\n')


if __name__ == '__main__':
    main()
