"""Evaluation-only shoulder/wrist sweeps using the ordinary Reacher simulator."""
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from env.collect.reacher import PARAM_NAMES, collect_table


def reference_statistics(reference_collection):
    path = Path(reference_collection).resolve()
    manifest_path = path.with_suffix('.manifest.json')
    source = manifest_path.read_bytes()
    manifest = json.loads(source)
    root = zarr.open_group(str(path), mode='r')
    attrs = manifest['attrs']
    if (attrs.get('schema_version') != 'reacher_wthe.v4'
            or attrs.get('param_names') != list(PARAM_NAMES) or attrs != dict(root.attrs)):
        raise ValueError('Reference must be a matching Reacher v4 collection and manifest')
    physics = np.asarray([[row[name] for name in PARAM_NAMES]
                          for row in manifest['physics_params_by_dynamics']], dtype=np.float64)
    dynamics, split = root['meta/episode_dynamics_id'][:], root['meta/split'][:]
    if (dynamics.shape != split.shape or dynamics.ndim != 1 or not len(dynamics)
            or not np.issubdtype(dynamics.dtype, np.integer)
            or np.any(dynamics < 0) or np.any(dynamics >= len(physics))):
        raise ValueError('Invalid reference episode dynamics')
    training = np.unique(dynamics[split == 0])
    mean = np.asarray(attrs['physics_param_mean'], dtype=np.float32)
    std = np.asarray(attrs['physics_param_std'], dtype=np.float32)
    if (not len(training) or physics.ndim != 2 or physics.shape[1] != 6
            or mean.shape != (6,) or std.shape != (6,)
            or not all(np.isfinite(a).all() for a in (physics, mean, std))
            or np.any(std < 0) or np.any(physics[:, [0, 1, 3, 4]] <= 0)
            or np.any(physics[:, [2, 5]] < 0)):
        raise ValueError('Invalid reference physics or training statistics')
    std = np.where(std < 1e-8, 1, std).astype(np.float32)
    return np.median(physics[training], axis=0), mean, std, {
        'collection': str(path), 'manifest_sha256': hashlib.sha256(source).hexdigest()}


def sample_table(reference_collection, parameter, bins, values_per_bin, episodes_per_dynamics, seed):
    if parameter not in ('joint1_damping', 'joint2_damping'):
        raise ValueError('parameter must be joint1_damping or joint2_damping')
    for name, value in [('values_per_bin', values_per_bin),
                        ('episodes_per_dynamics', episodes_per_dynamics), ('seed', seed)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} must be an integer with a valid positive count or nonnegative seed')
    bounds = np.asarray(bins, dtype=np.float32)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds < 0)
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(bounds[1:, 0] < bounds[:-1, 1])):
        raise ValueError('bins must be ordered, nonoverlapping finite [lower, upper) intervals')
    median, mean, std, reference = reference_statistics(reference_collection)
    column = PARAM_NAMES.index(parameter)
    rows, bin_ids, episode_seeds, identities = [], [], [], []
    for bin_id, (low, high) in enumerate(bins):
        # Preserve the legacy Reacher group index in the seed hierarchy.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, bin_id, 0]))
        values = rng.uniform(low, high, values_per_bin).astype(np.float32)
        values = np.minimum(values, np.nextafter(np.float32(high), np.float32(low)))
        for index, value in enumerate(values):
            dynamics = len(rows)
            row = median.copy()
            row[column] = value
            rows.append(row)
            bin_ids.append(bin_id)
            seeds = []
            for episode in range(episodes_per_dynamics):
                pair = np.random.SeedSequence([seed, 1, bin_id, index, episode, 1]).generate_state(2).tolist()
                seeds.append(pair)
                identities.append(dict(dynamics_id=dynamics, episode_index_in_dynamics=episode,
                                       episode_seed=pair[0], policy_seed=pair[1], split=1))
            episode_seeds.append(seeds)
    raw = np.asarray(rows, dtype=np.float32)
    table = dict(params=[dict(zip(PARAM_NAMES, map(float, row))) for row in raw],
                 raw=raw, mean=mean, std=std, normed=((raw-mean)/std).astype(np.float32))
    sampling = dict(master_seed=seed, group='reacher', parameter=parameter,
                    bins=bins, bin_ids=bin_ids, fixed={name: float(median[i])
                    for i, name in enumerate(PARAM_NAMES) if name != parameter},
                    physics=raw.tolist(), episode_seeds=episode_seeds,
                    episodes_per_dynamics=episodes_per_dynamics, values_per_bin=values_per_bin,
                    sampling='independent uniform per bin', split='validation',
                    normalization_mean=mean.tolist(), normalization_std=std.tolist(), reference=reference)
    return table, identities, sampling


def collect(output, reference_collection, parameter, bins, values_per_bin=15,
            episodes_per_dynamics=5, seed=20260924, length=100, image_size=224):
    for name, value in [('length', length), ('image_size', image_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    table, identities, sampling = sample_table(reference_collection, parameter, bins,
                                              values_per_bin, episodes_per_dynamics, seed)
    config = dict(reference_collection=str(reference_collection), parameter=parameter, bins=bins,
                  values_per_bin=values_per_bin, episodes_per_dynamics=episodes_per_dynamics,
                  seed=seed, length=length, image_size=image_size)
    return collect_table(output, table, identities, config, sampling=sampling)


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='reacher_extrapolation')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
