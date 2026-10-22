"""PushT Context diagnostics with matched reset and noise seeds across CoGs."""
import json
from pathlib import Path

import numpy as np
import zarr

from env.collect.common import append_arrays, prepare_output, save_run, CollectionLog
from env.collect.pusht import collect_episode, episode_noise_sigmas, normalized_cog
from env.collect.pusht_policy import load_policy


def collect(output, checkpoint, cog_seeds, clean_episodes_per_dynamics=5,
            noise_sigmas_px=(.1, .3, .5, 1., 1.5), noise_repeats=1,
            reset_seed_base=903000, noise_seed_base=904000, max_steps=300,
            stop_on_success=False, image_size=224, device='cpu'):
    sigmas = episode_noise_sigmas(clean_episodes_per_dynamics, noise_sigmas_px, noise_repeats)
    if not cog_seeds or len(set(cog_seeds)) != len(cog_seeds) or any(
            type(seed) is not int or not 0 <= seed <= np.iinfo(np.int32).max for seed in cog_seeds):
        raise ValueError('cog_seeds must be distinct nonnegative int32 integers')
    for name, base in [('reset_seed_base', reset_seed_base), ('noise_seed_base', noise_seed_base)]:
        if type(base) is not int or not 0 <= base <= np.iinfo(np.int32).max - len(sigmas) + 1:
            raise ValueError(f'{name} and episode seeds must fit nonnegative int32 integers')
    if any(type(value) is not int or value <= 0 for value in (max_steps, image_size)):
        raise ValueError('max_steps and image_size must be positive integers')
    if type(stop_on_success) is not bool:
        raise ValueError('stop_on_success must be a boolean')
    output = prepare_output(output)
    with CollectionLog(output, 'pusht_context', len(cog_seeds),
                       len(cog_seeds) * len(sigmas), 'cog_seed', directory=True) as progress:
        progress.logger.info('loading policy checkpoint=%s device=%s', checkpoint, device)
        policy = load_policy(checkpoint, device)
        config = dict(checkpoint=str(Path(checkpoint).resolve()), cog_seeds=list(cog_seeds),
                      clean_episodes_per_dynamics=clean_episodes_per_dynamics,
                      noise_sigmas_px=list(noise_sigmas_px), noise_repeats=noise_repeats,
                      reset_seed_base=reset_seed_base, noise_seed_base=noise_seed_base,
                      max_steps=max_steps, stop_on_success=stop_on_success, image_size=image_size, device=device)
        manifest = dict(schema='pusht.context_cog_seed.v1', cog_seeds=list(cog_seeds),
                        coordinates=[normalized_cog(seed).tolist() for seed in cog_seeds],
                        steps=max_steps, reset_seeds=list(range(reset_seed_base, reset_seed_base + len(sigmas))),
                        noise_seeds=list(range(noise_seed_base, noise_seed_base + len(sigmas))),
                        sigmas_px=sigmas, episodes_per_dynamics=len(sigmas),
                        episodes=len(cog_seeds) * len(sigmas), collection_config=config)
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        save_run(output/'manifest.json', config)
        for position, seed in enumerate(cog_seeds):
            progress.start_group(seed)
            shard = zarr.open_group(str(output/'rollouts'/f'shard_{position//10:02d}.zarr'), mode='a')
            group = shard.create_group(f'cog_{position+1:03d}')
            group.attrs.update(cog_seed=seed, cog_local_normed=manifest['coordinates'][position],
                               schema_version=manifest['schema'])
            data, meta = group.create_group('data'), group.create_group('meta')
            ends, records = [], []
            for slot, sigma in enumerate(sigmas):
                progress.episode = slot
                identity = dict(cog_seed=seed, reset_seed=reset_seed_base+slot,
                                noise_seed=noise_seed_base+slot, noise_sigma_px=sigma)
                episode, record = collect_episode(policy, identity, max_steps, image_size,
                                                  stop_on_success=stop_on_success)
                append_arrays(data, episode)
                ends.append((ends[-1] if ends else 0) + len(episode['state']))
                records.append(dict(record, slot=slot, sigma_px=sigma))
            meta.create_dataset('episode_ends', data=np.asarray(ends, dtype=np.int64))
            (Path(group.store.path)/group.path/'episodes.json').write_text(json.dumps(records, indent=2)+'\n')
            progress.complete_group(len(sigmas), ends[-1])
        return manifest


def main():
    import hydra
    from omegaconf import OmegaConf
    @hydra.main(version_base=None, config_path='../../conf/collection', config_name='pusht_context')
    def run(cfg):
        collect(**OmegaConf.to_container(cfg, resolve=True))
    run()


if __name__ == '__main__':
    main()
