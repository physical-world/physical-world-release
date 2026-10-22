"""Seeded or recorded evaluation donors shared by all conditioning methods."""
import json
from pathlib import Path
from omegaconf import OmegaConf

from datasets.context import ContextSampler, pusht_episode_values
from datasets.estimated_context import fixed_context_records, validate_context_records
from datasets.evaluation_targets import (collection_identity, local_rng, manifest_hash,
                                         manifest_text, target_episode_ids, with_hash)


def protocol_names(workflow):
    return ['planning'] if workflow == 'planning' else ['teacher_forced', 'fixed_horizon', 'open_loop']


def context_reader(environment):
    from datasets.context import (read_pusht_context, read_reacher_context,
                                  read_fetch_context, read_beerpong_context)
    return dict(pusht=read_pusht_context, reacher=read_reacher_context,
                fetch_slide=read_fetch_context, beerpong=read_beerpong_context)[environment]


def resolve_contexts(collection, targets, geometry, workflow, source=None, seed=42):
    environment = targets['environment']
    episodes = target_episode_ids(collection, targets)
    names = protocol_names(workflow)
    validation = sorted(collection.split_episode_ids()['validation'])
    metadata = dict(environment=environment, dataset=geometry)
    assignments = {}
    if OmegaConf.is_config(source):
        source = OmegaConf.to_container(source, resolve=True)
    if isinstance(source, str) and source.lstrip().startswith('{'):
        source = json.loads(source)
    if source is not None and not isinstance(source, dict):
        content = json.loads(Path(source).read_text())
        if 'protocols' in content:
            if content['data_identity'] != collection_identity(collection) or content['environment'] != environment:
                raise ValueError('Context manifest is bound to another collection')
            if content['split_identity'] != manifest_hash(validation):
                raise ValueError('Context manifest is bound to another split')
            if content['geometry'] != geometry:
                raise ValueError('Context geometry differs from the sampling protocol')
            for name in names:
                available = {row['target']: row for row in content['protocols'][name]}
                if len(available) != len(content['protocols'][name]):
                    raise ValueError('Context manifest requires unique targets')
                assignments[name] = [available[episode] for episode in episodes]
                validate_context_records(collection, metadata, {episode: available[episode] for episode in episodes})
            source_seed = content.get('ctx_seed')
        else:
            source = dict.fromkeys(names, source)
    if isinstance(source, dict):
        for name in names:
            records = fixed_context_records(source[name], collection, metadata, target_ids=episodes)
            assignments[name] = [records[episode] for episode in episodes]
    elif source is None:
        conditions = (pusht_episode_values(collection, validation, 'cog_seed') if environment == 'pusht'
                      else collection.dynamics[validation])
        sampler = ContextSampler(collection, validation, conditions, geometry['ctx_length'],
                                 geometry['ctx_frameskip'], geometry['fixed_start'], geometry['min_episode_length'])
        rows = []
        for episode in episodes:
            record = sampler.sample(episode, local_rng(seed, episode, 'context'))
            rows.append({key: int(record[key]) for key in ('target', 'donor', 'start')})
        assignments = {name: rows for name in names}
    result = dict(environment=environment, data_identity=collection_identity(collection),
                  split='validation', split_identity=manifest_hash(validation), geometry=geometry, protocols=assignments)
    if source is None:
        result['ctx_seed'] = int(seed)
    elif not isinstance(source, dict) and source_seed is not None:
        result['ctx_seed'] = source_seed
    return with_hash(result, 'ctx_set_hash')


def prepare_contexts(options, collection, targets, sampling, workflow, run):
    source = getattr(options, 'ctx_manifest', None)
    seed = options.eval_seed if options.ctx_seed is None else options.ctx_seed
    manifest = resolve_contexts(collection, targets, sampling['context'], workflow, source, seed)
    path = Path(run) / 'ctx_manifest.json'
    with path.open('x') as stream:
        stream.write(manifest_text(manifest))
    binding = Path(run) / 'sampling_sources.json'
    provenance = json.loads(binding.read_text())
    provenance['ctx_manifest'] = OmegaConf.to_container(source, resolve=True) if OmegaConf.is_config(source) else str(source) if isinstance(source, Path) else source
    references = provenance['ctx_manifest']
    if isinstance(references, str) and references.lstrip().startswith('{'):
        references = json.loads(references)
    if references is not None:
        if not isinstance(references, dict):
            references = dict.fromkeys(protocol_names(workflow), references)
        provenance['source_ctx_hashes'] = {}
        for name, reference in references.items():
            content = json.loads(Path(reference).read_text())
            provenance['source_ctx_hashes'][name] = {key: content[key] for key in ('target_set_hash', 'ctx_set_hash') if key in content}
    binding.write_text(manifest_text(provenance))
    return path, manifest


def context_records(manifest, name):
    return {row['target']: row for row in manifest['protocols'][name]}


def check_geometry(geometry, bound):
    for key in ('ctx_length', 'ctx_frameskip'):
        if bound[key] != geometry[key]:
            raise ValueError(f'{key} differs between model and shared sampling protocol')
