"""Deterministic evaluation selection, independent of model and global RNG state."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def manifest_text(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def manifest_hash(value):
    return hashlib.sha256(manifest_text(value).encode()).hexdigest()


def with_hash(value, field='target_set_hash'):
    content = {key: item for key, item in value.items() if key != field}
    return dict(content, **{field: manifest_hash(content)})


def local_rng(seed, episode, purpose):
    # Stable across processes; Python's salted hash() must not seed sampling.
    digest = hashlib.sha256(f'{int(seed)}:{int(episode)}:{purpose}'.encode()).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], 'little'))


def sampling_protocol(environment, path=None):
    path = path or Path(__file__).parents[1] / 'conf' / 'sampling' / f'{environment}.yaml'
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def collection_identity(collection):
    metadata = collection.metadata
    return metadata.get('episode_identity_sha256') or manifest_hash(metadata)


def load_collection(environment, path):
    if environment == 'pusht':
        from datasets.pusht import PushTManifest as Collection
    elif environment == 'reacher':
        from datasets.reacher import ReacherEpisodes as Collection
    elif environment == 'fetch_slide':
        from datasets.fetch_slide import FetchSlideEpisodes as Collection
    elif environment == 'beerpong':
        from datasets.beerpong import BeerPongEpisodes as Collection
    else:
        raise ValueError(environment)
    return Collection(path)


def target_episode_ids(collection, manifest):
    if manifest['environment'] == 'pusht':
        from evaluation.common import evaluation_episode_ids
        return evaluation_episode_ids(collection, manifest)
    return [row['source_episode'] for row in manifest['targets']]


def check_target_geometry(manifest, frameskip):
    if manifest['protocol']['frameskip'] != frameskip:
        raise ValueError('Target frameskip differs from the evaluation reader and model')


def generate_targets(collection, environment, workflow, protocol, num_targets=75, seed=42):
    from datasets.target_protocols import target_candidates, target_record
    ids = sorted(collection.split_episode_ids()['validation'])
    groups = defaultdict(list)
    for episode, condition in target_candidates(collection, environment, workflow, protocol, ids):
        groups[condition].append(episode)
    available = sum(map(len, groups.values()))
    if num_targets < 1 or num_targets > available:
        raise ValueError(f'Requested {num_targets} targets from {available} candidates')
    queues = []
    for condition in sorted(groups):
        candidates = sorted(groups[condition])
        order = local_rng(seed, condition, 'target-order').permutation(len(candidates))
        queues.append(iter([candidates[int(index)] for index in order]))
    records = []
    while queues and len(records) < num_targets:
        remaining = []
        for queue in queues:
            record = None
            for episode in queue:
                record = target_record(collection, environment, workflow, protocol, episode,
                                       local_rng(seed, episode, 'target-offset'))
                if record is not None:
                    break
            if record is not None:
                records.append(record)
                remaining.append(queue)
            if len(records) == num_targets:
                break
        queues = remaining
    if len(records) != num_targets:
        raise ValueError(f'Only {len(records)} eligible targets; requested {num_targets}')
    return with_hash(dict(environment=environment, data_identity=collection_identity(collection),
                          split='validation', split_identity=manifest_hash(ids), protocol=protocol,
                          target_seed=int(seed), targets=records))


def resolve_targets(collection, environment, workflow, protocol, path=None, num_targets=None, seed=42):
    if num_targets is not None and num_targets < 1:
        raise ValueError('num_targets must be positive')
    if path is None:
        return generate_targets(collection, environment, workflow, protocol,
                                75 if num_targets is None else num_targets, seed)
    source = json.loads(Path(path).read_text())
    count = len(source['targets']) if num_targets is None else num_targets
    if count < 1 or count > len(source['targets']):
        raise ValueError('num_targets exceeds the nonempty target manifest')
    if 'data_identity' in source:
        if source['environment'] != environment or source['data_identity'] != collection_identity(collection):
            raise ValueError('Target manifest is bound to another collection')
        if source['split_identity'] != manifest_hash(sorted(collection.split_episode_ids()['validation'])):
            raise ValueError('Target manifest is bound to another split')
        result = dict(source, targets=source['targets'][:count])
    else:
        # Preserve the bound record semantics, including its original temporal geometry.
        geometry = dict(protocol)
        for key in geometry:
            if key in source:
                geometry[key] = source[key]
        result = dict(environment=environment, data_identity=collection_identity(collection),
                      split='validation', split_identity=manifest_hash(sorted(collection.split_episode_ids()['validation'])),
                      protocol=geometry, targets=source['targets'][:count])
    episodes = target_episode_ids(collection, result)
    valid = set(collection.split_episode_ids()['validation'])
    if len(set(episodes)) != len(episodes) or any(episode not in valid for episode in episodes):
        raise ValueError('Targets must be unique validation episodes')
    return with_hash(result)


def selection_options(options):
    values = {}
    for new, old in (('target_manifest', 'targets'), ('num_targets', 'max_targets')):
        current, previous = getattr(options, new, None), getattr(options, old, None)
        if current is not None and previous is not None:
            raise ValueError(f'Specify only {new} or its alias {old}')
        values[new] = current if current is not None else previous
    return values


def add_sampling_arguments(parser):
    parser.add_argument('--target-manifest', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--num-targets', type=int)
    parser.add_argument('--max-targets', type=int)
    parser.add_argument('--target-seed', type=int, help='Target sampling seed; defaults to --eval-seed')
    parser.add_argument('--ctx-manifest', help='Context JSON path or JSON mapping of protocol names to reference paths')
    parser.add_argument('--ctx-seed', type=int, help='Context sampling seed; defaults to --eval-seed')
    parser.add_argument('--sampling-protocol', type=Path)


def prepare_targets(options, collection, environment, workflow, run):
    selected = selection_options(options)
    count = selected['num_targets']
    if count is not None and count < 1:
        raise ValueError('num_targets must be positive')
    if getattr(options, 'inspection', False):
        count = 1
    protocol = sampling_protocol(environment, getattr(options, 'sampling_protocol', None))
    seed = options.eval_seed if options.target_seed is None else options.target_seed
    manifest = resolve_targets(collection, environment, workflow, protocol['target'][workflow],
                               selected['target_manifest'], count, seed)
    path = Path(run) / 'target_manifest.json'
    with path.open('x') as stream:
        stream.write(manifest_text(manifest))
    provenance = dict(target_manifest=None if selected['target_manifest'] is None else str(Path(selected['target_manifest']).resolve()))
    if selected['target_manifest'] is not None:
        source = json.loads(Path(selected['target_manifest']).read_text())
        provenance['source_target_set_hash'] = source.get('target_set_hash')
    (Path(run) / 'sampling_sources.json').write_text(manifest_text(provenance))
    return path, manifest, protocol
