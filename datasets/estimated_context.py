"""Read the paper's fixed target-to-donor assignments in canonical episode IDs."""
import json
from pathlib import Path

from datasets.context import pusht_episode_values


def fixed_context_records(path,collection,metadata,target_ids=None):
    reference = json.loads(Path(path).read_text())
    targets = reference.get('per_target') or reference['targets']
    environment = metadata['environment']
    validation = collection.split_episode_ids()['validation']
    if environment=='pusht':
        assignments = [(row['target_source_episode'],row['context_source_episode'],row['context_start'],
                         row['ctx_length'],row['ctx_frameskip']) for row in reference['contexts']]
        assignments = [(validation[target],validation[donor],start,length,stride)
                       for target,donor,start,length,stride in assignments]
    else:
        contexts = reference.get('conditions') or [target['condition'] for target in targets]
        assignments = [(target['source_episode'],context['source_episode'],context['start'],
                         context.get('length',context.get('span_raw_rows')),context['frameskip'])
                       for target,context in zip(targets,contexts)]
    declared = {validation[row['source_episode']] if environment=='pusht' else row['source_episode'] for row in targets}
    if {row[0] for row in assignments}!=declared or len(assignments)!=len(targets):
        raise ValueError('Fixed Context target identities differ from the reference targets')
    records = {}
    expected = metadata['dataset']
    if target_ids is not None:
        requested = set(target_ids)
        available = {row[0] for row in assignments}
        if not requested <= available:
            raise KeyError(f'Missing Context targets: {sorted(requested-available)}')
        assignments = [row for row in assignments if row[0] in requested]
    for target,donor,start,length,stride in assignments:
        if target in records:
            raise ValueError('Fixed Context requires unique targets')
        if (length,stride)!=(expected['ctx_length'],expected['ctx_frameskip']):
            raise ValueError('Fixed Context geometry differs from the estimator binding')
        records[target] = dict(target=int(target),donor=int(donor),start=int(start))
    validate_context_records(collection,metadata,records)
    if target_ids is None and len(records)!=len(targets):
        raise ValueError('Fixed Context assignments do not cover the paper targets')
    return records


def validate_context_records(collection,metadata,records):
    """Validate restored donors once at the manifest loading boundary."""
    environment = metadata['environment']
    expected = metadata['dataset']
    valid = set(collection.split_episode_ids()['validation'])
    length = expected['ctx_length']
    minimum = expected.get('min_episode_length',40 if environment=='fetch_slide' else length)
    for target,row in records.items():
        donor,start = row['donor'],row['start']
        if target==donor or target not in valid or donor not in valid:
            raise ValueError('Fixed Context requires distinct validation donors')
        _,_,begin,end = collection.episode_info(donor)
        if end-begin<minimum or not 0<=start<=max(0,end-begin-length):
            raise ValueError('Fixed Context donor length or start differs from the bound protocol')
        if expected.get('fixed_start') is not None and start!=expected['fixed_start']:
            raise ValueError('Fixed Context start differs from the bound protocol')
    if environment=='pusht':
        ids = [episode for record in records.values() for episode in (record['target'],record['donor'])]
        conditions = pusht_episode_values(collection,ids,'cog_seed').reshape(-1,2)
        matching = (conditions[:,0]==conditions[:,1]).all()
    else:
        matching = all(collection.dynamics[row['target']]==collection.dynamics[row['donor']] for row in records.values())
    if not matching:
        raise ValueError('Fixed Context donor has different simulator physics')
