"""Recompute Reacher and FetchSlide success curves from bound endpoint errors."""
import json
from pathlib import Path

import numpy as np


def endpoint_errors(result, environment):
    if environment == 'reacher':
        return np.asarray([row['round_endpoints']['25']['max_abs_joint_error_rad'] for row in result['per_target']])
    if environment == 'fetch_slide':
        return np.asarray([row['puck_xy_l2_error_m'] if 'puck_xy_l2_error_m' in row else row['endpoint_error_m']
                           for row in result['per_target']])
    raise ValueError(environment)


def success_curve(errors, thresholds, *, strict):
    values = errors[:, None]
    tolerance = np.asarray(thresholds)[None]
    successes = values < tolerance if strict else values <= tolerance
    return successes.mean(axis=0)*100


def threshold_curves(config, environment, results):
    settings = config['environments'][environment]
    thresholds = np.deg2rad(settings['grid']) if environment == 'reacher' else np.asarray(settings['grid'])/100
    records = {}
    for method,asset in settings['results'].items():
        path = Path(results)/environment/asset['method']/'0_result_planning.json'
        result = json.loads(path.read_text())
        errors = endpoint_errors(result, environment)
        rates = success_curve(errors, thresholds, strict=environment == 'reacher')
        records[method] = {'errors':errors.tolist(), 'curve_percent':rates.tolist(), 'count':len(errors),
                           'target_hashes':[row.get('target_hash') for row in result['per_target']],
                           'source':str(path), 'grid':settings['grid'], 'reference_rate':float(rates[settings['grid'].index(settings['reference'])])}
    return records


def main():
    import argparse
    from analysis.artifacts import create_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('conf/analysis/success_thresholds.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True,
                        help='Planning results laid out as <environment>/<method>/0_result_planning.json')
    parser.add_argument('--environments', nargs='+', choices=['reacher','fetch_slide'], default=['reacher','fetch_slide'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = create_output(args.output, config)
    curves = {environment:threshold_curves(config, environment, args.results) for environment in args.environments}
    (output/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')


if __name__ == '__main__':
    main()
