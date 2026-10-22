"""Read saved inference and planning measurements without loading models."""
import json
from pathlib import Path
from statistics import fmean


def read_run_timings(inference_run, planning_run, methods):
    rows = []
    for method in methods:
        measurements = []
        for directory, metric in [(inference_run, 'total_h10'), (planning_run, 'solve')]:
            path = Path(directory) / f'{method}.json'
            payload = json.loads(path.read_text())
            try:
                if payload['method'] != method:
                    raise ValueError(f'{path}: expected method {method}')
                measurement = payload['latency'][metric]
                median = measurement['median_ms']
                mean = fmean(measurement['raw_ms'])
                measurements.append((median, mean))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}: invalid {method}/{metric} timing: {error}') from error
        prediction, planning = measurements
        rows.append({'method': method, 'rollout_ms': prediction[0],
                     'planning_s': planning[0] / 1000,
                     'rollout_mean_ms': prediction[1], 'planning_mean_s': planning[1] / 1000})
    return rows
