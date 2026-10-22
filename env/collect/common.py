"""Small shared output operations; each collector retains its own data semantics."""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from numcodecs import Blosc


def collection_indices(indices, count, name):
    if type(count) is not int or count <= 0:
        raise ValueError(f'{name}: population size must be a positive integer')
    indices = list(range(count) if indices is None else indices)
    if any(type(index) is not int or not 0 <= index < count for index in indices):
        raise ValueError(f'{name} must contain integer indices in [0, {count})')
    return indices


def prepare_output(output):
    output = Path(output).resolve()
    output.relative_to((Path(__file__).resolve().parents[2] / 'logs').resolve())
    sidecars = [output.with_suffix('.manifest.json'), output.with_suffix('.collection.json'),
                output.with_name(output.stem + '_mask.zarr'), output.with_suffix('.collection.log')]
    for path in [output, *sidecars]:
        if path.exists():
            raise FileExistsError(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


class CollectionLog:
    """One progress line per dynamics group, shared by CLI and notebook callers."""

    def __init__(self, output, env, groups, episodes, group_name='dynamics_id', *, directory=False):
        self.output = Path(output)
        self.env, self.groups, self.episodes = env, groups, episodes
        self.group_name = group_name
        self.directory = directory
        self.group = self.episode = None
        self.completed_groups = self.completed_episodes = self.frames = 0
        self.logger = logging.Logger('env.collect', level=logging.INFO)

    def __enter__(self):
        self.started = perf_counter()
        path = (self.output / 'collection.log' if self.directory
                else self.output.with_suffix('.collection.log'))
        file_handler = logging.FileHandler(path, mode='x', encoding='utf-8')
        for handler in (logging.StreamHandler(sys.stdout), file_handler):
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.logger.addHandler(handler)
        self.logger.info('start env=%s groups=%d episodes=%d output=%s log=%s',
                         self.env, self.groups, self.episodes, self.output, path)
        return self

    def start_group(self, group):
        self.group, self.episode = group, None
        self.group_started = perf_counter()

    def complete_group(self, episodes, frames):
        self.completed_groups += 1
        self.completed_episodes += episodes
        self.frames += frames
        self.logger.info('done=%d/%d %s=%s episodes=%d frames=%d elapsed=%.1fs',
                         self.completed_groups, self.groups, self.group_name, self.group,
                         episodes, frames, perf_counter() - self.group_started)
        self.episode = None

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                manifest = (self.output / 'manifest.json' if self.directory
                            else self.output.with_suffix('.manifest.json'))
                self.logger.info('finished groups=%d episodes=%d frames=%d elapsed=%.1fs manifest=%s',
                                 self.completed_groups, self.completed_episodes, self.frames,
                                 perf_counter() - self.started, manifest)
            else:
                self.logger.error('failed %s=%s episode=%s completed_groups=%d/%d elapsed=%.1fs',
                                  self.group_name, self.group, self.episode,
                                  self.completed_groups, self.groups, perf_counter() - self.started,
                                  exc_info=(exc_type, exc_value, traceback))
        finally:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()


def append_arrays(group, arrays, chunk_rows=64):
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    for name, values in arrays.items():
        values = np.asarray(values)
        if name not in group:
            group.create_dataset(name, shape=(0,) + values.shape[1:], dtype=values.dtype,
                                 chunks=(min(chunk_rows, len(values)),) + values.shape[1:],
                                 compressor=compressor)
        group[name].append(values, axis=0)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def save_run(output, config):
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                              capture_output=True, check=False).stdout.strip() or None
    write_json(Path(output).with_suffix('.collection.json'),
               dict(config=config, source_revision=revision, slurm_job_id=os.environ.get('SLURM_JOB_ID')))


def append_episode(root, episode, metadata):
    append_arrays(root.require_group('data'), episode)
    append_arrays(root.require_group('meta'),
                  {name: np.asarray([value]) for name, value in metadata.items()})
