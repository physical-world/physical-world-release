"""Create fresh analysis outputs and record execution provenance."""
import json
import os
import platform
import subprocess
from pathlib import Path


def create_output(path, config):
    path = Path(path).resolve()
    path.relative_to(Path('logs').resolve())
    path.mkdir(parents=True, exist_ok=False)
    (path/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
    except OSError:
        revision = None
    metadata = {'job_id':os.environ.get('SLURM_JOB_ID'), 'host':platform.node(),
                'source_revision':revision.stdout.strip() if revision is not None and revision.returncode == 0 else None}
    (path/'execution.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return path
