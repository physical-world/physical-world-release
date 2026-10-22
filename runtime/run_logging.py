"""Write identical metric names, steps and image pixels to W&B and TensorBoard."""

import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb


class RunLogger:
    def __init__(self, directory, *, project, config, mode="online"):
        directory = Path(directory)
        run_id = wandb.util.generate_id()
        self.run = wandb.init(project=project, id=run_id, name=directory.name,
                              config=config, dir=str(directory), mode=mode)
        self.writer = SummaryWriter(str(directory / "tensorboard"))
        metadata = {"id": run_id, "name": directory.name, "project": project,
                    "mode": mode, "url": self.run.get_url() if mode == "online" else None}
        (directory / "wandb.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def log(self, metrics, global_step, images=None):
        """Accumulate same-step calls; the next step or close commits W&B's row."""
        values = {key: float(value) for key, value in metrics.items()}
        for key, value in values.items():
            self.writer.add_scalar(key, value, global_step, new_style=True, double_precision=True)
        for key, image in (images or {}).items():
            pixels = ((image.detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.writer.add_image(key, pixels, global_step)
            values[key] = wandb.Image(pixels.permute(1, 2, 0).numpy())
        self.run.log(values, step=global_step, commit=False)

    def close(self):
        try:
            self.writer.close()
        finally:
            self.run.finish()
