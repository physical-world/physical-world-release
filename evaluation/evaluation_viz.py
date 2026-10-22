"""Historical non-planning GT / prediction / GT-mask sample layout."""

import torch
from torchvision.utils import make_grid, save_image

from evaluation.segmentation import overlay_masks


def save_sample_grid(path, predicted, target, masks, initial, num_samples=2, max_cols=8):
    """Keep initial frames, blank initial overlays, white padding and red dividers."""
    predicted, target, masks, initial = [x[:num_samples] for x in (predicted, target, masks, initial)]
    init = ((initial + 1) / 2).clamp(0, 1)
    rows = [torch.cat((init, ((target + 1) / 2).clamp(0, 1)), 1),
            torch.cat((init, ((predicted + 1) / 2).clamp(0, 1)), 1),
            torch.cat((torch.zeros_like(init), overlay_masks(predicted, masks)), 1)]
    rows = [row.flatten(0, 1) for row in rows]
    columns = len(rows[0])
    if columns <= max_cols:
        save_image(torch.cat(rows), path, nrow=columns)
        return
    grids = []
    for start in range(0, columns, max_cols):
        chunks = []
        for row in rows:
            chunk = row[start:start + max_cols]
            chunks.append(torch.cat((chunk, torch.ones_like(row[:1]).expand(max_cols - len(chunk), -1, -1, -1))))
        grids.append(make_grid(torch.cat(chunks), nrow=max_cols, padding=2, pad_value=1))
    divider = grids[0].new_tensor([1, 0, 0]).view(3, 1, 1).expand(-1, 4, grids[0].shape[-1])
    parts = [part for grid in grids[:-1] for part in (grid, divider)] + grids[-1:]
    save_image(torch.cat(parts, -2), path)
