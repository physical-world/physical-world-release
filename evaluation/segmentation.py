"""The bound three-class PushT UNet evaluation asset and foreground IoU."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(incoming, outgoing))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(incoming + skip, outgoing)

    def forward(self, deep, skip):
        deep = self.up(deep)
        dy, dx = skip.shape[2] - deep.shape[2], skip.shape[3] - deep.shape[3]
        deep = F.pad(deep, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat((skip, deep), 1))


class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.downs = nn.ModuleList([Down(a, b) for a, b in ((64, 128), (128, 256), (256, 512))])
        self.ups = nn.ModuleList([Up(a, b, b) for a, b in ((512, 256), (256, 128), (128, 64))])
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.out_conv(x)


def load_segmentation_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet(num_classes=checkpoint["model"]["out_conv.weight"].shape[0]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().requires_grad_(False), checkpoint


def segmentation_input(images):
    flat = ((images.reshape(-1, *images.shape[-3:]) + 1) / 2).clamp(0, 1)
    mean = flat.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = flat.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (flat - mean) / std


def segment_images(model, images):
    return model(segmentation_input(images)).argmax(1).reshape(*images.shape[:-3], *images.shape[-2:])


def remap_masks(raw):
    if raw.numel() and (raw.min() < 0 or raw.max() > 3):
        raise ValueError("PushT masks must contain labels 0..3")
    return torch.tensor([0, 0, 1, 2], device=raw.device)[raw.long()]


def iou_counts(predicted, target, classes=((1, "block"), (2, "agent"))):
    counts = {}
    regions = [(name, predicted == label, target == label) for label, name in classes]
    for name, a, b in [*regions, ("overall", predicted > 0, target > 0)]:
        counts[name] = dict(intersection=int((a & b).sum()), union=int((a | b).sum()))
    return counts


def iou_from_counts(counts):
    return {name: row["intersection"] / row["union"] if row["union"] else 1.0 for name, row in counts.items()}


def read_evaluation_masks(manifest, sample):
    """Read recorded masks from the bound shard sidecar at the target indices."""
    path, _, begin, _ = manifest.episode_info(sample["episode_id"])
    row = next(row for row in manifest.shards if manifest.path.parent / row["path"] == path)
    masks = manifest.group(manifest.path.parent / row["mask_path"])["data/img_mask"]
    return torch.from_numpy(masks.get_orthogonal_selection((sample["target_frames"] + begin).numpy()))


def overlay_masks(images, masks, alpha=.55):
    colors = images.new_tensor([[0, 0, 0], [1, .2, .2], [1, .95, 0]])
    rgb = ((images + 1) / 2).clamp(0, 1)
    colored = colors[masks.long()].movedim(-1, -3)
    return torch.where((masks > 0).unsqueeze(-3), rgb * (1 - alpha) + colored * alpha, rgb).clamp(0, 1)
