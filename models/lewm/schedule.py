"""Paper update-based warmup/cosine using the native PyTorch scheduler."""
import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer, total_updates):
    if total_updates < 1:
        raise ValueError("Training requires at least one optimizer update")
    warmup = max(1, int(0.01 * total_updates))
    def multiplier(step):
        if step < warmup:
            return step / warmup
        return (1 + math.cos(math.pi * (step - warmup)/max(1, total_updates - warmup))) / 2
    return LambdaLR(optimizer, multiplier)
