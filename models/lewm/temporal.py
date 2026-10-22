"""TC-LeWM regularizes the four-frame temporal residual, not the episode mean."""


def temporal_residual(visual):
    return visual - visual.mean(dim=1, keepdim=True)


def temporal_statistics(visual):
    visual = visual.detach().float()
    residual = temporal_residual(visual)
    energy = residual.square().mean()
    variance = visual.var(dim=(0, 1), unbiased=False).mean()
    return dict(temporal_residual_energy=energy.detach(),
                temporal_variance_ratio=(energy / variance.clamp_min(1e-12)).detach())
