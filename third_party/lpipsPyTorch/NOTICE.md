# Vendored LPIPS implementation

Upstream: https://github.com/S-aiueo32/lpips-pytorch (BSD 2-Clause; see LICENSE). The upstream revision used to retrieve the license and compare provenance is `016e0ae19ceff83450590738948941a7eb9b95b7`.

These four Python files are copied unchanged from the paper reference repository `dino_wm_rebase`, revision `bd20c954fbeb764a17811aa11c3fd753a5dec343`, under `metrics/lpipsPyTorch/`. The upstream lookup revision is not a claim that these files are an unmodified upstream checkout. File hashes are recorded in provenance.json.

The reference changes VGG16 construction to the explicit torchvision `IMAGENET1K_V1` weights and averages the spatially pooled, summed layer distances across the batch. Both changes are retained. Feature normalization uses epsilon outside the square root. Calibration weights come from the original PerceptualSimilarity v0.1 URLs retained in utils.py.

Production imports this vendored implementation directly; the reference repository is needed only for isolated development comparisons. Paper evaluation callers explicitly select `net_type="vgg", version="0.1"`, move the model to their device, and freeze it in evaluation mode. VGG16 internals remain provided by torchvision.
