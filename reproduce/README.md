# Reproducing Paper Results

This folder holds the evaluation targets and context clips behind the paper's prediction and planning results. By default, evaluation samples 75 targets and their context clips from `--target-seed` and `--ctx-seed`; pass the files below to evaluate on the paper's selection instead.

Download the checkpoints and datasets first (see the [main README](../README.md)).

| Environment | Workflow | Targets | Contexts |
| --- | --- | --- | --- |
| Push-T | `nonplanning`, `planning` | `pusht/targets.json` | `pusht/contexts.json` |
| Reacher | `nonplanning`, `planning` | `reacher/targets.json` | `reacher/contexts.json` |
| BeerPong | `nonplanning` | `beerpong/targets_nonplanning.json` | `beerpong/contexts_nonplanning.json` |
| BeerPong | `planning` | `beerpong/targets_planning.json` | `beerpong/contexts_planning.json` |
| FetchSlide | `nonplanning` | `fetch_slide/targets_nonplanning.json` | `fetch_slide/contexts_nonplanning.json` |
| FetchSlide | `planning` | `fetch_slide/targets_planning.json` | `fetch_slide/contexts_planning.json` |

All methods share the same targets. The contexts are read only by methods conditioned on a context clip (`context`, `prefix_context`, `estimated`, `lewm_context`, `tc_lewm_context`); other methods ignore them.

## DINO-WM Family

```bash
pixi run python dinowm_based_evaluate.py --environment reacher --conditioning context --workflow planning \
    --target-manifest reproduce/reacher/targets.json --ctx-manifest reproduce/reacher/contexts.json
pixi run python dinowm_based_evaluate.py --environment beerpong --conditioning vanilla --workflow nonplanning \
    --target-manifest reproduce/beerpong/targets_nonplanning.json --ctx-manifest reproduce/beerpong/contexts_nonplanning.json
```

## Long History

Long History uses only the targets:

```bash
pixi run python eval_pusht.py --config-name eval_pusht_long_history phase=nonplanning \
    target_manifest=reproduce/pusht/targets.json
pixi run python eval_visual.py --config-name eval_fetch_slide_long_history phase=planning \
    target_manifest=reproduce/fetch_slide/targets_planning.json
```

## LeWM Family

```bash
pixi run python lewm_based_evaluate.py --environment pusht --method lewm_context --workflow nonplanning \
    --target-manifest reproduce/pusht/targets.json --ctx-manifest reproduce/pusht/contexts.json
```

For the analysis figures, see the Analysis section of the [main README](../README.md#analysis).
