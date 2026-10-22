"""Canonical released windows for the paper LeWM training configurations."""
from torch.utils.data import Subset
from datasets.pusht import PushTManifest, PushTDataset, pusht_normalization, pusht_dataloader


def build_datasets(asset, bound, inspection=False):
    environment = bound.data.wthe_env
    frameskip = 15 if environment == 'beerpong' else 5
    if environment == 'fetch_slide':
        from datasets.fetch_slide import FetchSlideEpisodes, FetchSlideDataset, fetch_normalization
        collection = FetchSlideEpisodes(asset.manifest)
        normalization = fetch_normalization(collection)
        dataset_class = FetchSlideDataset
    elif environment == 'beerpong':
        from datasets.beerpong import BeerPongEpisodes, BeerPongDataset, beerpong_normalization
        collection = BeerPongEpisodes(asset.manifest)
        normalization = beerpong_normalization(collection)
        dataset_class = BeerPongDataset
    elif environment == 'reacher':
        from datasets.reacher import ReacherEpisodes, ReacherDataset, reacher_normalization
        collection = ReacherEpisodes(asset.manifest)
        normalization = reacher_normalization(collection)
        dataset_class = ReacherDataset
    else:
        collection = PushTManifest(asset.manifest)
        normalization = pusht_normalization(collection)
        dataset_class = PushTDataset
    if environment == 'fetch_slide':
        from datasets.fetch_slide import fetch_episode_ids
        splits = fetch_episode_ids(collection)
    else:
        splits = collection.split_episode_ids()
    datasets = {}
    for phase, ids in splits.items():
        selected = ids[:1] if inspection else ids
        windows = dataset_class(collection, selected, bound.history_size + bound.num_preds,
                                frameskip, bound.img_size, normalization)
        dataset = windows
        if bound.data.get('context', False):
            from datasets.context import (ContextDataset, ContextSampler, PushTContexts,
                                          read_pusht_context, read_reacher_context,
                                          read_fetch_context, read_beerpong_context)
            length = frameskip * 12
            if environment == 'pusht':
                sampler = PushTContexts(collection, ids, length, frameskip)
                reader = read_pusht_context
            else:
                sampler = ContextSampler(collection, ids, collection.dynamics[ids], length, frameskip,
                                         fixed_start=None if environment == 'reacher' else 0,
                                         min_episode_length=40 if environment == 'fetch_slide' else length)
                reader = (read_reacher_context if environment == 'reacher' else
                          read_fetch_context if environment == 'fetch_slide' else read_beerpong_context)
            dataset = ContextDataset(windows, sampler, reader, bound.img_size, normalization)
        count = 2 * bound.trainer.get('accumulate_grad_batches', 1) if phase == 'train' else 2
        datasets[phase] = Subset(dataset, range(count)) if inspection else dataset
    return collection, normalization, datasets
