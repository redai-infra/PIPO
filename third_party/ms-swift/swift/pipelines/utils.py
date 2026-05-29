# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
from datasets import load_from_disk

from swift.dataset import DatasetSyntax, sample_dataset
from swift.template import update_generation_config_eos_token
from swift.tuner_plugin import tuners_map
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()


def prepare_adapter(args, model, adapters=None):
    if args.tuner_backend == 'unsloth':
        if args.model_meta.is_multimodal:
            from unsloth import FastVisionModel as UnslothModel
        else:
            from unsloth import FastLanguageModel as UnslothModel
        UnslothModel.for_inference(model)
        return model
    if args.tuner_type in tuners_map:
        tuner = tuners_map[args.tuner_type]
    else:
        tuner = Swift
    # compat deploy
    adapters = adapters if adapters is not None else args.adapters
    for adapter in adapters:
        model = tuner.from_pretrained(model, adapter)
    if args.tuner_type == 'bone':
        # Bone has a problem of float32 matmul with bloat16 in `peft==0.14.0`
        model.to(model.dtype)
    return model


def prepare_model_template(args, **kwargs):
    adapters = kwargs.get('adapters')
    model, processor = args.get_model_processor(**kwargs)
    template = args.get_template(processor)
    if model is not None:
        if template.use_model:
            template.model = model
        model = prepare_adapter(args, model, adapters=adapters)
        update_generation_config_eos_token(model.generation_config, template)
    return model, template


def _select_dataset(dataset, max_length, min_length=None):
    if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
        # Compatible with ms-swift 3.x cache_dataset
        dataset = dataset.rename_column('length', 'lengths')
    idxs = []
    for i, length in enumerate(dataset['lengths']):
        sample_length = max(length) if isinstance(length, list) else length
        # Check max_length constraint
        if sample_length > max_length:
            continue
        # Check min_length constraint
        if min_length is not None and sample_length < min_length:
            continue
        idxs.append(i)
    new_dataset = dataset.select(idxs)
    filtered_count = len(dataset) - len(new_dataset)
    if filtered_count > 0:
        filter_msg = f'max_length <= {max_length}'
        if min_length is not None:
            filter_msg += f', min_length >= {min_length}'
        logger.info(f'Dataset filtered by ({filter_msg}), origin length: {len(dataset)}, '
                    f'filtered dataset length: {len(new_dataset)}, filtered: {filtered_count}')
    return new_dataset


def get_cached_dataset(args):
    train_datasets, val_datasets = [], []
    random_state = np.random.RandomState(args.data_seed)
    min_length = getattr(args, 'min_length', None)
    for cached_dataset, datasets in zip([args.cached_dataset, args.cached_val_dataset], [train_datasets, val_datasets]):
        for path in cached_dataset:
            if os.path.exists(path):
                dataset_sample = None
            else:
                path, dataset_sample = DatasetSyntax._safe_split(path, '#', True, 'right')
            dataset = _select_dataset(load_from_disk(path), args.max_length, min_length)
            if dataset_sample is not None:
                dataset = sample_dataset(
                    dataset, int(dataset_sample), args.dataset_shuffle, random_state=random_state, shuffle_all=True)
            datasets.append(dataset)
    return train_datasets, val_datasets
