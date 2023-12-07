#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from fvcore.common.registry import Registry
from VideoMAEv2.dataset.datasets import RawFrameClsDataset, VideoClsDataset

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()

    if dataset_name == "AI-City-Track-3":
        data_path = cfg.OUTPUT_DIR
        anno_path = os.path.join(data_path, split + '.csv')

        cfg.sample_mode = cfg.DATA.SAMPLE_MODE
        cfg.img_diff_json_path = cfg.DATA.IMG_DIFF_JSON_PATH

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=data_path,
            mode=split if split != 'val' else 'validation',
            clip_len=cfg.DATA.NUM_FRAMES,
            frame_sample_rate=cfg.DATA.SAMPLING_RATE,
            num_segment=1,
            test_num_segment=cfg.TEST.NUM_ENSEMBLE_VIEWS,
            test_num_crop=cfg.TEST.NUM_SPATIAL_CROPS,
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=cfg.DATA.TEST_CROP_SIZE,
            short_side_size=cfg.DATA.TEST_CROP_SIZE,
            new_height=256,
            new_width=320,
            args=cfg)
        
        return dataset
    
    return DATASET_REGISTRY.get(name)(cfg, split)
