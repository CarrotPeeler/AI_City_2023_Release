#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view tal for video classification models"""

import numpy as np
import os
import torch
import pandas as pd
import time as ti

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from torch.utils.data import DataLoader
from slowfast.models import build_model
from slowfast.datasets import utils as slowfast_utils
from inference.ProposalDataset import ProposalDataset
from tqdm.auto import tqdm
from glob import glob

pd.options.mode.chained_assignment = None  # default='warn'

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_video_id_tal(cfg, video_id, models_dict, tal_loader):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        model_dict: dict with list of the pretrained video models to test by camera view
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        video_id (int): id of the video
        video_paths (list(str)): list of string paths to videos for given id
    """
    # Enable eval mode
    for view_models in models_dict.values():
        for model in view_models:
            model.eval()

    probs_dict = {
        "Dashboard": [list() for i in range(len(models_dict["Dashboard"]))],
        "Rear_view": [list() for i in range(len(models_dict["Rear_view"]))],
        "Right_side_window": [list() for i in range(len(models_dict["Right_side_window"]))]
    }

    for cur_iter, (prop_frames_dict, index) in enumerate(
        tal_loader
    ):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            for cam_view in prop_frames_dict:    
                if isinstance(prop_frames_dict[cam_view], list):
                    prop_frames_dict[cam_view] = prop_frames_dict[cam_view][0]
                prop_frames_dict[cam_view] = prop_frames_dict[cam_view].cuda(non_blocking=True)

        # PREDICTION STEP: make predictions for each camera angle using identical proposal length and space
        prop_probs = predict_cam_views(cfg, models_dict, prop_frames_dict)

        for view, folds in prop_probs.items():
            for fold, probs in enumerate(folds):
                probs_dict[view][fold].append(probs)

    # save probs for each camera view, and each fold if performing K-Fold CV
    for view, folds in probs_dict.items():
        for fold, video_probs in enumerate(folds):
            video_probs = np.vstack(video_probs)
            with open(f"./inference/probs/{view.lower()}/video_{video_id}_{view.lower()}_fold_{fold}_probs.npy", "wb") as f:
                np.save(f, video_probs)


def tal(cfg):
    """
    Perform multi-view tal on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if not os.path.exists("./inference/probs"):
        print("CREATING probs folder")
        os.mkdir("./inference/probs")
    
    if not os.path.exists("./inference/probs/dashboard"):
        print("CREATING dashboard folder")
        os.mkdir("./inference/probs/dashboard")

    if not os.path.exists("./inference/probs/rear_view"):
        print("CREATING rear view folder ")
        os.mkdir("./inference/probs/rear_view")

    if not os.path.exists("./inference/probs/right_side_window"):
        print("CREATING right side window folder")
        os.mkdir("./inference/probs/right_side_window")

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # build models
    models = build_models(cfg)

    # gather testing videos
    proposals_df = pd.read_csv(cfg.TAL.PROPOSAL_CSV_PATH, sep=" ")

    # run tal for each video_id in the test dataset
    for i in tqdm(range(len(proposals_df["video_id"].unique()))):
        video_id = i + 1
        video_id_proposals_df = proposals_df.loc[proposals_df["video_id"] == video_id]

        tal_loader = create_tal_loader(cfg, video_id_proposals_df)

        # run TAL simultaneously for all videos belonging to same video id
        perform_video_id_tal(cfg, video_id, models, tal_loader)

    logger.info("TAL complete.")
    return


def create_tal_loader(cfg, proposals_df):
    """create dataset and dataloader object for given videos"""
    # create proposal dataset
    tal_dataset = ProposalDataset(
        cfg,
        anno_df=proposals_df
    )

    sampler = slowfast_utils.create_sampler(tal_dataset, False, cfg)

    # load proposal dataset
    tal_loader = DataLoader(
                tal_dataset,
                sampler=sampler,
                shuffle=False,
                batch_size=1, 
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
                persistent_workers=True)
    
    return tal_loader


def build_models(cfg):
    """
    Build all models for each checkpoint available

    params:
        cfg: config file
    returns:
        dict of models for each camera view
    """

    # create dict for models
    models = {"Dashboard": [],
              "Rear_view": [],
              "Right_side_window": []}
    
    view_ckpt_dirs = {"Dashboard": cfg.TAL.CHECKPOINTS_DIR_DASH,
                      "Rear_view": cfg.TAL.CHECKPOINTS_DIR_REAR, 
                      "Right_side_window": cfg.TAL.CHECKPOINTS_DIR_RIGHT}
    
    # for each camera view, find all ckpts
    for view, ckpt_dir in view_ckpt_dirs.items():
        subfolders = list(map(lambda x: ckpt_dir + '/' + x, os.listdir(ckpt_dir)))
        assert len(subfolders) > 0, "Error: no checkpoints found. Make sure checkpoints are in correct folder."

        # check for 5-Fold CV configuration
        if len(subfolders) > 1 and all(["fold" in x for x in subfolders]):
            logger.info("Loading checkpoints for K-Fold Cross Validation...")
            for fold_dir in subfolders:
                model = find_and_load_checkpoint(cfg, fold_dir)
                models[view].append(model)
        else:
            logger.info("Loading single checkpoint per view...")
            model = find_and_load_checkpoint(cfg, ckpt_dir)
            models[view].append(model)
    
    return models


def find_and_load_checkpoint(cfg, dir):
    """
    Given a directory, search for a .pt file to load

    params:
        cfg: config used to create model
        dir: directory to search
    returns:
        two models (diff. GPUs) loaded with the found checkpoint
    """
    ckpt_types = [".pt",".pyth"]
    ckpt_path = []
    for ckpt_type in ckpt_types:
        ckpt_path.extend(glob(f"{dir}/**/*{ckpt_type}", recursive=True))
    assert len(ckpt_path) > 0, "Error: no checkpoints found. Make sure checkpoint is .pt, .pyth and in correct folder."
    assert len(ckpt_path) == 1, "Error: found more than one checkpoint to load. Please make sure only one checkpoint exists."
    
    model = build_model(cfg, 0)
    
    # load model checkpoints
    cu.load_test_checkpoint(cfg, model, ckpt_path[0])

    return model


def predict_cam_views(cfg, models_dict, frames_dict):
    """
    Performs prediction step over all three camera views for a single proposal and/or aggregation of frames

    params:
        cfg: cfg object
        models_dict: dict with pytorch models for each camera view
        frames_dict: dict containing frames from a single proposal for each camera view
    returns:
        dict containing list of prediction probabilities matrices for each camera view
    """
    prop_probs = {"Dashboard": [],
                  "Rear_view": [],
                  "Right_side_window": []}

    for cam_view_type in frames_dict.keys():
        view_models = models_dict[cam_view_type]
        prop_frames = frames_dict[cam_view_type]

        # get predictions from each model (more than one if K-Fold CV)
        for model in view_models:
            prop_input = [prop_frames]

            cam_view_prop_probs = model(prop_input)

            if cfg.MODEL.MODEL_NAME == 'VideoMAEv2':
                cam_view_prop_probs = torch.softmax(cam_view_prop_probs, dim=-1)

            cam_view_prop_probs = cam_view_prop_probs.cpu().numpy()

            prop_probs[cam_view_type].append(cam_view_prop_probs)
            
    return prop_probs


