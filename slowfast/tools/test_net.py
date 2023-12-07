#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import sys
import pandas as pd
import time as ti

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
# import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, models, test_meter, cfg, writer=None, prompter=None):
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
        test_loader (loader): video testing loader.
        models (model): the pretrained video models to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model = models[0]
    model_2 = models[1]

    for mod in models:
        mod.eval()

    test_meter.iter_tic()

    # delete existing predictions.txt if exists
    if os.path.exists(cfg.TAL.OUTPUT_FILE_PATH.rpartition('.')[0] + "_unmerged.txt"):
        print("DELETING unmerged csv")
        os.remove(cfg.TAL.OUTPUT_FILE_PATH.rpartition('.')[0] + "_unmerged.txt")

    if os.path.exists(cfg.TAL.OUTPUT_FILE_PATH):
        print("DELETING merged csv")
        os.remove(cfg.TAL.OUTPUT_FILE_PATH)

    if cfg.TAL.ENABLE == True:
        start_time = None
        end_time = None
        video_id = None
        prev_agg_pred = None
        prev_consol_codes = []

        # stores aggregated clips (batches) for each cam view type
        cam_view_clips = {}

        # num clips aggregated for current temporal action interval
        clip_agg_cnt = 0

        # num of aggregated frames to keep from last iteration
        frame_agg_threshold = cfg.TAL.CLIP_AGG_THRESHOLD - cfg.DATA.NUM_FRAMES 

        # initialize matrix with weighted action probs for each cam view (weights taken from Purdue's M2DAR Submission https://arxiv.org/abs/2305.08877)
        weights_df = pd.read_csv(os.getcwd() + '/inference/weighted_cam_view_action_probs.csv')
        cam_view_weights = { col:weights_df[col].to_numpy() for col in weights_df.columns }  

        short_seg_filtering_thresholds = pd.read_csv(os.getcwd() + '/inference/weighted_action_filter_gaussian_probs.csv', names=['Probs'])['Probs'].to_list()

        # stores all prob mats for single proposals
        consolidated_prop_prob_mats = [] 

        # for recording FPS metric
        video_exec_times = []
        start_t = ti.time()


    for cur_iter, (inputs, labels, video_idx, time, meta, proposal) in enumerate(
        test_loader
    ):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()

            if cfg.TEST.DATASET != "AI-City-Track-3":
                video_idx = video_idx.cuda()

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)

        else:
            # Perform the forward pass.
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1 and cfg.TAL.ENABLE == False:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS and cfg.TAL.ENABLE == False:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE and cfg.TAL.ENABLE == False:
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )

        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()

    if cfg.TAL.ENABLE == False:
        # Log epoch stats and print the final testing results.
        if not cfg.DETECTION.ENABLE:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()
            if writer is not None:
                writer.plot_eval(preds=all_preds, labels=all_labels)

            if cfg.TEST.SAVE_RESULTS_PATH != "":
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

                if du.is_root_proc():
                    with pathmgr.open(save_path, "wb") as f:
                        pickle.dump([all_preds, all_labels], f)

                logger.info(
                    "Successfully saved prediction results to {}".format(save_path)
                )

        test_meter.finalize_metrics()
    
    logger.info("Inference complete.")

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg, 0)

        if cfg.TAL.ENABLE == True and cfg.TAL.USE_2_GPUS == True:
            model_2 = build_model(cfg, 1) # build same model on 2nd GPU
        else:
            model_2 = model # reference same model, 1st GPU

        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)

        if cfg.TAL.ENABLE == True and cfg.TAL.USE_2_GPUS == True:
            cu.load_test_checkpoint(cfg, model_2)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            if cfg.TEST.DATASET == "AI-City-Track-3":
                num_videos = len(test_loader.dataset.dataset_samples)
            else:
                num_videos = test_loader.dataset.num_videos 

            assert (
                num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES
                if not cfg.TASK == "ssl"
                else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
                cfg.LOG_PERIOD,
            )

        prompter = None

        # # Set up writer for logging to Tensorboard format.
        # if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        #     cfg.NUM_GPUS * cfg.NUM_SHARDS
        # ):
        #     writer = tb.TensorboardWriter(cfg)
        # else:
        writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, [model, model_2], test_meter, cfg, writer, prompter)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    if(cfg.TAL.ENABLE == False):
        for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
            logger.info(
                "Finalized testing with {} temporal clips and {} spatial crops".format(
                    view, cfg.TEST.NUM_SPATIAL_CROPS
                )
            )
            result_string_views += "_{}a{}" "".format(
                view, test_meter.stats["top1_acc"]
            )

            result_string = (
                "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
                "".format(
                    params / 1e6,
                    flops,
                    view,
                    test_meter.stats["top1_acc"],
                    test_meter.stats["top5_acc"],
                    misc.gpu_mem_usage(),
                    flops,
                )
            )

            logger.info("{}".format(result_string))
        logger.info("{}".format(result_string_views))
    else:
        return 
    
    return result_string + " \n " + result_string_views