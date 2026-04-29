#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader

logger = logging.get_logger(__name__)


@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    max_iter = cfg.TRAIN.NUM_TEST_TASKS
    top1_per_class = {}
    num_per_class = {}

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= max_iter:
            break

        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        model_dict = model(task_dict)
        target_logits = model_dict["logits"]

        loss = F.cross_entropy(
            model_dict["logits"],
            task_dict["target_labels"].long()
        ) / cfg.TRAIN.BATCH_SIZE

        labels = task_dict["target_labels"]
        preds = target_logits

        num_topks_correct = metrics.topks_correct(
            preds,
            task_dict["target_labels"],
            (1, 5),
        )

        _top_max_k_vals, top_max_k_inds = torch.topk(
            preds,
            1,
            dim=1,
            largest=True,
            sorted=True,
        )

        # per-class accuracy 統計
        for index, score in enumerate(top_max_k_inds):
            real_label = str(task_dict["real_target_labels"][index].cpu().item())

            if real_label in num_per_class:
                num_per_class[real_label] += 1
            else:
                num_per_class[real_label] = 1

            if real_label not in top1_per_class:
                top1_per_class[real_label] = 0

            if score[0] == labels[index]:
                top1_per_class[real_label] += 1

        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce(
                [loss, top1_err, top5_err]
            )

        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )

        val_meter.iter_toc()

        val_meter.update_stats(
            top1_err,
            top5_err,
            val_loader.batch_size * max(misc.get_num_gpus(cfg), 1),
        )
        val_meter.update_predictions(preds, labels)

        # 關掉原本格式，避免出現 iter 50/10000 但沒有 epoch 資訊
        # val_meter.log_iter_stats(cur_epoch, cur_iter)

        top1_acc = 100.0 - top1_err

        epoch_iter = cur_iter % cfg.TRAIN.NUM_TEST_TASKS

        if (epoch_iter + 1) % 50 == 0 or (epoch_iter + 1) == cfg.TRAIN.NUM_TEST_TASKS:
            logger.info(
                "[Test] [Iter {}/{}] loss={:.6f}, top1_acc={:.2f}".format(
                    epoch_iter + 1,
                    max_iter,
                    loss,
                    top1_acc,
                )
            )

        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)

    for perclass in top1_per_class:
        logger.info(
            "[Test][Class {}] acc={:.4f} ({}/{})".format(
                perclass,
                top1_per_class[perclass] / num_per_class[perclass],
                top1_per_class[perclass],
                num_per_class[perclass],
            )
        )

    val_meter.reset()

def test_few_shot(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("TEST with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None
    cu.load_test_checkpoint(cfg, model, model_bucket)

    # Create the video train and val loaders.
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None
    cur_epoch = 0
    test_epoch(val_loader, model, val_meter, cur_epoch, cfg)
