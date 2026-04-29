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
import math
import torch.nn as nn
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import TrainMeter, ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, start_epoch, cfg, val_meter, val_loader):
    model.train()

    model_bucket = None
    start_iter = start_epoch * cfg.SOLVER.STEPS_ITER
    max_iter = cfg.TRAIN.NUM_TRAIN_TASKS

    logger.info(
        "Resume training from epoch {} / {}, global_iter {} / {}.".format(
            start_epoch + 1,
            cfg.SOLVER.MAX_EPOCH,
            start_iter,
            max_iter,
        )
    )

    best_metric = cu.load_best_metric(cfg.OUTPUT_DIR)
    logger.info("Current best_metric before training: {:.4f}".format(best_metric))

    train_meter.iter_tic()

    # 剩下還要跑幾個 task
    remaining_iter = max_iter - start_iter

    for local_iter, task_dict in enumerate(train_loader):
        if local_iter >= remaining_iter:
            break

        # 真正用於 lr / epoch / log / checkpoint 的全域 iter
        cur_iter = start_iter + local_iter

        cur_epoch = cur_iter // cfg.SOLVER.STEPS_ITER
        epoch_iter = cur_iter % cfg.SOLVER.STEPS_ITER

        if cur_epoch >= cfg.SOLVER.MAX_EPOCH:
            break

        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        lr = optim.get_epoch_lr(float(cur_iter) / cfg.SOLVER.STEPS_ITER, cfg)
        optim.set_lr(optimizer, lr)

        model_dict = model(task_dict)
        target_logits = model_dict["logits"]

        if hasattr(cfg.TRAIN, "USE_CLASSIFICATION_VALUE"):
            loss = (
                F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
                + cfg.TRAIN.USE_CLASSIFICATION_VALUE
                * F.cross_entropy(
                    model_dict["class_logits"],
                    torch.cat(
                        [task_dict["real_support_labels"], task_dict["real_target_labels"]],
                        0,
                    ).long(),
                )
            ) / cfg.TRAIN.BATCH_SIZE
        else:
            loss = F.cross_entropy(
                model_dict["logits"],
                task_dict["target_labels"].long()
            ) / cfg.TRAIN.BATCH_SIZE

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        loss.backward(retain_graph=False)

        if (cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0:
            optimizer.step()
            optimizer.zero_grad()

        preds = target_logits
        num_topks_correct = metrics.topks_correct(
            preds,
            task_dict["target_labels"],
            (1, 5),
        )
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )

        train_meter.iter_toc()
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            train_loader.batch_size * max(misc.get_num_gpus(cfg), 1),
        )

        top1_acc = 100.0 - top1_err

        if (epoch_iter + 1) % 50 == 0 or (epoch_iter + 1) == cfg.SOLVER.STEPS_ITER:
            logger.info(
                "[Epoch {}/{}] [Iter {}/{}] lr={:.6g}, loss={:.6f}, top1_acc={:.2f}".format(
                    cur_epoch + 1,
                    cfg.SOLVER.MAX_EPOCH,
                    cur_iter + 1,
                    max_iter,
                    lr,
                    loss,
                    top1_acc,
                )
            )

        train_meter.iter_tic()

        if (cur_iter + 1) % cfg.SOLVER.STEPS_ITER == 0:
            finished_epoch = cur_epoch

            train_meter.log_epoch_stats(finished_epoch)
            train_meter.reset()

            cu.save_last_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                finished_epoch,
                cfg,
                model_bucket,
            )

            if (
                hasattr(cfg.TRAIN, "CHECKPOINT_PERIOD")
                and cfg.TRAIN.CHECKPOINT_PERIOD > 0
                and (finished_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0
            ):
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    finished_epoch,
                    cfg,
                    model_bucket,
                )

            if (
                val_loader is not None
                and cfg.TRAIN.VAL_FRE_ITER > 0
                and (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0
            ):
                val_meter.set_model_ema_enabled(False)

                val_top1_acc = eval_epoch(
                    val_loader,
                    model,
                    val_meter,
                    finished_epoch,
                    cfg,
                )

                best_metric = cu.save_best_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    finished_epoch,
                    cfg,
                    cur_metric=val_top1_acc,
                    best_metric=best_metric,
                    model_bucket=model_bucket,
                )

                model.train()
                

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    max_iter = cfg.TRAIN.NUM_TEST_TASKS

    total_top1_correct = 0.0
    total_samples = 0.0

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= max_iter:
            break

        epoch_iter = cur_iter % cfg.SOLVER.STEPS_ITER

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

        top1_correct = num_topks_correct[0]
        num_samples = preds.size(0)

        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err, top1_correct = du.all_reduce(
                [loss, top1_err, top5_err, top1_correct]
            )
            num_samples = num_samples * misc.get_num_gpus(cfg)

        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )

        if torch.is_tensor(top1_correct):
            top1_correct = top1_correct.item()

        total_top1_correct += top1_correct
        total_samples += num_samples

        val_meter.iter_toc()

        val_meter.update_stats(
            top1_err,
            top5_err,
            val_loader.batch_size * max(misc.get_num_gpus(cfg), 1),
        )
        val_meter.update_predictions(preds, labels)

        top1_acc = 100.0 - top1_err

        if (epoch_iter + 1) % 50 == 0 or (epoch_iter + 1) == cfg.SOLVER.STEPS_ITER:
            logger.info(
                "[Eval] [Epoch {}/{}] [Iter {}/{}] loss={:.6f}, top1_acc={:.2f}".format(
                    cur_epoch + 1,
                    cfg.SOLVER.MAX_EPOCH,
                    epoch_iter + 1,
                    cfg.SOLVER.STEPS_ITER,
                    loss,
                    top1_acc,
                )
            )

        val_meter.iter_tic()

    val_top1_acc = 100.0 * total_top1_correct / max(total_samples, 1.0)

    logger.info(
        "[Eval] [Epoch {}/{}] final_top1_acc={:.4f}".format(
            cur_epoch + 1,
            cfg.SOLVER.MAX_EPOCH,
            val_top1_acc,
        )
    )

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    return val_top1_acc


def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH - start_epoch) % cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, start_epoch)
    
    # freeze some parameters
    for name, param in model.named_parameters():
        if 'class_embedding' not in name and 'temporal_embedding' not in name and 'Adapter' not in name and 'ln_post' not in name and 'classification_layer' not in name:
            param.requires_grad = False

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        for name, param in model.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    train_epoch(train_loader, model, optimizer, train_meter, start_epoch, cfg, val_meter, val_loader)