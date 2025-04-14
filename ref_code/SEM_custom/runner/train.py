import shutil
import torch
import tqdm
import json
import os
import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.comm import distributed, synchronize
from libs.utils.checkpoint import load_checkpoint, save_checkpoint
from libs.data import create_train_dataloader, create_valid_dataloader
from libs.utils.model_synchronizer import ModelSynchronizer
from libs.utils.time_counter import TimeCounter
from libs.utils.utils import is_simple_table
from libs.utils.utils import cal_mean_lr
from libs.utils.counter import Counter
from libs.utils import logger
from libs.model import build_model
from libs.configs import cfg, setup_config
import os
import csv
import time
import torch


metrics_name = ['f1']
best_metrics = [0.0]


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='debug')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    setup_config(args.cfg)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    num_gpus = int(os.environ['MORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()
    logger.setup_logger('Line Detect Model', cfg.work_dir, 'train.log')
    logger.info('Use config:%s' % args.cfg)



def train(cfg, epoch, num_epochs, dataloader, model, optimizer, scheduler, time_counter, synchronizer=None):
    model.train()
    counter = Counter(cache_nums=1000)

    # === Log file setup ===
    os.makedirs("logs", exist_ok=True)
    iter_log_file = os.path.join("logs", "train_metrics.csv")
    epoch_log_file = os.path.join("logs", "epoch_times.csv")

    # === Reset and write header for train_metrics.csv at epoch 0 ===
    if epoch == 0:
        with open(iter_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "iter", "memory_MB", "mean_lr", "time_left", "time_elapsed_iter_s",
                "row_segments_loss", "row_precision", "row_recall",
                "col_segments_loss", "col_precision", "col_recall",
                "divide_loss", "divide_acc",
                "cls_loss", "spatial_att_logit_loss",
                "cls_acc", "cls_none_acc", "cls_bold_acc", "cls_space_acc", "cls_blank_acc",
                "spatial_att_logit_acc"
            ])

    # === Write header for epoch_times.csv once ===
    if not os.path.exists(epoch_log_file):
        with open(epoch_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_elapsed_s", "final_lr", "max_memory_MB"])

    # === Track time for the entire epoch ===
    epoch_start = time.time()

    cfg.device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    for it, data_batch in enumerate(dataloader):
        iter_start = time.time()

        # === Load data ===
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images'].to(cfg.device)
        cls_labels = data_batch['cls_labels'].to(cfg.device)
        labels_mask = data_batch['labels_mask'].to(cfg.device)
        rows_fg_spans = data_batch['rows_fg_spans']
        rows_bg_spans = data_batch['rows_bg_spans']
        cols_fg_spans = data_batch['cols_fg_spans']
        cols_bg_spans = data_batch['cols_bg_spans']
        cells_spans = data_batch['cells_spans']
        divide_labels = data_batch['divide_labels'].to(cfg.device)
        layouts = data_batch['layouts'].to(cfg.device)

        try:
            # === Forward & backward ===
            optimizer.zero_grad()
            pred_result, result_info = model(
                images, images_size,
                cls_labels, labels_mask, layouts,
                rows_fg_spans, rows_bg_spans,
                cols_fg_spans, cols_bg_spans,
                cells_spans, divide_labels,
            )

            # print(">> pred_result type:", type(pred_result))
            # print(">> pred_result length:", len(pred_result))

            loss = sum([val for key, val in result_info.items() if 'loss' in key])
            loss.backward()
            optimizer.step()
            scheduler.step()
            counter.update(result_info)
        except Exception as e:
            raise e
            logger.error(f'[Train] Exception during training: {e}')

        iter_elapsed = time.time() - iter_start

        # === Iteration Logging ===
        if it % cfg.log_sep == 0:
            # Handle memory logging safely
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                mem = 0  # fallback for CPU/MPS

            lr = cal_mean_lr(optimizer)
            left_time = time_counter.step(epoch, it + 1)
            formatted = counter.format_mean(sync=False)

            logger.info(
                '[Train][Epoch %03d/%03d Iter %04d/%04d][Memory: %.0fMB ][Mean LR: %.6f ][Left: %s][Time Elapsed: %.2fs] %s' %
                (epoch, num_epochs, it, len(dataloader), mem, lr, left_time, iter_elapsed, formatted)
            )

            # Save iteration metrics
            with open(iter_log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, it, round(mem, 2), lr, str(left_time), round(iter_elapsed, 4)
                ] + [round(sum(counter.metric_dict[key]) / len(counter.metric_dict[key]), 6) if len(counter.metric_dict[key]) > 0 else 0.0
                    for key in counter.metric_dict])

        # === Distributed sync ===
        if synchronizer is not None:
            synchronizer()
        if synchronizer is not None:
            synchronizer(final_align=True)

    # === Epoch Logging ===
    epoch_elapsed = time.time() - epoch_start
    logger.info(f"[Train][Epoch %03d Completed][Elapsed Time: %.2fs]" % (epoch, epoch_elapsed))

    with open(epoch_log_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            round(epoch_elapsed, 2),
            cal_mean_lr(optimizer),
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        ])

def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()

    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images'].to(cfg.device)
        tables = data_batch['tables']
        pred_result, _ = model(images, images_size)
        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                                 (pred_result[0][batch_idx], pred_result[1][batch_idx],
                                  pred_result[2][batch_idx], pred_result[3][batch_idx])
            )
            for batch_idx in range(len(ids))
        ]
        pred_relations = [table_to_relations(table) for table in pred_tables]
        total_pred_relations.extend(pred_relations)
        # label
        label_relations = []
        for table in tables:
            label_path = os.path.join(cfg.valid_data_dir, table['label_path'])
            with open(table['label_path'], 'r') as f:
                label_relations.append(json.load(f))
        total_label_relations.extend(label_relations)

    # cal P, R, F1
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    P, R, F1 = np.array(total_relations_metric).mean(0).tolist()
    F1 = 2 * P * R / (P + R)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))
    return (F1,)


def build_optimizer(cfg, model):
    params = list()
    for _, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
    optimizer = torch.optim.Adam(params, cfg.base_lr)
    return optimizer


def build_scheduler(cfg, optimizer, epoch_iters, start_epoch=0):
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.num_epochs * epoch_iters,
        eta_min=cfg.min_lr,
        last_epoch=-1 if start_epoch == 0 else start_epoch * epoch_iters
    )
    return scheduler


def main():
    init()

    train_dataloader = create_train_dataloader(
        cfg.vocab,
        cfg.train_lrcs_path,
        cfg.train_num_workers,
        cfg.train_max_batch_size,
        cfg.train_max_pixel_nums,
        cfg.train_bucket_seps,
        cfg.train_data_dir
    )

    logger.info(
        'Train dataset have %d samples, %d batchs' %
        (
            len(train_dataloader.dataset),
            len(train_dataloader.batch_sampler)
        )
    )

    valid_dataloader = create_valid_dataloader(
        cfg.vocab,
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size,
        cfg.valid_data_dir
    )

    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' %
        (
            len(valid_dataloader.dataset),
            len(valid_dataloader.batch_sampler),
            valid_dataloader.batch_size
        )
    )

    model = build_model(cfg)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    # device = torch.device("cpu")
    # model.to(device)

    # model.cuda()

    if distributed():
        synchronizer = ModelSynchronizer(model, cfg.sync_rate)
    else:
        synchronizer = None

    epoch_iters = len(train_dataloader.batch_sampler)
    optimizer = build_optimizer(cfg, model)

    global metrics_name
    global best_metrics
    start_epoch = 0

    resume_path = os.path.join(cfg.work_dir, 'latest_model.pth')
    if os.path.exists(resume_path):
        best_metrics, start_epoch = load_checkpoint(resume_path, model, optimizer)
        start_epoch += 1
        logger.info('resume from: %s' % resume_path)
    elif cfg.train_checkpoint is not None:
        load_checkpoint(cfg.train_checkpoint, model)
        logger.info('load checkpoint from: %s' % cfg.train_checkpoint)

    scheduler = build_scheduler(cfg, optimizer, epoch_iters, start_epoch)

    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_iters)
    time_counter.reset()

    for epoch in range(start_epoch, cfg.num_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        train(cfg, epoch, cfg.num_epochs, train_dataloader, model, optimizer, scheduler, time_counter, synchronizer)

        with torch.no_grad():
            metrics = valid(cfg, valid_dataloader, model)

        for metric_idx in range(len(metrics_name)):
            if metrics[metric_idx] > best_metrics[metric_idx]:
                best_metrics[metric_idx] = metrics[metric_idx]
                save_checkpoint(os.path.join(cfg.work_dir, 'best_%s_model.pth' % metrics_name[metric_idx]), model, optimizer, best_metrics, epoch)
                logger.info('Save current model as best_%s_model' % metrics_name[metric_idx])

        save_checkpoint(os.path.join(cfg.work_dir, 'latest_model.pth'), model, optimizer, best_metrics, epoch)


if __name__ == '__main__':
    main()
