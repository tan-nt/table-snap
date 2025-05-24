import numpy as np
from libs.configs import cfg
from libs.data import create_train_dataloader
from libs.utils import logger
from libs.configs import cfg, setup_config
import os
import shutil
import torch
from libs.utils.comm import  synchronize
import argparse

def get_exist_path(path):
    os.makedirs(path,exist_ok=True)
    return path


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='semv3')
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--version_name",type=str, default="v1")
    args = parser.parse_args()

    setup_config(args.cfg)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir+"_"+cfg.tips

    # dst_path = os.path.join( cfg.work_dir, args.cfg+".py" )
    # if not os.path.isfile(dst_path):
    #     src_path = os.path.join(f"./libs/configs",args.cfg+".py")
    #     shutil.copyfile(src_path, dst_path)

    os.environ['LOCAL_RANK'] = str(args.local_rank)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'train.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Args: %s' % args)

def main():
    init()

    np.random.seed(0)
    scale_bucket =  0.8 + (1.2-0.8)*np.random.random((cfg.num_epochs,1000))
    print('cfg.train_lrcs_path=', cfg.train_lrcs_path)
    print('cfg.train_num_workers=', cfg.train_num_workers)
    print('cfg.train_max_batch_size=', cfg.train_max_batch_size)


    train_dataloader = create_train_dataloader(
        cfg.train_lrcs_path,
        cfg.train_num_workers,
        cfg.train_max_batch_size,
        cfg.train_max_pixel_nums,
        cfg.train_max_row_nums,
        cfg.train_max_col_nums,
        cfg.train_bucket_seps,
        cfg.max_img_size,
        cfg.height_norm,
        cfg.train_rota,
        0,
        scale_bucket
    )

    logger.info(
        'Train dataset have %d samples, %d batchs' % \
            (
                len(train_dataloader.dataset),
                len(train_dataloader.batch_sampler)
            )
    )

    # valid_dataloader = create_valid_dataloader(
    #     cfg.valid_lrc_path,
    #     cfg.valid_num_workers,
    #     cfg.valid_batch_size,
    #     cfg.max_img_size,
    #     cfg.height_norm
    # )

    # logger.info(
    #     'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
    #         (
    #             len(valid_dataloader.dataset),
    #             len(valid_dataloader.batch_sampler),
    #             valid_dataloader.batch_size
    #         )
    # )

    # model = build_model(cfg)
    # model.to(cfg.device)

    # if distributed():
    #     print(os.environ["WORLD_SIZE"])
    #     synchronizer = ModelSynchronizer(model, cfg.sync_rate)
    # else:
    #     synchronizer = None

    # epoch_iters = len(train_dataloader.batch_sampler)
    # optimizer = build_optimizer(cfg, model)

    # global metrics_name
    # global best_metrics
    # start_epoch = 0
    # if cfg.resume_path ==None:

    #     resume_path =  os.path.join(cfg.work_dir, 'latest_model.pth')
    # else:
    #     resume_path =  cfg.resume_path
    # if os.path.exists(resume_path):
    #     best_metrics, start_epoch = load_checkpoint(resume_path, model, optimizer)
    #     if len(best_metrics)==1:
    #         best_metrics.append(0)
    #     start_epoch += 1
    #     logger.info('resume from: %s' % resume_path)
    # elif cfg.train_checkpoint is not None:
    #     load_checkpoint(cfg.train_checkpoint, model)
    #     logger.info('load checkpoint from: %s' % cfg.train_checkpoint)

    # scheduler = build_scheduler(cfg, optimizer, epoch_iters, start_epoch)

    # time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_iters)
    # time_counter.reset()

    # for epoch in range(start_epoch, cfg.num_epochs):
    #     np.random.seed(epoch)
    #     scale_bucket =  0.8 + (1.2-0.8)*np.random.random((cfg.num_epochs,1000))

    #     train_dataloader = create_train_dataloader(
    #         cfg.train_lrcs_path,
    #         cfg.train_num_workers,
    #         cfg.train_max_batch_size,
    #         cfg.train_max_pixel_nums,
    #         cfg.train_max_row_nums,
    #         cfg.train_max_col_nums,
    #         cfg.train_bucket_seps,
    #         cfg.max_img_size,
    #         cfg.height_norm,
    #         cfg.valid_rota,
    #         epoch,
    #         scale_bucket
    #     )

    #     if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
    #         train_dataloader.batch_sampler.set_epoch(epoch)

    #     logger.info(
    #         'Train dataset have %d samples, %d batchs' % \
    #             (
    #                 len(train_dataloader.dataset),
    #                 len(train_dataloader.batch_sampler)
    #             )
    #     )

    #     train(cfg, epoch, train_dataloader, model, optimizer, scheduler, time_counter, synchronizer, epoch)

    #     if epoch >= cfg.start_eval and epoch % cfg.eval_epochs == 0:
    #         with torch.no_grad():
    #             # metrics = valid(cfg, valid_dataloader, model)
    #             metrics = valid(cfg, valid_dataloader, model)

    #         for metric_idx in range(len(metrics_name)):
    #             if metrics[metric_idx] > best_metrics[metric_idx]:
    #                 best_metrics[metric_idx] = metrics[metric_idx]
    #                 save_checkpoint(os.path.join(cfg.work_dir, 'best_%s_model.pth' % metrics_name[metric_idx]), model, optimizer, best_metrics, epoch)
    #                 logger.info('Save current model as best_%s_model' % metrics_name[metric_idx])

    #     save_checkpoint(os.path.join(cfg.work_dir, 'latest_model.pth'), model, optimizer, best_metrics, epoch)


if __name__ == '__main__':
    main()