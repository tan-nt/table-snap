import torch
from torch.utils.data.distributed import DistributedSampler

from . import transform as T
from .dataset import LRCRecordLoader
from .utils import random_sample_ratio
from .batch_sampler import BucketSampler
from .dataset import Dataset, collate_func
from libs.utils.comm import distributed, get_rank, get_world_size

def create_train_dataloader(
    lrcs_path,
    num_workers,
    max_batch_size,
    max_pixel_nums,
    max_row_nums,
    max_col_nums,
    bucket_seps,
    max_img_size,
    height_norm,
    rota=False, epoch=0,
    scale_bucket=[]):
    loaders = list()
    img_size_dict ={}

    for lrc_path in lrcs_path:
        loader = LRCRecordLoader(lrc_path)
        loaders.append(loader)
        line_list = [line.strip('\n') for line in open(lrc_path+'_info.txt', 'r').readlines()]

        for i, line in enumerate(line_list):
            line = line.split('\t')
            img_name, h, w, height_ave = '\t'.join(line[:-3]), int(line[-3]), int(line[-2]), int(line[-1])
            h = int(h/height_ave*height_norm)
            w = int(w/height_ave*height_norm)

            img_size_dict[img_name] = random_sample_ratio((w, h), scale_bucket[epoch, i%1000])
    transforms =T.Compose([
        T.CallResizeImage(ratio_range=(0.8, 1.2), keep_ratio=True, bool_training=True, max_size=max_img_size, img_size_dict=img_size_dict),
        T.CallImageDistortion(brightness_delta=8, contrast_range=(0.75, 1.25), saturation_range=(0.9, 1.1), hue_delta=8),
        T.CallRowColStartBox_v0(),
        T.CallRowColLineMask(),
        T.CallRandomRotation_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],is_rota=rota),
        T.CallLayout()
    ])
    dataset = Dataset(loaders, transforms)
    batch_sampler = BucketSampler(dataset, img_size_dict, get_world_size(), get_rank(), \
        max_row_nums=max_row_nums, max_col_nums=max_col_nums, max_pixel_nums=max_pixel_nums, \
            max_batch_size=max_batch_size, seps=bucket_seps, epoch=epoch)
    print(f"train data path:{','.join(lrcs_path)}, train data num: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_func,
        batch_sampler=batch_sampler
    )
    return dataloader