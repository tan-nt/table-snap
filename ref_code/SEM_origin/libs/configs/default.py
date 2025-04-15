import os
import torch
from libs.utils.vocab import Vocab

device = torch.device('cuda')

train_lrcs_path = [
    "../data/SciTSR_lrc/train/table.lrc",
]
train_data_dir = ''
train_max_pixel_nums = 400 * 400 * 5
train_bucket_seps = (50, 50, 50)

valid_lrc_path = '../data/SciTSR_lrc/test/table.lrc'
valid_data_dir = ''

# train_max_batch_size = 16       # ↓ from 48 → safer with large models like EfficientNet
cpu_train_max_batch_size = 1
gpu_mps_train_max_batch_size = 2
gpu_train_max_batch_size = 16


cpu_train_num_workers = 0          # Keep as is if CPU load is low
gpu_train_num_workers = 16
gpu_mps_train_num_workers = 0


cpu_valid_batch_size = 1
gpu_valid_batch_size = 1
gpu_mps_valid_batch_size = 1


gpu_valid_num_workers = 8
cpu_valid_num_workers = 0           # Reduce slightly to match smaller batches
gpu_mps_valid_num_workers = 0


vocab = Vocab()

# model params
# backbone
arch = "res34"
pretrained_backbone = True
backbone_out_channels = (64, 128, 256, 512)

# fpn
fpn_out_channels = 256

# pan
pan_num_levels = 4
pan_in_dim = 256
pan_out_dim = 256

# row segment predictor
rs_scale = 1

# col segment predictor
cs_scale = 1

# divide predictor
dp_head_nums = 8
dp_scale = 1

# cells extractor params
ce_scale = 1 / 8
ce_pool_size = (3, 3)
ce_dim = 512
ce_head_nums = 8
ce_heads = 1

# decoder
embed_dim = 512
feat_dim = 512
lm_state_dim = 512
proj_dim = 512
cover_kernel = 7
att_threshold = 0.5
spatial_att_weight_loss_wight = 1.0

# train params
base_lr = 0.0001
min_lr = 1e-6
weight_decay = 0

num_epochs = 5

cpu_max_iters_per_epoch = 50
gpu_mps_max_iters_per_epoch = 20
gpu_max_iters_per_epoch = 1000000


cpu_max_valid_iters = 10
gpu_mps_max_valid_iters = 5
gpu_max_valid_iters = 30000 # run all

sync_rate = 20

cpu_log_sep = 10
gpu_mps_log_sep = 10
gpu_log_sep = 100


work_dir = './experiments/heads_1'

train_checkpoint = None

eval_checkpoint = os.path.join(work_dir, 'best_f1_model.pth')

