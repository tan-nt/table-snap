import os
import torch
from libs.utils.vocab import Vocab

device = torch.device('cuda')

train_lrcs_path = [
    "./dataset/SciTSR_lrc/train/table.lrc"
]
train_data_dir = ''
train_max_pixel_nums = 400 * 400 * 5
train_bucket_seps = (50, 50, 50)

valid_lrc_path = './dataset/SciTSR_lrc/test/table.lrc'
valid_data_dir = ''

train_max_batch_size = 16       # ↓ from 48 → safer with large models like EfficientNet
train_num_workers = 0         # Keep as is if CPU load is low

valid_batch_size = 4            # ↓ from 16 → prevents memory/index overflows
valid_num_workers = 0           # Reduce slightly to match smaller batches

vocab = Vocab()

# model params
# backbone
arch = 'efficientnet_b0'

pretrained_backbone = True
# backbone_out_channels = (64, 128, 256, 512) -- for resnet
# backbone_out_channels = [32, 56, 160, 448]  # <-- For EfficientNet-B4
backbone_out_channels = [24, 40, 112, 320]  # update this in your config
  # <-- For EfficientNet-B0

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
# ce_scale = 1 / 8
ce_scale = 1 / 32
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

num_epochs = 1
sync_rate = 20

log_sep = 100

work_dir = './experiments/heads_1'

train_checkpoint = None

eval_checkpoint = os.path.join(work_dir, 'best_f1_model.pth')
