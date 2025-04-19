import torch
from torch import nn
from torchvision.ops import roi_align


def convert_to_roi_format(lines_box):
    concat_boxes = torch.cat(lines_box, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full((lines_box_pi.shape[0], 1), i, dtype=dtype, device=device)
            for i, lines_box_pi in enumerate(lines_box)
        ],
        dim=0
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class RoiFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim
        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, feats, lines_box):
        rois = convert_to_roi_format(lines_box)
        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )

        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.fc(lines_feat)
        lines_feat = torch.split(lines_feat, [item.shape[0] for item in lines_box])
        return list(lines_feat)


class RoiPosFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim
        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        self.bbox_ln = nn.LayerNorm(self.output_dim)
        self.bbox_tranform = nn.Linear(4, self.output_dim)

        self.add_ln = nn.LayerNorm(self.output_dim)

    def forward(self, feats, lines_box, img_sizes):
        # lines_box: List[Tensor[N_i, 4]] per batch
        rois = []
        for batch_idx, boxes in enumerate(lines_box):
            if boxes.numel() == 0:
                continue
            rois.append(
                torch.cat([
                    torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device),
                    boxes
                ], dim=1)
            )
        if len(rois) == 0:
            return []

        rois = torch.cat(rois, dim=0)  # [K, 5]

        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2,
            aligned=True
        )
        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.fc(lines_feat)

        lines_feat = list(torch.split(lines_feat, [item.shape[0] for item in lines_box]))

        # Add positional embedding
        feats_H, feats_W = feats.shape[-2:]
        for idx, (line_box, img_size) in enumerate(zip(lines_box, img_sizes)):
            if line_box.numel() == 0:
                continue
            norm_box = line_box.clone()
            norm_box[:, 0] = norm_box[:, 0] * self.scale / feats_W
            norm_box[:, 1] = norm_box[:, 1] * self.scale / feats_H
            norm_box[:, 2] = norm_box[:, 2] * self.scale / feats_W
            norm_box[:, 3] = norm_box[:, 3] * self.scale / feats_H
            lines_feat[idx] = self.add_ln(lines_feat[idx] + self.bbox_ln(self.bbox_tranform(norm_box)))

        return lines_feat
