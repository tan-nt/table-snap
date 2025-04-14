import torch
import math
from torch import nn
from torch.nn import functional as F
from .extractor import RoiPosFeatExtraxtor


class SALayer(nn.Module):
    def __init__(self, in_dim, att_dim, head_nums):
        super().__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.head_nums = head_nums

        assert self.att_dim % self.head_nums == 0, "att_dim must be divisible by head_nums"

        # Optional projection layer for adapting input to att_dim if needed
        self.input_proj = (
            nn.Conv1d(in_dim, att_dim, kernel_size=1)
            if in_dim != att_dim else nn.Identity()
        )

        self.key_layer = nn.Conv1d(att_dim, att_dim, kernel_size=1)
        self.query_layer = nn.Conv1d(att_dim, att_dim, kernel_size=1)
        self.value_layer = nn.Conv1d(att_dim, att_dim, kernel_size=1)

        self.output_proj = (
            nn.Conv1d(att_dim, in_dim, kernel_size=1)
            if att_dim != in_dim else nn.Identity()
        )

        self.scale = 1 / math.sqrt(att_dim // head_nums)

    def forward(self, feats, masks=None):
        bs, c, n = feats.shape
        if n == 0:
            return feats

        x = self.input_proj(feats)  # [B, att_dim, N]

        keys = self.key_layer(x).reshape(bs, self.head_nums, -1, n)     # [B, H, C', N]
        queries = self.query_layer(x).reshape(bs, self.head_nums, -1, n)
        values = self.value_layer(x).reshape(bs, self.head_nums, -1, n)

        logits = torch.einsum('bhcn,bhcm->bhnm', keys, queries) * self.scale  # [B, H, N, N]

        if masks is not None:
            if masks.shape[-1] != n:
                raise ValueError(f"[SALayer] Mask shape mismatch: {masks.shape[-1]} != {n}")
            logits = logits.masked_fill(masks[:, None, None, :] == 0, float('-inf'))

        weights = torch.softmax(logits, dim=-1)
        out = torch.einsum('bhcm,bhnm->bhcn', values, weights)  # [B, H, C', N]
        out = out.reshape(bs, -1, n)  # [B, att_dim, N]

        out = self.output_proj(out)  # project back to original dim if needed
        return out + feats  # Residual connection


def gen_cells_bbox(row_segments, col_segments, device):
    cells_bbox = list()
    for row_segments_pi, col_segments_pi in zip(row_segments, col_segments):
        num_rows = len(row_segments_pi) - 1
        num_cols = len(col_segments_pi) - 1
        cells_bbox_pi = list()
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                bbox = [
                    col_segments_pi[col_idx],
                    row_segments_pi[row_idx],
                    col_segments_pi[col_idx + 1],
                    row_segments_pi[row_idx + 1]
                ]
                cells_bbox_pi.append(bbox)
        cells_bbox_pi = torch.tensor(cells_bbox_pi, dtype=torch.float, device=device)
        cells_bbox.append(cells_bbox_pi)
    return cells_bbox


def align_cells_feat(cells_feat, num_rows, num_cols):
    batch_size = len(cells_feat)

    # Early sanity check
    if batch_size == 0 or len(cells_feat[0]) == 0:
        raise ValueError("[align_cells_feat] Empty input. Check upstream cell_spans or segment extraction.")

    dtype = cells_feat[0].dtype
    device = cells_feat[0].device

    max_row_nums = max(num_rows)
    max_col_nums = max(num_cols)

    aligned_cells_feat = []
    masks = torch.zeros([batch_size, max_row_nums, max_col_nums], dtype=dtype, device=device)

    for batch_idx in range(batch_size):
        num_rows_pi = num_rows[batch_idx]
        num_cols_pi = num_cols[batch_idx]
        cells_feat_pi = cells_feat[batch_idx]

        if num_rows_pi == 0 or num_cols_pi == 0 or cells_feat_pi.numel() == 0:
            print(f"[align_cells_feat] Skipping batch {batch_idx} due to empty cell features")
            dummy = torch.zeros([cells_feat_pi.shape[0], max_row_nums, max_col_nums], dtype=dtype, device=device)
            aligned_cells_feat.append(dummy)
            continue

        # (C, N) → (C, H, W)
        try:
            cells_feat_pi = cells_feat_pi.transpose(0, 1).reshape(-1, num_rows_pi, num_cols_pi)
        except Exception as e:
            print(f"[align_cells_feat] Reshape error for batch {batch_idx}: {e}")
            dummy = torch.zeros([cells_feat_pi.shape[0], max_row_nums, max_col_nums], dtype=dtype, device=device)
            aligned_cells_feat.append(dummy)
            continue

        padded_feat = F.pad(
            cells_feat_pi,
            (0, max_col_nums - num_cols_pi, 0, max_row_nums - num_rows_pi, 0, 0),
            mode='constant',
            value=0
        )
        aligned_cells_feat.append(padded_feat)
        masks[batch_idx, :num_rows_pi, :num_cols_pi] = 1

    aligned_cells_feat = torch.stack(aligned_cells_feat, dim=0)
    return aligned_cells_feat, masks


class CellsExtractor(nn.Module):
    def __init__(self, in_dim, cell_dim, heads, head_nums, pool_size, scale=1):
        super().__init__()
        self.in_dim = in_dim
        self.cell_dim = cell_dim
        self.pool_size = pool_size
        self.scale = scale

        # Flexible box feature extractor that matches variable in_dim
        self.box_feat_extractor = RoiPosFeatExtraxtor(
            self.scale,
            self.pool_size,
            self.in_dim,
            self.cell_dim
        )

        self.heads = heads
        self.row_sas = nn.ModuleList()
        self.col_sas = nn.ModuleList()

        for _ in range(self.heads):
            # SALayer now handles input projection internally if needed
            self.row_sas.append(SALayer(cell_dim, cell_dim, head_nums))
            self.col_sas.append(SALayer(cell_dim, cell_dim, head_nums))

    def forward(self, feats, row_segments, col_segments, img_sizes):
        device = feats.device
        num_rows = [len(row_segments_pi) - 1 for row_segments_pi in row_segments]
        num_cols = [len(col_segments_pi) - 1 for col_segments_pi in col_segments]

        cells_bbox = gen_cells_bbox(row_segments, col_segments, device)
        cells_feat = self.box_feat_extractor(feats, cells_bbox, img_sizes)

        if len(cells_feat) == 0 or cells_feat[0].numel() == 0:
            # If no valid cells, return dummy tensor
            dummy_feat = torch.zeros((feats.shape[0], self.cell_dim, 1, 1), device=feats.device)
            dummy_mask = torch.zeros((feats.shape[0], 1, 1), device=feats.device)
            return dummy_feat, dummy_mask

        aligned_cells_feat, masks = align_cells_feat(cells_feat, num_rows, num_cols)
        bs, c, nr, nc = aligned_cells_feat.shape

        for idx in range(self.heads):
            # Column-wise self-attention
            col_cells_feat = aligned_cells_feat.permute(0, 2, 1, 3).contiguous().reshape(bs * nr, c, nc)
            col_masks = masks.reshape(bs * nr, nc)
            col_cells_feat = self.col_sas[idx](col_cells_feat, col_masks)
            aligned_cells_feat = col_cells_feat.reshape(bs, nr, c, nc).permute(0, 2, 1, 3).contiguous()

            # Row-wise self-attention
            row_cells_feat = aligned_cells_feat.permute(0, 3, 1, 2).contiguous().reshape(bs * nc, c, nr)
            row_masks = masks.transpose(1, 2).reshape(bs * nc, nr)
            row_cells_feat = self.row_sas[idx](row_cells_feat, row_masks)
            aligned_cells_feat = row_cells_feat.reshape(bs, nc, c, nr).permute(0, 2, 3, 1).contiguous()

        return aligned_cells_feat, masks