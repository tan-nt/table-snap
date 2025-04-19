import torch
from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: list of input channels from c2 to c5 (e.g. [24, 40, 112, 320] for EfficientNet-B0)
        out_channels: number of channels for all FPN outputs
        """
        super().__init__()
        assert len(in_channels) == 4, "Expected 4 input feature maps (c2 to c5)"

        # Lateral layers to unify channel dimensions
        self.lat_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Output conv layers for each FPN level
        self.out_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
            for _ in in_channels
        ])

    def forward(self, feats):
        # Take last 4 feature maps (c2 to c5)
        c2, c3, c4, c5 = feats[-4:]

        # Top-down fusion
        p5 = self.lat_layers[3](c5)
        p4 = self.lat_layers[2](c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lat_layers[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat_layers[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)

        # Final output layers
        p2 = self.out_layers[0](p2)
        p3 = self.out_layers[1](p3)
        p4 = self.out_layers[2](p4)
        p5 = self.out_layers[3](p5)

        return p2, p3, p4, p5


def build_fpn(in_channels, out_channels):
    return FPN(in_channels, out_channels)