import torch
import torch.nn as nn
import torch.nn.functional as F
from common.metrics import SoftArgmax2D


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=groups, bias=False),
            nn.GroupNorm(8, out_planes),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x_up = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x_up, skip], dim=1))


class FullCardRefinerNet(nn.Module):
    """
    High-Precision Stage 2: Full-Card Corner Refiner.
    Uses a Global Heatmap Architecture:
    1. Feature Pyramid matching Stride 32 up to Stride 4.
    2. Prediction of 4 full-image heatmaps.
    3. SoftArgmax decoding to normalized coordinates.
    Input Size: 640x640
    """
    def __init__(self, input_size=640):
        super().__init__()
        self.input_size = input_size
        
        # 1. Backbone: Stride 4 -> 8 -> 16 -> 32
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),   # 320
            ConvBNReLU(32, 64, stride=2),  # 160 (Stride 4)
        )
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 64, stride=1),
            ConvBNReLU(64, 128, stride=2), # 80 (Stride 8)
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 128, stride=1),
            ConvBNReLU(128, 256, stride=2), # 40 (Stride 16)
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 256, stride=1),
            ConvBNReLU(256, 512, stride=2), # 20 (Stride 32)
        )
        
        # 2. Decoder: Up to Stride 4
        self.dec3 = DecoderBlock(512, 256, 256) # 20 -> 40
        self.dec2 = DecoderBlock(256, 128, 128) # 40 -> 80
        self.dec1 = DecoderBlock(128, 64, 64)   # 80 -> 160
        
        # 3. Global Heatmap Head [64 -> 4]
        self.heatmap_head = nn.Sequential(
            ConvBNReLU(64, 64, stride=1),
            ConvBNReLU(64, 64, stride=1),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)
        )
        
        # We use a smaller beta for global soft-argmax to ensure gradients remain stable
        # over the larger 160x160 area. SoftArgmax beta=50.0 provides sharp peaks.
        self.soft_argmax = SoftArgmax2D(beta=50.0)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Critical CenterNet focal loss initialization
        # Initialize the final Conv2d bias so it predicts background (pi=0.1) initially
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 640, 640]
        Returns:
            final_coords: [B, 4, 2] in [0, 1] space
            heatmaps: [B, 4, 160, 160] raw logits
        """
        s = self.stem(x)         # 160x160 (Stride 4)
        f1 = self.stage1(s)      # 80x80 (Stride 8)
        f2 = self.stage2(f1)     # 40x40 (Stride 16)
        f3 = self.stage3(f2)     # 20x20 (Stride 32)
        
        # Decoder Path
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, s)
        
        # Global Heatmaps [B, 4, 160, 160]
        heatmaps = self.heatmap_head(d1)
        
        # Decode global coords [B, 4, 2] in [0, 1] range
        final_coords = self.soft_argmax(heatmaps)
        
        return final_coords, heatmaps

# Compatibility alias
FullCardCornerNet = FullCardRefinerNet
PatchRefinerNet = FullCardRefinerNet
