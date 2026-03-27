import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBNReLU(nn.Module):
    """Standard Conv2d + BatchNorm + ReLU6 block."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1,
                 groups: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet-style)."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, stride=stride, groups=in_planes)
        self.pw = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class OrientNet(nn.Module):
    """Lightweight card orientation classifier.

    Input:  [B, 3, 128, 128]  — warped card crop (output of Coarse homography).
    Output: [B, 4]            — logits for rotation classes 0°, 90°, 180°, 270°.

    Architecture (~50k parameters):
        128 → 64 → 32 → 16 → 8 → Global Average Pool → 4-class head.
    This is enough capacity for text-direction / face-side discrimination
    on a normalised, already-cropped card image.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            ConvBNReLU(3, 16, stride=2),            # 64
            DepthwiseSeparableConv(16, 32, stride=2),   # 32
            DepthwiseSeparableConv(32, 64, stride=2),   # 16
            DepthwiseSeparableConv(64, 96, stride=2),   # 8
            DepthwiseSeparableConv(96, 128, stride=2),  # 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(128, num_classes)

        # Initialise classifier to small weights for stable early training
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalised card crop (ImageNet mean/std).

        Returns:
            logits: [B, num_classes]  (raw, before softmax)
        """
        feats = self.backbone(x)          # [B, 128, 4, 4]
        feats = self.pool(feats)          # [B, 128, 1, 1]
        feats = feats.flatten(1)          # [B, 128]
        feats = self.dropout(feats)
        return self.classifier(feats)     # [B, 4]

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted rotation class (0–3) without gradient."""
        with torch.no_grad():
            logits = self.forward(x)
        return logits.argmax(dim=-1)   # [B]
