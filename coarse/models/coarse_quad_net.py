import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm + ReLU6 block."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, stride=stride, groups=in_planes)
        self.pw = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pw(self.dw(x))))


class CoarseQuadNet(nn.Module):
    """Refined Coarse Quadrilateral Network for ID Card Detection.

    Stage 1 Structured Design (v6.2 - Performance Boost v2):
    1.  **High-Res Geometry Neck**: Fuses multi-scale features (1/4 to 1/32) 
        for ultra-precise geographic anchoring at 96x96 resolution.
    2.  **Dense Geometric Path**: Predicts Card Mask and Gaussian Boundary Contours.
    3.  **Structured Parameterization**: Regresses Center, Size, Rotation, and 4 residuals.
    4.  **Geometry-to-Corner Reconstruction**: Builds a rotated rectangle refined by residuals.
    """
    def __init__(self) -> None:
        super().__init__()

        # Backbone: Input 384x384 -> 12x12
        self.stem = ConvBNReLU(3, 16, stride=2)                       # 192
        
        # Split stage1 to get 96x96 features
        self.stage1_feat = DepthwiseSeparableConv(16, 32, stride=2)    # 96
        
        self.stage1_down = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),                  # 48
            DepthwiseSeparableConv(64, 64, stride=1),                  # 48
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),                 # 24
            DepthwiseSeparableConv(128, 128, stride=1),                # 24
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),                # 12
            DepthwiseSeparableConv(256, 256, stride=1),                # 12
            DepthwiseSeparableConv(256, 512, stride=1),                # 12
        )

        # FPN Neck for 96x96 Geometric Path
        self.lat_96 = nn.Conv2d(32, 64, kernel_size=1)
        self.lat_48 = nn.Conv2d(64, 64, kernel_size=1)
        self.lat_24 = nn.Conv2d(128, 64, kernel_size=1)
        self.lat_12 = nn.Conv2d(512, 64, kernel_size=1)
        
        # Dense Refinement Head (96x96)
        self.dense_refine = nn.Sequential(
            DepthwiseSeparableConv(64, 64, stride=1),                 # 96
            DepthwiseSeparableConv(64, 64, stride=1),                 # 96
        )
        self.geom_head = nn.Conv2d(64, 2, kernel_size=1)             # [Mask, Contour]

        # Structured Quadrilateral Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))               # [512, 1, 1]
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))                # [64, 4, 4]

        self.quad_mlp = nn.Sequential(
            nn.Linear(512 + 64*4*4, 256),
            nn.ReLU6(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, 14)
        )
        
        self.register_buffer('base_rect', torch.tensor([
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
        ], dtype=torch.float32).unsqueeze(0))

        self.score_head = nn.Linear(512, 1)

    def _reconstruct_corners(self, p: torch.Tensor) -> torch.Tensor:
        """Reconstructs 4 keypoints from structured rotated-rectangle parameterization."""
        B = p.size(0)

        cx, cy = torch.sigmoid(p[:, 0:1]), torch.sigmoid(p[:, 1:2])
        w, h = torch.sigmoid(p[:, 2:3]), torch.sigmoid(p[:, 3:4])
        
        sin_phi, cos_phi = p[:, 4:5], p[:, 5:6]
        norm = torch.sqrt(sin_phi**2 + cos_phi**2 + 1e-8)
        s, c = sin_phi / norm, cos_phi / norm
        
        base = self.base_rect.expand(B, -1, -1)
        bx, by = base[:, :, 0], base[:, :, 1]
        
        rx = (bx * w * c) - (by * h * s)
        ry = (bx * w * s) + (by * h * c)

        corners = torch.stack([rx + cx, ry + cy], dim=-1)
        
        res = p[:, 6:14].view(B, 4, 2)
        corners = corners + 0.1 * torch.tanh(res)
        
        return corners

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        
        # 1. Backbone
        f_stem = self.stem(x)
        f_96 = self.stage1_feat(f_stem)
        f_48 = self.stage1_down(f_96)
        f_24 = self.stage2(f_48)
        f_12 = self.stage3(f_24)

        # 2. Geometry Neck (FPN style)
        p12 = self.lat_12(f_12)
        p24 = self.lat_24(f_24) + F.interpolate(p12, size=(24, 24), mode='bilinear', align_corners=False)
        p48 = self.lat_48(f_48) + F.interpolate(p24, size=(48, 48), mode='bilinear', align_corners=False)
        # Upsample to 96x96 and fuse with lat_96
        p96 = self.lat_96(f_96) + F.interpolate(p48, size=(96, 96), mode='bilinear', align_corners=False)
        
        dense_feat = self.dense_refine(p96)
        geom_out = self.geom_head(dense_feat)
        mask = torch.sigmoid(geom_out[:, 0:1])
        edges = torch.sigmoid(geom_out[:, 1:2])

        # 3. Structured Quadrilateral Path
        f_global = self.global_pool(f_12).view(B, -1)
        f_local = self.local_pool(dense_feat).view(B, -1)
        q_feat = torch.cat([f_global, f_local], dim=1)
        
        params = self.quad_mlp(q_feat)
        corners = self._reconstruct_corners(params)
        res = params[:, 6:14].view(B, 4, 2)

        # 4. Confidence Score
        score = torch.sigmoid(self.score_head(f_global))

        return {
            'score': score,
            'corners': corners,
            'mask': mask,
            'edges': edges,
            'residuals': res
        }
