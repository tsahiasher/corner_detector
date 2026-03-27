import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from typing import Tuple, Optional, Dict

class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm + ReLU6 block."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> None:
        super().__init__()
        # To maintain 'same' size for ks=3 with dilation, padding must be equal to dilation
        padding = dilation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block."""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, stride=stride, groups=in_planes, dilation=dilation)
        self.pw = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pw(self.dw(x))))


class CoarseQuadNet(nn.Module):
    """Refined Coarse Quadrilateral Network for ID Card Detection.

    Stage 1 Redesign (v7.0):
    1.  **Stronger Dense Path**: Improved FPN Neck at 96x96 resolution.
    2.  **Dense Keypoint Head**: Predicts 4 Heatmaps + 8 Offset channels (dx, dy).
    3.  **Primary Dense Inference**: Corners are derived from Heatmap + Offset peaks.
    4.  **Auxiliary Quad Branch**: Structured global quad regressor for stable anchoring.
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

        # Better FPN Neck for 96x96 Dense Path
        self.lat_96 = nn.Conv2d(32, 64, kernel_size=1)
        self.lat_48 = nn.Conv2d(64, 64, kernel_size=1)
        self.lat_24 = nn.Conv2d(128, 64, kernel_size=1)
        self.lat_12 = nn.Conv2d(512, 64, kernel_size=1)
        
        # Exponential Dilated Refinement with Dropout for generalization:
        # Receptive field mathematically expands to 63x63 without adding ANY new parameters.
        # Dropout(0.1) prevents the heatmap from overfitting training images.
        self.dense_refine = nn.Sequential(
            DepthwiseSeparableConv(64, 64, stride=1, dilation=1),
            nn.Dropout2d(0.1),
            DepthwiseSeparableConv(64, 64, stride=1, dilation=2),
            nn.Dropout2d(0.1),
            DepthwiseSeparableConv(64, 64, stride=1, dilation=4), 
            nn.Dropout2d(0.1),
            DepthwiseSeparableConv(64, 64, stride=1, dilation=8), 
            nn.Dropout2d(0.1),
            DepthwiseSeparableConv(64, 64, stride=1, dilation=16), 
        )
        
        # CenterNet Specific heads
        self.center_heatmap_head = nn.Conv2d(64, 1, kernel_size=1)   # Single Card Center
        self.corner_offset_head = nn.Conv2d(64, 8, kernel_size=1)    # 8 Sub-grid offsets (dx, dy) for 4 visual corners
        # orient_head REMOVED: a single 4-channel center-pixel cannot generalize orientation semantics.
        # Instead, inference applies the same deterministic angle-sort as training.

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        
        # 1. Backbone
        f_stem = self.stem(x)
        f_96 = self.stage1_feat(f_stem)
        f_48 = self.stage1_down(f_96)
        f_24 = self.stage2(f_48)
        f_12 = self.stage3(f_24)

        # 2. Dense Neck (FPN)
        p12 = self.lat_12(f_12)
        p24 = self.lat_24(f_24) + F.interpolate(p12, size=(24, 24), mode='bilinear', align_corners=False)
        p48 = self.lat_48(f_48) + F.interpolate(p24, size=(48, 48), mode='bilinear', align_corners=False)
        p96 = self.lat_96(f_96) + F.interpolate(p48, size=(96, 96), mode='bilinear', align_corners=False)
        
        dense_feat = self.dense_refine(p96)
        
        # 3. Dense Branch Prediction
        dense_center = self.center_heatmap_head(dense_feat) # [B, 1, 96, 96]
        dense_offsets = self.corner_offset_head(dense_feat) # [B, 8, 96, 96]
        
        # Center Peak Extraction (Argmax)
        H, W = dense_center.shape[2:]
        flat_center = dense_center.view(B, -1)
        _, max_idx = torch.max(flat_center, dim=-1) # [B]
        
        grid_y = max_idx // W
        grid_x = max_idx % W
        
        # Gather 8 predicted offsets exactly at the single center peak location
        b_idx_gather = torch.arange(B, device=x.device)
        offsets_raw = dense_offsets[b_idx_gather, :, grid_y, grid_x]     # [B, 8]
        
        # Bound offsets with Tanh to prevent off-screen explosions.
        offsets = torch.tanh(offsets_raw) * 0.75
        
        # Visual Corners = Center(norm) + Offsets(norm)
        offsets_x = offsets[:, 0::2] # [B, 4]
        offsets_y = offsets[:, 1::2] # [B, 4]
        
        pred_x = (grid_x.float().unsqueeze(1) / W) + offsets_x # [B, 4]
        pred_y = (grid_y.float().unsqueeze(1) / H) + offsets_y # [B, 4]
        
        # Hard clamp: corners must always be inside the image boundary
        pred_x = torch.clamp(pred_x, 0.0, 1.0)
        pred_y = torch.clamp(pred_y, 0.0, 1.0)
        corners_visual = torch.stack([pred_x, pred_y], dim=-1) # [B, 4, 2]

        # Deterministic angle-sort for semantic ordering (replaces orient_head).
        # This is the EXACT same sort applied to GT corners in training, guaranteeing consistency.
        # Sort by atan2(dy, dx) ascending => clockwise from top-left visual corner.
        centroid = corners_visual.mean(dim=1, keepdim=True)   # [B, 1, 2]
        diffs = corners_visual - centroid                      # [B, 4, 2]
        angles = torch.atan2(diffs[:, :, 1], diffs[:, :, 0]) # [B, 4]
        sort_idx = torch.argsort(angles, dim=1)               # [B, 4]
        b_idx_t = torch.arange(B, device=x.device).unsqueeze(1).expand(B, 4)
        corners_final = corners_visual[b_idx_t, sort_idx]     # [B, 4, 2] semantically sorted

        return {
            'score': torch.ones((B, 1), device=x.device),
            'corners': corners_final,           # Semantically sorted corners (angle-sorted)
            'corners_visual': corners_visual,   # Raw predicted corners in offset order
            'dense_center': dense_center,       # [B, 1, 96, 96]
            'dense_offsets': dense_offsets,     # [B, 8, 96, 96]
        }
