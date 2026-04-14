import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import torchvision.models as models
from common.geometry import sort_corners_clockwise

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

class AddCoords(nn.Module):
    """CoordConv equivalent: Appends 2 normalized spatial coordinate channels (-1 to 1) to the input tensor."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        yy_channel = torch.linspace(-1, 1, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx_channel = torch.linspace(-1, 1, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, yy_channel, xx_channel], dim=1)

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

    Stage 1 Redesign (v8.0 ResNet18):
    1.  **Stronger Dense Path**: ResNet18 Backbone with FPN Neck targeting 1/4 resolution.
    2.  **Dense Keypoint Head**: Predicts 1 Heatmap + 8 Offset channels (dx, dy).
    """
    def __init__(self) -> None:
        super().__init__()

        # Backbone: ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # STEM: Input -> 1/4 resolution
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # 1/4, 64 ch
        self.layer2 = resnet.layer2 # 1/8, 128 ch
        self.layer3 = resnet.layer3 # 1/16, 256 ch
        self.layer4 = resnet.layer4 # 1/32, 512 ch

        # Better FPN Neck for Dense Path targeting 1/4 resolution
        self.lat_32 = nn.Conv2d(512, 128, kernel_size=1)
        self.lat_16 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat_8  = nn.Conv2d(128, 128, kernel_size=1)
        self.lat_4  = nn.Conv2d(64, 128, kernel_size=1)
        
        self.smooth_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.add_coords = AddCoords()

        # Exponential Dilated Refinement (Input is now 128 + 2 coordinate channels)
        self.dense_refine = nn.Sequential(
            DepthwiseSeparableConv(130, 128, stride=1, dilation=1),
            DepthwiseSeparableConv(128, 128, stride=1, dilation=2),
            DepthwiseSeparableConv(128, 128, stride=1, dilation=4), 
            DepthwiseSeparableConv(128, 128, stride=1, dilation=8), 
            DepthwiseSeparableConv(128, 128, stride=1, dilation=16), 
        )
        
        # CenterNet Specific heads
        self.center_heatmap_head = nn.Conv2d(128, 1, kernel_size=1)   # Single Card Center
        self.corner_geom_head = nn.Conv2d(128, 10, kernel_size=1)     # 2 for w/h, 8 for sub-quad offsets

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        
        # 1. Backbone
        f_stem = self.stem(x)
        f_4 = self.layer1(f_stem)
        f_8 = self.layer2(f_4)
        f_16 = self.layer3(f_8)
        f_32 = self.layer4(f_16)

        # 2. Dense Neck (FPN)
        p32 = self.lat_32(f_32)
        p16 = self.lat_16(f_16) + F.interpolate(p32, size=f_16.shape[2:], mode='bilinear', align_corners=False)
        p8  = self.lat_8(f_8)   + F.interpolate(p16, size=f_8.shape[2:], mode='bilinear', align_corners=False)
        p4  = self.lat_4(f_4)   + F.interpolate(p8, size=f_4.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.smooth_4(p4)
        p4_coords = self.add_coords(p4)
        dense_feat = self.dense_refine(p4_coords)
        
        # 3. Dense Branch Prediction
        dense_center = self.center_heatmap_head(dense_feat) # [B, 1, H/4, W/4]
        dense_geom = self.corner_geom_head(dense_feat)      # [B, 10, 96, 96]
        
        # Center Peak Extraction (Argmax)
        H, W = dense_center.shape[2:]
        flat_center = dense_center.view(B, -1)
        _, max_idx = torch.max(flat_center, dim=-1) # [B]
        
        grid_y = max_idx // W
        grid_x = max_idx % W
        
        # Gather predicted geometry exactly at the single center peak location
        b_idx_gather = torch.arange(B, device=x.device)
        
        # Bounding Box Prior (Width, Height)
        w_pred = torch.sigmoid(dense_geom[b_idx_gather, 0, grid_y, grid_x]) # [B]
        h_pred = torch.sigmoid(dense_geom[b_idx_gather, 1, grid_y, grid_x]) # [B]
        
        # Quad refinement offsets (Bounded by Tanh to stay within ~20% of image size)
        quad_raw = dense_geom[b_idx_gather, 2:10, grid_y, grid_x]          # [B, 8]
        quad_pred = torch.tanh(quad_raw) * 0.2                             # [B, 8]
        
        cx = (grid_x.float() / W)
        cy = (grid_y.float() / H)
        
        # Reconstruct Bounding Box Corner anchors ([TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y])
        box_tl_x = cx - w_pred / 2.0
        box_tl_y = cy - h_pred / 2.0
        box_tr_x = cx + w_pred / 2.0
        box_tr_y = cy - h_pred / 2.0
        box_br_x = cx + w_pred / 2.0
        box_br_y = cy + h_pred / 2.0
        box_bl_x = cx - w_pred / 2.0
        box_bl_y = cy + h_pred / 2.0
        
        base_box = torch.stack([
            box_tl_x, box_tl_y, box_tr_x, box_tr_y, 
            box_br_x, box_br_y, box_bl_x, box_bl_y
        ], dim=-1) # [B, 8]
        
        # Final Corners = Base Box + Quad Refinements
        corners_visual_flat = base_box + quad_pred
        
        # Reshape to [B, 4, 2] and hard clamp
        corners_visual = corners_visual_flat.view(B, 4, 2)
        corners_visual = torch.clamp(corners_visual, 0.0, 1.0)

        corners_final = sort_corners_clockwise(corners_visual)

        return {
            'score': torch.ones((B, 1), device=x.device),
            'corners': corners_final,           # Semantically sorted corners (angle-sorted)
            'corners_visual': corners_visual,   # Raw predicted corners matching visual GT
            'dense_center': dense_center,       # [B, 1, H/4, W/4]
            'dense_geom': dense_geom,           # [B, 10, H/4, W/4]
        }
