import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
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


class FullCardRefinerNet(nn.Module):
    """
    Two-Stage Refiner:
    1. Coarse Global Head: Predicts initial 4 corners from full image context.
    2. RoI Refinement Head: Dynamically aligns features to the coarse prediction 
       and refines corners via local heatmaps.
    """
    def __init__(self, input_size=(320, 192)):
        super().__init__()
        self.input_size = input_size
        
        # 1. Shared Backbone: Stride 4 -> 8 -> 16
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),   # 1/2
            ConvBNReLU(32, 64, stride=2),  # 1/4 (Stride 4)
        )
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 64, stride=1),
            ConvBNReLU(64, 128, stride=2), # 1/8
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 128, stride=1),
            ConvBNReLU(128, 256, stride=2), # 1/16
        )
        
        # 2. Coarse Global Head (from Stride 16 features)
        self.coarse_head = nn.Sequential(
            ConvBNReLU(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8), # 4 corners * 2 (x,y)
            nn.Sigmoid()
        )
        
        # 3. Dynamic RoI Extractor
        self.roi_align = ops.RoIAlign(
            output_size=56, 
            spatial_scale=0.25, # Extract from Stride 4
            sampling_ratio=2,
            aligned=True
        )
        
        # 4. Refinement Keypoint Head
        self.refine_head = nn.Sequential(
            ConvBNReLU(64, 128, stride=1),
            ConvBNReLU(128, 128, stride=1),
            ConvBNReLU(128, 128, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 112x112
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1)
        )
        
        self.soft_argmax = SoftArgmax2D(beta=100.0)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Bias for heatmap background
        self.refine_head[-1].bias.data.fill_(-2.19)

    def get_roi_boxes_from_pts(self, pts, img_size, padding=0.1):
        """
        Computes Enclosing BBOX for quads and expands them.
        Args:
            pts: [B, 4, 2] in [0, 1] space
            img_size: (W, H)
            padding: extra margin as ratio of box size
        Returns:
            boxes: [B, 4] as [x1, y1, x2, y2] in pixels
        """
        B = pts.size(0)
        W, H = img_size
        
        min_pts = pts.min(dim=1)[0] # [B, 2]
        max_pts = pts.max(dim=1)[0] # [B, 2]
        
        centers = (min_pts + max_pts) / 2
        dims = (max_pts - min_pts) * (1.0 + padding)
        
        # Clip to [0, 1]
        x1 = torch.clamp(centers[:, 0] - dims[:, 0] / 2, 0, 1)
        y1 = torch.clamp(centers[:, 1] - dims[:, 1] / 2, 0, 1)
        x2 = torch.clamp(centers[:, 0] + dims[:, 0] / 2, 0, 1)
        y2 = torch.clamp(centers[:, 1] + dims[:, 1] / 2, 0, 1)
        
        # Map to pixel space
        boxes = torch.stack([x1 * W, y1 * H, x2 * W, y2 * H], dim=-1)
        return boxes

    def forward(self, x, gt_pts=None):
        """
        Returns:
            coarse_pts: [B, 4, 2] in [0, 1] space
            refined_pts: [B, 4, 2] in [0, 1] space
            heatmaps: [B, 4, 112, 112] raw logits in RoI space
            roi_boxes: [B, 4] pixels in input image space (ACTIVE RoI used for align)
            pred_roi_boxes: [B, 4] pixels in input image space (Derived from coarse)
        """
        B, C, H, W = x.shape
        
        # 1. Shared Backbone
        feat_s4 = self.stem(x)   # 1/4
        feat_s8 = self.stage1(feat_s4) # 1/8
        feat_s16 = self.stage2(feat_s8) # 1/16
        
        # 2. Coarse Head
        coarse_out = self.coarse_head(feat_s16) # [B, 8]
        coarse_pts = coarse_out.view(B, 4, 2)
        
        # 3. Dynamic RoI Extraction
        # Training: use GT to avoid collapse if coarse is wrong.
        # Val/Inf: use coarse predicted corners.
        pred_roi_boxes = self.get_roi_boxes_from_pts(coarse_pts, (W, H))
        
        if gt_pts is not None:
            roi_boxes = self.get_roi_boxes_from_pts(gt_pts, (W, H))
        else:
            roi_boxes = pred_roi_boxes
            
        indices = torch.arange(B, dtype=x.dtype, device=x.device).view(-1, 1)
        roi_with_indices = torch.cat([indices, roi_boxes], dim=1)
        roi_feats = self.roi_align(feat_s4, roi_with_indices)
        
        # 4. Refinement Head
        heatmaps = self.refine_head(roi_feats)
        roi_coords = self.soft_argmax(heatmaps) # [B, 4, 2]
        
        # 5. Map RoI coords [0, 1] to Crop space [0, 1]
        roi_x1 = roi_boxes[:, 0].view(B, 1)
        roi_y1 = roi_boxes[:, 1].view(B, 1)
        roi_w = (roi_boxes[:, 2] - roi_boxes[:, 0]).view(B, 1)
        roi_h = (roi_boxes[:, 3] - roi_boxes[:, 1]).view(B, 1)
        
        # Avoid division by zero
        roi_w = torch.clamp(roi_w, min=1e-6)
        roi_h = torch.clamp(roi_h, min=1e-6)
        
        refined_x = (roi_coords[:, :, 0] * roi_w + roi_x1) / W
        refined_y = (roi_coords[:, :, 1] * roi_h + roi_y1) / H
        refined_pts = torch.stack([refined_x, refined_y], dim=-1)
        
        return coarse_pts, refined_pts, heatmaps, roi_boxes, pred_roi_boxes


# Compatibility
FullCardCornerNet = FullCardRefinerNet
PatchRefinerNet = FullCardRefinerNet
