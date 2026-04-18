"""Spatially-Aware Global Bounding Box Regressor for ID cards.

- Single object prediction per image.
- ResNet-18 Backbone.
- Soft-Attention localization pooling weighting salient features dynamically.
- Outputs cx, cy, w, h bounding box alongside explicit localization logits.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import torchvision.models as models

class BoundingBoxQuadNet(nn.Module):
    """Spatially-Aware Global Bounding Box ID Card detector.

    Detects a single ID card and outputs a bounding box representing it explicitly
    derived driven by native structural layout density masking representations.

    Architecture:
        Backbone    : ResNet-18 (ImageNet pre-trained), up to layer4 (512, 12x12 at 384px)
        Spatial Neck: Conv3x3 (256 ch) maintaining dense geometry bounds.
        Loc Head    : 1x1 Conv outputting single-channel positional layout logit grids.
        Attn Pooling: Element-wise weighting applying Softmax map directly across Spatial Net mappings natively extracting highly optimized single dense layout representation limits.
        Head        : Linear MLP mapping the final dynamic extracted vector onto exactly 4 target bound constraints natively.
        
    Outputs:
        A dictionary containing:
            'box': [B, 4] normalized (cx, cy, w, h)
            'loc_logits': [B, 1, 12, 12] explicit implicit positioning masks (train only)
    """

    def __init__(self) -> None:
        super().__init__()

        # ---- Backbone: ResNet-18 ----
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # output: [B, 512, H/32, W/32] -> at 384x384 it produces 12x12
        )

        # ---- Spatially-Aware Neck ----
        self.neck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # ---- Localization Probability Head ----
        self.loc_head = nn.Conv2d(256, 1, kernel_size=1)

        # ---- Final Extrapolation Box Target Configuration ----
        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)           # [B, 512, H/32, W/32] e.g. 12x12
        spatial_features = self.neck(features) # [B, 256, H/32, W/32]
        
        # Gen bounding probabilistic grid mapping layout implicitly
        loc_logits = self.loc_head(spatial_features) # [B, 1, H/32, W/32]
        
        # Sigmoid Spatial Gating
        B, C, H, W = spatial_features.size()
        attn_weights = torch.sigmoid(loc_logits)
        
        # Map element-wise across the spatial density, normalizing by total probability mass.
        weighted_features = spatial_features * attn_weights
        pooled_feat = weighted_features.sum(dim=(2, 3)) / (attn_weights.sum(dim=(2, 3)) + 1e-6) # [B, 256]
        
        box = torch.sigmoid(self.box_head(pooled_feat)) # [B, 4]

        return {
            'box': box,
            'loc_logits': loc_logits
        }
