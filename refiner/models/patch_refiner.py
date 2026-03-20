import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class HeatmapRefinerNet(nn.Module):
    """
    Stage 2: Heatmap-based Patch Refiner Network.
    Uses a spatial heatmap + Soft-Argmax to preserve precision and 
    handle larger Stage 1 offsets robustly.
    """
    def __init__(self, input_size=96, heatmap_size=24):
        super().__init__()
        self.heatmap_size = heatmap_size
        
        # Backbone: Preserves more spatial resolution
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),   # 96 -> 48
            ConvBNReLU(32, 64, stride=1),  # 48 -> 48
            ConvBNReLU(64, 64, stride=2),  # 48 -> 24
            ConvBNReLU(64, 128, stride=1), # 24 -> 24
            ConvBNReLU(128, 128, stride=1),
        )
        
        # Heatmap Head: Predicts a single-channel spatial mask
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 1, 1) # Raw logits for SoftArgmax
        )
        
        from common.metrics import SoftArgmax2D
        # beta=20.0 is much more stable than 100.0 for initial training
        self.soft_argmax = SoftArgmax2D(beta=20.0)
        
        # Zero-initialize the last layer so the model starts with a flat heatmap (prediction at center 0.5, 0.5)
        nn.init.zeros_(self.heatmap_head[-1].weight)
        nn.init.zeros_(self.heatmap_head[-1].bias)

    def forward(self, x):
        """
        Input: [B, 3, 96, 96]
        Output: [B, 2] offsets in [0, 1] patch space
        """
        feats = self.features(x)
        heatmap = self.heatmap_head(feats)
        coords = self.soft_argmax(heatmap)
        return coords

# Re-export as PatchRefinerNet for backward compatibility in imports
PatchRefinerNet = HeatmapRefinerNet
