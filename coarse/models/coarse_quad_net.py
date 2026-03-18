import torch
import torch.nn as nn
from typing import Tuple, Optional

class ConvBNReLU(nn.Module):
    """Convolution followed by Batch Normalization and ReLU6.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Convolution stride. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Convolution stride. Defaults to 1.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNReLU(in_planes, in_planes, stride=stride, groups=in_planes)
        self.pw = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.relu(self.bn(self.pw(self.dw(x))))

class SpatialSoftArgmax2D(nn.Module):
    """Differentiable spatial to numerical transform (DSNT) with optional offset aggregation.
    
    Extracts continuous [0, 1] coordinates from 2D heatmaps.
    """
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        
    def forward(self, heatmaps: torch.Tensor, offset_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            heatmaps (torch.Tensor): Unnormalized heatmaps of shape [B, C, H, W].
            offset_maps (torch.Tensor, optional): Spatial offset fields [B, C * 2, H, W] for X and Y shifts.
            
        Returns:
            torch.Tensor: Decoded [B, C, 2] normalized coordinates in range [0, 1].
        """
        B, C, H, W = heatmaps.size()
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, C, H * W)
        
        # Apply softmax over spatial dimension to get probability distributions
        weights = torch.nn.functional.softmax(heatmaps_flat / self.temperature, dim=-1) # [B, C, H*W]
        
        # Create normalized coordinate grid (trace-safe implementation)
        range_y = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
        range_x = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
        
        grid_y = range_y.view(H, 1).expand(H, W)
        grid_x = range_x.view(1, W).expand(H, W)
        
        # Normalize grid to [0, 1] relative to center of bins
        grid_x = (grid_x + 0.5) / W
        grid_y = (grid_y + 0.5) / H
        
        grid_x_flat = grid_x.contiguous().view(1, 1, H * W)
        grid_y_flat = grid_y.contiguous().view(1, 1, H * W)
        
        # Compute expected coordinates via soft-argmax
        expected_x = torch.sum(weights * grid_x_flat, dim=-1) # [B, C]
        expected_y = torch.sum(weights * grid_y_flat, dim=-1) # [B, C]
        
        coarse_coords = torch.stack([expected_x, expected_y], dim=-1) # [B, C, 2]
        
        # Aggregate sub-pixel offsets if provided
        if offset_maps is not None:
            # Reshape offsets to [B, C, 2, H*W]
            offsets_flat = offset_maps.view(B, C, 2, H * W)
            # Soft-sum offsets
            expected_offsets = torch.sum(weights.unsqueeze(2) * offsets_flat, dim=-1) # [B, C, 2]
            refined_coords = coarse_coords + expected_offsets
            return refined_coords
            
        return coarse_coords

class CoarseQuadNet(nn.Module):
    """Coarse Quadrilateral Network for ID Card Corner Detection.
    
    Improved stage-1 model featuring a fully convolutional feature extractor and 
    a hybrid DSNT (soft-argmax) spatial prediction head. Instead of collapsing 
    to a single global vector early, it produces low-resolution per-corner heatmaps 
    and associated offset maps, preserving spatial layout and geometric robustness.
    
    Predicts 4 ordered coarse corners (top-left, top-right, bottom-right, bottom-left)
    normalized to [0, 1]. Total params: ~0.8M. Target deploy bounds suitable for CPU.
    """
    def __init__(self) -> None:
        super().__init__()
        
        # Backbone input shape: [B, 3, 384, 384]
        self.stem = ConvBNReLU(3, 16, stride=2)                     # -> 192x192x16
        
        # Lightweight MobileNet-style downsampling
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),               # -> 96x96x32
            DepthwiseSeparableConv(32, 64, stride=2),               # -> 48x48x64
            DepthwiseSeparableConv(64, 64, stride=1),               # -> 48x48x64
            DepthwiseSeparableConv(64, 128, stride=2),              # -> 24x24x128
            DepthwiseSeparableConv(128, 128, stride=1),             # -> 24x24x128
            DepthwiseSeparableConv(128, 256, stride=2),             # -> 12x12x256
            DepthwiseSeparableConv(256, 256, stride=1),             # -> 12x12x256
            DepthwiseSeparableConv(256, 512, stride=1),             # -> 12x12x512
            DepthwiseSeparableConv(512, 512, stride=1),             # -> 12x12x512
        )
        
        # Spatial Prediction Head
        # Heatmap branch: 4 channels (one per corner)
        self.heatmap_head = nn.Sequential(
            DepthwiseSeparableConv(512, 256, stride=1),
            nn.Conv2d(256, 4, kernel_size=1, stride=1)
        )
        
        # Offset branch: 8 channels (dx, dy for each of the 4 corners)
        self.offset_head = nn.Sequential(
            DepthwiseSeparableConv(512, 256, stride=1),
            nn.Conv2d(256, 8, kernel_size=1, stride=1)
        )
        
        # Optional global score head
        self.score_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.score_head = nn.Linear(512, 1)
        
        # Soft-argmax coordinate decoder
        self.decoder = SpatialSoftArgmax2D(temperature=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor [B, 3, 384, 384].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - score: Global confidence score [B, 1] mapped to [0,1].
                - corners: Normalized coordinate tensor [B, 4, 2] constrained to [0,1],
                  incorporating robust proxy heatmap distributions with spatial offset shifts.
        """
        x = self.stem(x)
        features = self.blocks(x)                                   # [B, 512, 12, 12]
        
        # Predict robust spatial probabilities and geometric refinements
        heatmaps = self.heatmap_head(features)                      # [B, 4, 12, 12]
        offsets = self.offset_head(features)                        # [B, 8, 12, 12]
        
        # Diff-friendly decode of exact coordinates
        corners = self.decoder(heatmaps, offsets)                   # [B, 4, 2]
        
        # Clamp bounds strictly within typical image dimensions to maintain [0, 1] scale stability
        corners = torch.clamp(corners, 0.0, 1.0)
        
        # Predict overall object existence score
        pooled = self.score_pool(features)                          # [B, 512, 1, 1]
        pooled = torch.flatten(pooled, 1)
        score = torch.sigmoid(self.score_head(pooled))              # [B, 1]
        
        return score, corners
