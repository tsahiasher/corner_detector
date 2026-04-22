import torch
import torch.nn as nn
import torch.nn.functional as F
from common.metrics import SoftArgmax2D


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class FullCardRefinerNet(nn.Module):
    """
    High-Precision Stage 2: Full-Card Corner Refiner.
    Uses Hierarchical Local Refinement:
    1. Coarse prediction from bottleneck features (Stride 32).
    2. Local patch sampling from high-resolution features (Stride 8).
    3. Residual refinement from local patches.
    Input Size: 640x640
    """
    def __init__(self, input_size=640, patch_size=7):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size # Step 1 Regression Patch Size (7x7 at Stride 8 = 56px)
        
        # 1. Backbone: Stride 4 -> 8 -> 16 -> 32
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),   # 320
            ConvBNReLU(32, 64, stride=2),  # 160 (Stride 4)
        )
        
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 64, stride=1),
            ConvBNReLU(64, 128, stride=2), # 80 (Stride 8) - FINE FEATURES (f1)
        )
        
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 128, stride=1),
            ConvBNReLU(128, 256, stride=2), # 40 (Stride 16)
        )
        
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 256, stride=1),
            ConvBNReLU(256, 512, stride=2), # 20 (Stride 32) - COARSE FEATURES (f3)
        )
        
        # 2. Coarse Head: 512 x 20 x 20 -> 4 points [0, 1]
        self.coarse_head = nn.Sequential(
            ConvBNReLU(512, 128, stride=2), # 10 x 10
            ConvBNReLU(128, 64, stride=2),  # 5 x 5
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU6(inplace=True),
            nn.Linear(256, 8),
            nn.Sigmoid() 
        )
        
        # 3. Local Refine Head 1: Shared for 4 patches [128, 7, 7]
        self.local_head = nn.Sequential(
            ConvBNReLU(128, 64, stride=1),
            ConvBNReLU(64, 64, stride=1),  
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, 2), # 2D residual (dx, dy)
            nn.Tanh() # Bounded [-1, 1]
        )
        
        # 3.1 Auxiliary Diagnostic Heatmap Head: Larger patches [128, 31, 31]
        self.heatmap_patch_size = 31
        self.aux_heatmap_head = nn.Sequential(
            ConvBNReLU(128, 64, stride=1),
            ConvBNReLU(64, 64, stride=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # 4. Fusion Layer for high-resolution refinement [128+64 -> 128]
        self.fine_fusion = ConvBNReLU(128 + 64, 128, stride=1)
        
        # 5. Local Refine Head 2: Smaller patches [128, 5, 5]
        self.patch_size_2 = 5
        self.local_head_2 = nn.Sequential(
            ConvBNReLU(128, 64, stride=1),
            ConvBNReLU(64, 64, stride=1),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_local_patches(self, features, coords, p_size=None):
        """
        Differentiable sampling of local patches around predicted coords.
        Args:
            features: [B, C, Hf, Wf] (e.g. 80x80)
            coords: [B, 4, 2] in [0, 1] normalized space
        Returns:
            patches: [B*4, C, S, S]
        """
        B, C, Hf, Wf = features.size()
        S = p_size if p_size is not None else self.patch_size
        device = features.device
        
        # 1. Map [0, 1] to [-1, 1] for grid_sample
        center_pts = 2.0 * coords - 1.0 # [B, 4, 2]
        
        # 2. Create relative grid for SxS patch
        # Step in [-1, 1] space is 2.0 / feature_resolution
        step_x = 2.0 / Wf
        step_y = 2.0 / Hf
        
        offsets = torch.linspace(-(S-1)/2, (S-1)/2, S, device=device)
        ox, oy = torch.meshgrid(offsets, offsets, indexing='xy') # [S, S]
        rel_grid = torch.stack([ox * step_x, oy * step_y], dim=-1) # [S, S, 2]
        
        # 3. Create global grid: [B, 4, S, S, 2]
        # center_pts: [B, 4, 1, 1, 2], rel_grid: [1, 1, S, S, 2]
        grid = center_pts.view(B, 4, 1, 1, 2) + rel_grid.view(1, 1, S, S, 2)
        
        # 4. Reshape for grid_sample: [B, 4*S, S, 2]
        grid = grid.view(B, 4 * S, S, 2)
        
        # 5. Sample: [B, C, 4*S, S]
        sampled = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # 6. Reshape to [B*4, C, S, S]
        patches = sampled.view(B, C, 4, S, S).permute(0, 2, 1, 3, 4).reshape(B * 4, C, S, S)
        return patches

    def forward(self, x):
        """
        Args:
            x: [B, 3, 640, 640]
        Returns:
            final_coords: [B, 4, 2]
            coarse_coords: [B, 4, 2]
        """
        B = x.size(0)
        s = self.stem(x)         # 160x160
        f1 = self.stage1(s)      # 80x80 (fine resolution)
        f2 = self.stage2(f1)     # 40x40
        f3 = self.stage3(f2)     # 20x20
        
        # Step 1: Coarse Prediction [B, 4, 2]
        coarse = self.coarse_head(f3).view(B, 4, 2)
        
        # Step 2: First Residual Refinement (Regression, Stride 8, 7x7)
        # Restore Phase 1 to stable regression path
        patches1 = self.extract_local_patches(f1, coarse, p_size=self.patch_size)
        res1 = self.local_head(patches1).view(B, 4, 2)
        refined1 = torch.clamp(coarse + res1 * 0.1, 0.0, 1.0)
        
        # Construct Stride 4 Fine Feature Map (160x160) for Stage 2 & Auxiliary Head
        up_f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=False)
        f_fine = self.fine_fusion(torch.cat([up_f1, s], dim=1))
        
        # Auxiliary Path: Diagnostic heatmap from separate large patches (Stride 4, 31x31)
        # Detach features/coordinates to isolate gradients from the main regression model
        hm_patches = self.extract_local_patches(f_fine.detach(), coarse.detach(), p_size=self.heatmap_patch_size)
        aux_heatmaps = self.aux_heatmap_head(hm_patches) # [B*4, 1, 31, 31]
        
        # Step 3: Second Residual Refinement (Regression, Stride 4, 5x5)
        patches2 = self.extract_local_patches(f_fine, refined1, p_size=self.patch_size_2)
        res2 = self.local_head_2(patches2).view(B, 4, 2)
        final = torch.clamp(refined1 + res2 * 0.05, 0.0, 1.0)
        
        return final, refined1, coarse, aux_heatmaps.view(B, 4, self.heatmap_patch_size, self.heatmap_patch_size)

# Compatibility alias
FullCardCornerNet = FullCardRefinerNet
PatchRefinerNet = FullCardRefinerNet
