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

class IterativeRefinerNet(nn.Module):
    """
    Stage 2: Iterative High-Precision Patch Refiner.
    Performs a two-level search:
    1. Global: Finds the corner in the 96x96 patch (24x24 heatmap).
    2. Local: Crops a 32x32 sub-patch and predicts a high-precision offset.
    """
    def __init__(self, input_size=96, fine_patch_size=32):
        super().__init__()
        self.input_size = input_size
        self.fine_patch_size = fine_patch_size
        self.size_ratio = fine_patch_size / input_size
        
        # --- Stage 1: Global Backbone (96x96 -> 24x24) ---
        self.global_features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),   # 48
            ConvBNReLU(32, 64, stride=1),  # 48
            ConvBNReLU(64, 64, stride=2),  # 24
            ConvBNReLU(64, 128, stride=1), # 24
            ConvBNReLU(128, 128, stride=1),
        )
        
        self.coarse_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
        # --- Stage 2: Fine Refinement (32x32 -> 32x32) ---
        self.fine_features = nn.Sequential(
            ConvBNReLU(3, 32, stride=1),    # 32
            ConvBNReLU(32, 64, stride=2),   # 16
            ConvBNReLU(64, 64, stride=1),   # 16
            ConvBNReLU(64, 128, stride=2),  # 8
        )
        
        self.fine_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Upsample(size=(fine_patch_size, fine_patch_size), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1)
        )
        
        from common.metrics import SoftArgmax2D
        self.soft_argmax_coarse = SoftArgmax2D(beta=20.0)
        self.soft_argmax_fine = SoftArgmax2D(beta=50.0)
        
        # Initialization: Force initial predictions to center
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.fine_head[-1].weight)
        nn.init.zeros_(self.fine_head[-1].bias)

    @torch.jit.ignore
    def _get_grid(self, B: int, device: torch.device):
        """Helper to create sampling grid for local crop."""
        grid_base = torch.meshgrid(
            torch.linspace(-1, 1, self.fine_patch_size, device=device),
            torch.linspace(-1, 1, self.fine_patch_size, device=device),
            indexing='ij'
        )
        # Convert to (x, y) format for grid_sample
        grid_base = torch.stack([grid_base[1], grid_base[0]], dim=-1) # [fine, fine, 2]
        return grid_base.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, x):
        B = x.size(0)
        
        # 1. Coarse Stage
        g_feats = self.global_features(x)
        g_heatmap = self.coarse_head(g_feats)
        coarse_coords = self.soft_argmax_coarse(g_heatmap) # [B, 2] in [0, 1]
        
        # 2. Local Stage (Differentiable Zoom)
        # grid_sample expects coordinates in [-1, 1]
        grid = self._get_grid(B, x.device) * self.size_ratio
        shift = (coarse_coords * 2.0 - 1.0).view(B, 1, 1, 2)
        grid = grid + shift
        
        fine_patch = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # 3. Fine Stage
        f_feats = self.fine_features(fine_patch)
        f_heatmap = self.fine_head(f_feats)
        fine_offsets = self.soft_argmax_fine(f_heatmap) # [B, 2] in [0, 1] relative to fine_patch
        
        # 4. Integrate Coordinates
        # coarse_coords is absolute 96x96 space center
        # fine_offsets is [0, 1] within the 32x32 patch area
        final_coords = coarse_coords + (fine_offsets - 0.5) * self.size_ratio
        
        return final_coords, coarse_coords

# Re-export as PatchRefinerNet for backward compatibility in imports
PatchRefinerNet = IterativeRefinerNet
