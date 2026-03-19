import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Union, Any
import numpy as np


def compute_pixel_error(pred_corners: torch.Tensor, target_corners: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Computes the Euclidean pixel error distance per corner vertex."""
    scale = torch.tensor([width, height], dtype=torch.float32, device=pred_corners.device)
    pred_px = pred_corners * scale
    target_px = target_corners * scale
    diff = pred_px - target_px
    dist = torch.norm(diff, dim=-1)  # [B, 4]
    return dist


def compute_patch_recall(errors: Any, patch_sizes: Tuple[int, ...] = (64, 80, 96)) -> Dict[str, float]:
    """Computes patch recall percentage from distance errors.

    Robust to both torch.Tensor and numpy.ndarray.
    """
    if isinstance(errors, torch.Tensor):
        errors_np = errors.detach().cpu().numpy()
    else:
        errors_np = np.array(errors)
        
    results: Dict[str, float] = {}
    total_elements = errors_np.size
    if total_elements == 0:
        return {f'recall_{ps}': 0.0 for ps in patch_sizes}

    for ps in patch_sizes:
        # A corner is within a patch of size P if L-inf distance <= P/2
        # However, typically "Mean Pixel Error" is L2. 
        # For simplicity and consistency with coarse Stage 1 goal (GT in patch),
        # we check if Euclidean distance is <= PatchSize/2.
        # This is a bit more conservative than L-inf.
        recalled = (errors_np <= (ps / 2.0)).sum()
        recall_pct = (recalled / total_elements) * 100.0
        results[f'recall_{ps}'] = recall_pct
    return results


def calculate_accuracy_metrics(errors: Any, thresholds: Tuple[int, ...] = (2, 3, 5, 10)) -> Dict[str, float]:
    """Calculates accuracy summary metrics. Robust to both torch.Tensor and numpy.ndarray."""
    if isinstance(errors, torch.Tensor):
        errors_np = errors.detach().cpu().numpy()
    else:
        errors_np = np.array(errors)

    if errors_np.size == 0:
        return {}

    flat_errors = errors_np.flatten()
    metrics: Dict[str, float] = {
        'mean': float(np.mean(flat_errors)),
        'median': float(np.median(flat_errors)),
        'p90': float(np.percentile(flat_errors, 90)),
        'p95': float(np.percentile(flat_errors, 95)),
        'max': float(np.max(flat_errors)),
    }

    if errors_np.ndim == 2 and errors_np.shape[1] == 4:
        metrics['tl'] = float(np.mean(errors_np[:, 0]))
        metrics['tr'] = float(np.mean(errors_np[:, 1]))
        metrics['br'] = float(np.mean(errors_np[:, 2]))
        metrics['bl'] = float(np.mean(errors_np[:, 3]))

    for t in thresholds:
        acc = np.mean(flat_errors < t) * 100.0
        metrics[f'acc_{t}px'] = float(acc)

    return metrics


class WingLoss(nn.Module):
    """Wing Loss for robust keypoint regression."""
    def __init__(self, wing_w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = wing_w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1.0 + self.w / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                width: Union[float, torch.Tensor] = 384.0,
                height: Union[float, torch.Tensor] = 384.0) -> torch.Tensor:
        """
        Supports both Stage 1 (width=384, height=384) and 
        Stage 2 (width=patch_size, height=patch_size).
        """
        B = pred.size(0)
        diff = torch.abs(pred - target)
        
        # Handle scaling to pixel space
        if isinstance(width, torch.Tensor) and isinstance(height, torch.Tensor):
            w_tens: torch.Tensor = width
            h_tens: torch.Tensor = height
            # Map tensors to [B, 1] or [B, 1, 1] to match pred dims
            while w_tens.dim() < pred.dim():
                w_tens = w_tens.unsqueeze(-1)
                h_tens = h_tens.unsqueeze(-1)
            scale = torch.cat([w_tens, h_tens], dim=-1)
        else:
            w_f: float = float(width)
            h_f: float = float(height)
            scale = torch.tensor([w_f, h_f], device=pred.device)
        
        diff_scaled = diff * scale
        small = diff_scaled < self.w
        loss_small = self.w * torch.log(1.0 + diff_scaled / self.epsilon)
        loss_large = diff_scaled - self.C
        loss = torch.where(small, loss_small, loss_large)
        return loss.mean()


class SoftArgmax2D(nn.Module):
    """
    Differentiable Soft-Argmax for sub-pixel heatmap localization.
    Maps a heatmap to [0, 1] coordinates.
    """
    def __init__(self, beta: float = 100.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, H, W] or [B, 1, H, W]
        Output: [B, 2] in [0, 1]
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        B, H, W = x.shape
        device = x.device

        # Softmax over spatial dimensions
        flat_x = x.view(B, -1)
        weights = F.softmax(flat_x * self.beta, dim=-1)
        weights = weights.view(B, H, W)

        # Create coordinate grids
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )

        # Weighted average
        expected_y = torch.sum(weights * pos_y, dim=(1, 2))
        expected_x = torch.sum(weights * pos_x, dim=(1, 2))

        return torch.stack([expected_x, expected_y], dim=-1)


class QuadShapeLoss(nn.Module):
    """Geometry-aware quadrilateral shape regularizer and ordering constraint."""
    def __init__(self, weight_diag: float = 1.0, weight_convex: float = 1.0) -> None:
        super().__init__()
        self.weight_diag = weight_diag
        self.weight_convex = weight_convex

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        # 1. Diagonal Ratio (encourages rectangle-like aspect ratios)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        dists = []
        for i, j in pairs:
            d = torch.norm(corners[:, i, :] - corners[:, j, :], dim=-1)
            dists.append(d)
        dist_stack = torch.stack(dists, dim=-1)
        top2, _ = dist_stack.topk(2, dim=-1)
        diag_loss = torch.abs(top2[:, 0] - top2[:, 1]).mean()

        # 2. Clockwise Convexity & Ordering (TL -> TR -> BR -> BL)
        # In a y-down coordinate system, clockwise vectors should have positive cross products.
        edges = [
            corners[:, 1, :] - corners[:, 0, :], # TL -> TR
            corners[:, 2, :] - corners[:, 1, :], # TR -> BR
            corners[:, 3, :] - corners[:, 2, :], # BR -> BL
            corners[:, 0, :] - corners[:, 3, :], # BL -> TL
        ]
        cross_products = []
        for i in range(4):
            e1 = edges[i]
            e2 = edges[(i + 1) % 4]
            # Cross product: x1*y2 - y1*x2
            cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            cross_products.append(cross)
        
        cross_stack = torch.stack(cross_products, dim=-1)
        # Penalize negative cross products (counter-clockwise or flipped)
        # and small cross products (nearly collinear)
        convex_loss = torch.relu(0.1 - cross_stack).mean() 
        
        # 3. Min Distance (avoid collapsed corners)
        min_dist_loss = torch.relu(0.05 - dist_stack).mean()

        return self.weight_diag * diag_loss + self.weight_convex * (convex_loss + min_dist_loss)


class GeometryAlignmentLoss(nn.Module):
    """Enforces that corners align with predicted geometric features (e.g., edges)."""
    def forward(self, corners: torch.Tensor, edge_map: torch.Tensor, mask_map: torch.Tensor) -> torch.Tensor:
        # corners: [B, 4, 2] in [0, 1]
        # edge_map: [B, 1, H, W] in [0, 1]
        B, N, _ = corners.size()
        
        # Grid sample expects [-1, 1] with (x, y) order
        grid = corners.view(B, N, 1, 2) * 2.0 - 1.0
        
        # 1. Edge Alignment: Corners should be on the peak of predicted edges
        # We sample a 3x3 neighborhood around the corner to provide better gradients
        # if the predicted edge is slightly offset.
        offsets = torch.tensor([[-1, -1], [0, -1], [1, -1],
                               [-1,  0], [0,  0], [1,  0],
                               [-1,  1], [0,  1], [1,  1]], device=corners.device, dtype=torch.float32)
        # Scale offsets to grid space (at 96x96 resolution for v2 boost)
        pix_step = 2.0 / 96.0 
        neighborhood_grid = grid.unsqueeze(2) + offsets.view(1, 1, 9, 2) * pix_step
        
        # [B, 1, 4, 9]
        sampled_edges = F.grid_sample(edge_map, neighborhood_grid.view(B, N*9, 1, 2), 
                                     mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_edges = sampled_edges.view(B, N, 9)
        
        # Max-pool over neighborhood to find the strongest edge nearby
        best_edge, _ = torch.max(sampled_edges, dim=-1)
        edge_align_loss = (1.0 - best_edge).mean()
        
        # 2. Mask Alignment: Corners should be exactly at the 0.5 boundary
        sampled_mask = F.grid_sample(mask_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # Use squared error for stronger gradients near the boundary
        mask_align_loss = ((sampled_mask.squeeze(1).squeeze(-1) - 0.5) ** 2).mean()
        
        return edge_align_loss + 2.0 * mask_align_loss


class DiceLoss(nn.Module):
    """Dice Loss for robust segmentation."""
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()


class QuadConsistencyLoss(nn.Module):
    """Loss to synchronize local and global corner predictions."""
    def forward(self, corners_spatial: torch.Tensor, corners_global: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(corners_spatial, corners_global)


class HomographyReprojectionLoss(nn.Module):
    """Loss that penalizes the reprojected distance across the entire card surface."""
    def __init__(self, grid_size: int = 4) -> None:
        super().__init__()
        self.grid_size = grid_size
        x = torch.linspace(0.1, 0.9, grid_size)
        y = torch.linspace(0.1, 0.9, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('grid_pts', torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1))

    def _get_homography_matrix(self, corners: torch.Tensor) -> torch.Tensor:
        device = corners.device
        B = corners.size(0)
        src = torch.tensor([[0,0],[1,0],[1,1],[0,1]], dtype=corners.dtype, device=device).unsqueeze(0).expand(B, 4, 2)
        A = []
        b = []
        for i in range(4):
            sx, sy = src[:, i, 0], src[:, i, 1]
            dx, dy = corners[:, i, 0], corners[:, i, 1]
            A.append(torch.stack([sx, sy, torch.ones_like(sx), torch.zeros_like(sx), torch.zeros_like(sx), torch.zeros_like(sx), -dx*sx, -dx*sy], dim=1))
            A.append(torch.stack([torch.zeros_like(sx), torch.zeros_like(sx), torch.zeros_like(sx), sx, sy, torch.ones_like(sx), -dy*sx, -dy*sy], dim=1))
            b.append(dx)
            b.append(dy)
        A = torch.stack(A, dim=1)
        b = torch.stack(b, dim=1).unsqueeze(-1)
        try:
            h = torch.linalg.solve(A, b).squeeze(-1)
        except RuntimeError:
            h = torch.zeros((B, 8), device=device)
            h[:, 0] = 1.0; h[:, 4] = 1.0
        H = torch.cat([h, torch.ones((B, 1), device=device)], dim=1).view(B, 3, 3)
        return H

    def forward(self, pred_corners: torch.Tensor, gt_corners: torch.Tensor) -> torch.Tensor:
        B = pred_corners.size(0)
        H_pred = self._get_homography_matrix(pred_corners)
        H_gt = self._get_homography_matrix(gt_corners)
        grid = self.grid_pts.to(pred_corners.device).unsqueeze(0).expand(B, -1, -1)
        def project(H, pts):
            pts_h = torch.cat([pts, torch.ones((B, pts.size(1), 1), device=pts.device)], dim=-1)
            proj_h = torch.bmm(pts_h, H.transpose(1, 2))
            proj = proj_h[:, :, :2] / (proj_h[:, :, 2:3] + 1e-8)
            return proj
        return F.smooth_l1_loss(project(H_pred, grid), project(H_gt, grid))


class HeatmapLoss(nn.Module):
    """Gaussian heatmap loss for corner supervision."""
    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, pred_heatmaps: torch.Tensor, gt_corners: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_heatmaps (torch.Tensor): [B, 4, H, W] raw logits.
            gt_corners (torch.Tensor): [B, 4, 2] normalized coordinates.
        """
        B, C, H, W = pred_heatmaps.size()
        device = pred_heatmaps.device
        
        # Generate target Gaussian heatmaps
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0) # [1, 1, H, W, 2]
        
        # Scale GT to pixel coords [B, 4, 1, 1, 2]
        gt_px = gt_corners.view(B, C, 1, 1, 2) * torch.tensor([W, H], device=device).view(1, 1, 1, 1, 2)
        
        # Gaussian: [B, 4, H, W]
        dist_sq = torch.sum((grid - gt_px) ** 2, dim=-1)
        target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        # Supervise sigmoid heatmaps
        return F.mse_loss(torch.sigmoid(pred_heatmaps), target)
