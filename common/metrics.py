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


def calculate_accuracy_metrics(errors: Any, thresholds: Tuple[int, ...] = (1, 2, 3, 5, 10)) -> Dict[str, float]:
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


class BoundingBoxCornerL1Loss(nn.Module):
    """L1 Loss for the 10 spatial geometry channels (width, height, 8 quad offsets) supervised ONLY at the true GT center."""
    def forward(self, pred_geom: torch.Tensor, gt_centers: torch.Tensor, gt_corners: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred_geom.shape
        device = pred_geom.device
        
        # True center integer locations
        center_grid_x = gt_centers[:, 0, 0] * W
        center_grid_y = gt_centers[:, 0, 1] * H
        
        idx_x = torch.clamp(torch.floor(center_grid_x).long(), 0, W - 1)
        idx_y = torch.clamp(torch.floor(center_grid_y).long(), 0, H - 1)
        
        # Gather predictions at center location
        b_idx = torch.arange(B, device=device)
        pred_raw = pred_geom[b_idx, :, idx_y, idx_x] # [B, 10]
        
        w_pred = torch.sigmoid(pred_raw[:, 0])
        h_pred = torch.sigmoid(pred_raw[:, 1])
        quad_pred = torch.tanh(pred_raw[:, 2:10]) * 0.2
        
        # Real mapped center in [0, 1] based on the grid index
        cx = (idx_x.float() / W)
        cy = (idx_y.float() / H)
        
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
        pred_corners_flat = base_box + quad_pred
        gt_corners_flat = gt_corners.view(B, 8)
        
        return F.l1_loss(pred_corners_flat, gt_corners_flat, reduction='mean')


class SoftArgmax2D(nn.Module):
    """Differentiable 2D Argmax layer for sub-pixel keypoint localization.
    
    Converts a heatmap into [0, 1] normalized coordinates by computing the 
    center-of-mass of the softmax distribution.
    """
    def __init__(self, beta: float = 100.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1)
        weights = F.softmax(x_flat * self.beta, dim=-1)
        
        # Create coordinate grid in [0, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        )
        # [H*W, 2]
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        
        # Weighted sum of coordinates
        coords = torch.matmul(weights, grid) # [B, C, 2]
        return coords.squeeze(1)


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


class HeatmapFocalLoss(nn.Module):
    """Gaussian heatmap loss for corner supervision using keypoint Focal Loss.
    
    Prevents background (0s) from overwhelmingly dominating sparse peaks (1s).
    Based on CornerNet/CenterNet formulations.
    """
    def __init__(self, sigma: float = 2.0, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_heatmaps: torch.Tensor, gt_corners: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_heatmaps (torch.Tensor): [B, 4, H, W] raw logits.
            gt_corners (torch.Tensor): [B, 4, 2] normalized coordinates.
        """
        B, C, H, W = pred_heatmaps.size()
        device = pred_heatmaps.device
        
        # 1. Generate target Gaussian heatmaps
        yy = torch.linspace(0, H - 1, H, device=device)
        xx = torch.linspace(0, W - 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]
        
        # Gaussian anchors MUST be placed EXACTLY at integer coordinate bounds
        # Otherwise target.eq(1) fails across the floating-point gap, suppressing all positive gradients!
        gt_px = gt_corners.view(B, C, 1, 1, 2) * torch.tensor([W, H], device=device, dtype=torch.float32).view(1, 1, 1, 1, 2)
        gt_px_int = torch.floor(gt_px)
        gt_px_int[..., 0] = torch.clamp(gt_px_int[..., 0], 0, W - 1)
        gt_px_int[..., 1] = torch.clamp(gt_px_int[..., 1], 0, H - 1)
        
        dist_sq = torch.sum((grid.view(1, 1, H, W, 2) - gt_px_int) ** 2, dim=-1)
        target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        # 2. Keypoint Focal Loss
        pred = torch.clamp(torch.sigmoid(pred_heatmaps), min=1e-4, max=1-1e-4)
        
        # Exact peaks (target behaves as 1)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, self.beta)
        
        loss_pos = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        loss_neg = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        
        num_pos = pos_inds.sum()
        
        loss_pos = loss_pos.sum()
        loss_neg = loss_neg.sum()
        
        if num_pos == 0:
            loss = -loss_neg
        else:
            loss = -(loss_pos + loss_neg) / num_pos
            
        return loss
