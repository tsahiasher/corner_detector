import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, List


def compute_pixel_error(pred_corners: torch.Tensor, target_corners: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Computes the Euclidean pixel error distance per corner vertex.

    Args:
        pred_corners (torch.Tensor): Predicted normalized corners [B, 4, 2].
        target_corners (torch.Tensor): Target normalized corners [B, 4, 2].
        width (float): Original image width in pixels.
        height (float): Original image height in pixels.

    Returns:
        torch.Tensor: Euclidean distances in pixels of shape [B, 4].
    """
    scale = torch.tensor([width, height], dtype=torch.float32, device=pred_corners.device)
    pred_px = pred_corners * scale
    target_px = target_corners * scale

    diff = pred_px - target_px
    dist = torch.norm(diff, dim=-1)  # [B, 4]
    return dist


def compute_patch_recall(pred_corners: torch.Tensor, target_corners: torch.Tensor,
                         width: float, height: float,
                         patch_sizes: Tuple[int, ...] = (64, 80, 96)) -> Dict[str, float]:
    """Computes patch recall: fraction of GT corners that fall inside a square patch
    centered on the predicted corner.

    For a patch of size P, a GT corner is recalled if both |dx| <= P/2 and |dy| <= P/2
    (L-infinity / Chebyshev distance check), which corresponds to the GT point falling
    within the patch area.

    Args:
        pred_corners (torch.Tensor): Predicted normalized corners [B, 4, 2].
        target_corners (torch.Tensor): Target normalized corners [B, 4, 2].
        width (float): Original image width in pixels.
        height (float): Original image height in pixels.
        patch_sizes (Tuple[int, ...], optional): Patch sizes to evaluate. Defaults to (64, 80, 96).

    Returns:
        Dict[str, float]: Recall percentages keyed as 'patch_recall_{size}px'.
    """
    scale = torch.tensor([width, height], dtype=torch.float32, device=pred_corners.device)
    pred_px = pred_corners * scale
    target_px = target_corners * scale

    # Per-corner absolute differences [B, 4, 2]
    abs_diff = torch.abs(pred_px - target_px)

    results: Dict[str, float] = {}
    for ps in patch_sizes:
        half = ps / 2.0
        # A corner is recalled if BOTH dx and dy are within half the patch size
        within_x = abs_diff[..., 0] <= half  # [B, 4]
        within_y = abs_diff[..., 1] <= half  # [B, 4]
        recalled = (within_x & within_y).float()
        recall_pct = recalled.mean().item() * 100.0
        results[f'patch_recall_{ps}px'] = recall_pct

    return results


def calculate_accuracy_metrics(errors: torch.Tensor, thresholds: Tuple[int, ...] = (2, 3, 5, 10)) -> Dict[str, float]:
    """Calculates evaluation summary accuracy metrics from pixel distances.

    Args:
        errors (torch.Tensor): Tensor of per-corner distance errors. Shape [N, 4].
        thresholds (Tuple[int, ...], optional): Distance thresholds in pixels. Defaults to (2, 3, 5, 10).

    Returns:
        Dict[str, float]: Performance summary with mean, median, per-corner, outliers, and threshold coverage.
    """
    if errors.numel() == 0:
        return {}

    flat_errors = errors.reshape(-1)

    metrics: Dict[str, float] = {
        'mean_error': flat_errors.mean().item(),
        'median_error': flat_errors.median().item(),
        'p90_error': torch.quantile(flat_errors, 0.90).item() if flat_errors.numel() > 0 else 0.0,
        'p95_error': torch.quantile(flat_errors, 0.95).item() if flat_errors.numel() > 0 else 0.0,
        'max_error': flat_errors.max().item() if flat_errors.numel() > 0 else 0.0,
    }

    if errors.ndim == 2 and errors.size(1) == 4:
        metrics['tl_mean'] = errors[:, 0].mean().item()
        metrics['tr_mean'] = errors[:, 1].mean().item()
        metrics['br_mean'] = errors[:, 2].mean().item()
        metrics['bl_mean'] = errors[:, 3].mean().item()

    for t in thresholds:
        perc = (flat_errors < t).float().mean().item() * 100
        metrics[f'acc_under_{t}px'] = perc

    return metrics


class WingLoss(nn.Module):
    """Wing Loss for robust keypoint regression.

    Combines logarithmic behavior for small errors (boosting gradient signal for precise
    localization) with linear behavior for large errors (robustness to outliers/noise).
    This is superior to SmoothL1 for keypoint tasks because SmoothL1 has vanishing
    gradients for small errors exactly where we need the model to keep refining.

    Reference: Feng et al., "Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks", CVPR 2018.

    Args:
        wing_w (float): Threshold below which the log term dominates. Defaults to 10.0.
        epsilon (float): Curvature control inside the log region. Defaults to 2.0.
    """
    def __init__(self, wing_w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = wing_w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1.0 + self.w / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Wing Loss.

        Args:
            pred (torch.Tensor): Predicted coordinates [B, 4, 2].
            target (torch.Tensor): Target coordinates [B, 4, 2].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        diff = torch.abs(pred - target)
        # Scale normalized [0,1] coords to a pixel-like range for meaningful w threshold
        # With 384 input, multiply by 384 so w=10 means ~10px
        diff_scaled = diff * 384.0

        small = diff_scaled < self.w
        loss_small = self.w * torch.log(1.0 + diff_scaled / self.epsilon)
        loss_large = diff_scaled - self.C

        loss = torch.where(small, loss_small, loss_large)
        return loss.mean()


class QuadShapeLoss(nn.Module):
    """Geometry-aware quadrilateral shape regularizer.

    Encourages the predicted quadrilateral to maintain plausible rectangular geometry
    by penalizing:
    1. Diagonal length asymmetry (both diagonals of a rectangle should be equal)
    2. Non-convexity via cross-product sign consistency

    This prevents the model from predicting collapsed, crossed, or wildly
    non-rectangular quads that would break the downstream homography rectification.

    Args:
        weight_diag (float): Weight for diagonal symmetry term. Defaults to 1.0.
        weight_convex (float): Weight for convexity term. Defaults to 0.5.
    """
    def __init__(self, weight_diag: float = 1.0, weight_convex: float = 0.5) -> None:
        super().__init__()
        self.weight_diag = weight_diag
        self.weight_convex = weight_convex

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        """Compute shape regularization loss.

        Args:
            corners (torch.Tensor): Predicted corners [B, 4, 2] in order TL, TR, BR, BL.

        Returns:
            torch.Tensor: Scalar shape loss.
        """
        tl = corners[:, 0, :]  # [B, 2]
        tr = corners[:, 1, :]
        br = corners[:, 2, :]
        bl = corners[:, 3, :]

        # Diagonal symmetry: |diag1| should be close to |diag2|
        diag1 = torch.norm(br - tl, dim=-1)  # [B]
        diag2 = torch.norm(bl - tr, dim=-1)  # [B]
        diag_loss = torch.abs(diag1 - diag2).mean()

        # Convexity: all cross products of consecutive edges should have the same sign
        edges = [tr - tl, br - tr, bl - br, tl - bl]  # 4 edge vectors
        cross_products = []
        for i in range(4):
            e1 = edges[i]
            e2 = edges[(i + 1) % 4]
            cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # [B]
            cross_products.append(cross)

        cross_stack = torch.stack(cross_products, dim=-1)  # [B, 4]
        # Penalize if any cross product is negative (should all be positive for CCW order)
        convex_loss = torch.relu(-cross_stack).mean()

        return self.weight_diag * diag_loss + self.weight_convex * convex_loss
