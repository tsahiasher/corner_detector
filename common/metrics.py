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


class YOLOPoseLoss(nn.Module):
    """Combined YOLO-Pose loss for single-object keypoint detection.

    Computes three loss terms, all supervised at the ground-truth centre cell:

    1. **Objectness** – CenterNet focal loss on a Gaussian heatmap target.
    2. **Bounding box** – Complete-IoU (CIoU) loss on axis-aligned bbox.
    3. **Keypoints** – Smooth-L1 on absolute normalised corner positions.

    Args:
        w_obj:  Weight for objectness focal loss.
        w_box:  Weight for CIoU bounding-box loss.
        w_kpt:  Weight for keypoint regression loss.
        sigma:  Gaussian spread for the objectness heatmap target.
    """

    def __init__(self, w_obj: float = 1.0, w_box: float = 5.0,
                 w_kpt: float = 10.0, sigma: float = 2.0) -> None:
        super().__init__()
        self.w_obj = w_obj
        self.w_box = w_box
        self.w_kpt = w_kpt
        self.obj_loss_fn = HeatmapFocalLoss(sigma=sigma)

    @staticmethod
    def _ciou(pcx: torch.Tensor, pcy: torch.Tensor,
              pw: torch.Tensor, ph: torch.Tensor,
              gcx: torch.Tensor, gcy: torch.Tensor,
              gw: torch.Tensor, gh: torch.Tensor) -> torch.Tensor:
        """Complete-IoU loss between predicted and GT boxes (normalised coords).

        Args:
            pcx, pcy, pw, ph: Predicted centre-x, centre-y, width, height.
            gcx, gcy, gw, gh: Ground-truth centre-x, centre-y, width, height.

        Returns:
            Scalar CIoU loss (1 − CIoU), averaged over batch.
        """
        eps = 1e-7

        # Convert to xyxy
        px1, py1 = pcx - pw / 2, pcy - ph / 2
        px2, py2 = pcx + pw / 2, pcy + ph / 2
        gx1, gy1 = gcx - gw / 2, gcy - gh / 2
        gx2, gy2 = gcx + gw / 2, gcy + gh / 2

        # Intersection
        ix1 = torch.max(px1, gx1)
        iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2)
        iy2 = torch.min(py2, gy2)
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

        # Union
        union = pw * ph + gw * gh - inter + eps
        iou = inter / union

        # Smallest enclosing box diagonal²
        ex1 = torch.min(px1, gx1)
        ey1 = torch.min(py1, gy1)
        ex2 = torch.max(px2, gx2)
        ey2 = torch.max(py2, gy2)
        c_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps

        # Centre distance²
        rho_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

        # Aspect-ratio consistency penalty
        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
        ) ** 2
        with torch.no_grad():
            alpha = v / ((1.0 - iou) + v + eps)

        ciou = iou - rho_sq / c_sq - alpha * v
        return (1.0 - ciou).mean()

    def forward(self, raw_pred: torch.Tensor,
                gt_corners: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the combined YOLO-Pose loss.

        Args:
            raw_pred:   Raw model grid output ``[B, 13, Hg, Wg]``.
            gt_corners: Ground-truth corner keypoints ``[B, 4, 2]``,
                        normalised to ``[0, 1]``.

        Returns:
            Dict with scalar tensors: ``total``, ``obj``, ``box``, ``kpt``.
        """
        B, _, Hg, Wg = raw_pred.shape
        device = raw_pred.device

        # ---- Ground-truth derived quantities ----
        gt_ctr = gt_corners.mean(dim=1, keepdim=True)  # [B, 1, 2]

        gx_min = gt_corners[:, :, 0].min(1).values
        gy_min = gt_corners[:, :, 1].min(1).values
        gx_max = gt_corners[:, :, 0].max(1).values
        gy_max = gt_corners[:, :, 1].max(1).values
        gcx = (gx_min + gx_max) / 2
        gcy = (gy_min + gy_max) / 2
        gw = (gx_max - gx_min).clamp(min=1e-6)
        gh = (gy_max - gy_min).clamp(min=1e-6)

        # Positive cell indices
        gj = torch.clamp((gt_ctr[:, 0, 0] * Wg).long(), 0, Wg - 1)
        gi = torch.clamp((gt_ctr[:, 0, 1] * Hg).long(), 0, Hg - 1)
        bi = torch.arange(B, device=device)

        # ---- 1. Objectness (Focal Heatmap) ----
        loss_obj = self.obj_loss_fn(raw_pred[:, 0:1], gt_ctr)

        # ---- 2. Bounding-box CIoU (at GT cell) ----
        tx = raw_pred[bi, 1, gi, gj]
        ty = raw_pred[bi, 2, gi, gj]
        tw = raw_pred[bi, 3, gi, gj]
        th = raw_pred[bi, 4, gi, gj]

        pcx = (torch.sigmoid(tx) + gj.float()) / Wg
        pcy = (torch.sigmoid(ty) + gi.float()) / Hg
        pw = torch.sigmoid(tw)
        ph = torch.sigmoid(th)

        loss_box = self._ciou(pcx, pcy, pw, ph, gcx, gcy, gw, gh)

        # ---- 3. Keypoint Smooth-L1 (at GT cell) ----
        kpt_raw = raw_pred[bi, 5:13, gi, gj]   # [B, 8]
        pred_kpt = torch.sigmoid(kpt_raw).view(B, 4, 2)
        loss_kpt = F.smooth_l1_loss(pred_kpt, gt_corners)

        # ---- Total ----
        total = (loss_obj * self.w_obj
                 + loss_box * self.w_box
                 + loss_kpt * self.w_kpt)

        return {
            'total': total,
            'obj':   loss_obj,
            'box':   loss_box,
            'kpt':   loss_kpt,
        }


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
