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


class KeyPointLoss(nn.Module):
    """KeyPoints loss with center-radius multi-scale positive assignment.

    Five loss terms are computed across all prediction scales:

    1. **Objectness** - BCE (all cells, sum / num_pos for balance).
    2. **Bounding box** - CIoU (positive cells only).
    3. **Keypoints** - L1 on cell-relative decoded normalised coordinates
       (positive cells only).
    4. **Keypoint confidence** - BCE on ALL cells (pos target=1, neg
       target=0) so that non-responsible cells predict low confidence
       and joint ``obj x mean(kpt_conf)`` scoring works at inference.
    5. **Box-corner consistency** - (1 - IoU) between the predicted box
       and the box derived from predicted corners (positive cells only).
       Forces box and corner heads to agree geometrically.

    Positive assignment: all grid cells within ``center_radius`` of the
    GT centre are labelled positive.

    Args:
        w_obj:  Weight for objectness loss.
        w_box:  Weight for CIoU bounding-box loss.
        w_kpt:  Weight for keypoint L1 loss.
        w_kpt_conf: Weight for keypoint confidence BCE loss.
        w_consist: Weight for box-corner consistency loss.
        center_radius: Radius (in grid cells) for positive assignment.
        num_kpt: Number of keypoints (4 corners).
    """

    def __init__(
        self,
        w_obj: float = 1.0,
        w_box: float = 7.5,
        w_kpt: float = 12.0,
        w_kpt_conf: float = 1.0,
        w_consist: float = 1.0,
        center_radius: float = 2.5,
        num_kpt: int = 4,
    ) -> None:
        super().__init__()
        self.w_obj = w_obj
        self.w_box = w_box
        self.w_kpt = w_kpt
        self.w_kpt_conf = w_kpt_conf
        self.w_consist = w_consist
        self.center_radius = center_radius
        self.num_kpt = num_kpt

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ciou_elementwise(
        pcx: torch.Tensor, pcy: torch.Tensor,
        pw: torch.Tensor, ph: torch.Tensor,
        gcx: torch.Tensor, gcy: torch.Tensor,
        gw: torch.Tensor, gh: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised per-element CIoU loss  →  ``(1 − CIoU)``."""
        eps = 1e-7
        px1, py1 = pcx - pw / 2, pcy - ph / 2
        px2, py2 = pcx + pw / 2, pcy + ph / 2
        gx1, gy1 = gcx - gw / 2, gcy - gh / 2
        gx2, gy2 = gcx + gw / 2, gcy + gh / 2

        ix1 = torch.max(px1, gx1)
        iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2)
        iy2 = torch.min(py2, gy2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        union = pw * ph + gw * gh - inter + eps
        iou = inter / union

        ex1 = torch.min(px1, gx1)
        ey1 = torch.min(py1, gy1)
        ex2 = torch.max(px2, gx2)
        ey2 = torch.max(py2, gy2)
        c_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps
        rho_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
        ) ** 2
        with torch.no_grad():
            alpha = v / ((1.0 - iou) + v + eps)

        return 1.0 - (iou - rho_sq / c_sq - alpha * v)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        raw_preds: List[torch.Tensor],
        gt_corners: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-scale KeyPoints loss.

        Args:
            raw_preds:  List of ``[B, 17, Hi, Wi]`` raw predictions
                        (one per FPN scale).
            gt_corners: ``[B, 4, 2]`` normalised GT corner coordinates.

        Returns:
            Dict with scalar tensors:
            ``total``, ``obj``, ``box``, ``kpt``, ``kpt_conf``, ``consist``.
        """
        device = raw_preds[0].device
        B = raw_preds[0].size(0)

        # GT axis-aligned bounding box derived from corners
        gx_min = gt_corners[:, :, 0].min(1).values
        gy_min = gt_corners[:, :, 1].min(1).values
        gx_max = gt_corners[:, :, 0].max(1).values
        gy_max = gt_corners[:, :, 1].max(1).values
        gt_cx = (gx_min + gx_max) / 2
        gt_cy = (gy_min + gy_max) / 2
        gt_w = (gx_max - gx_min).clamp(min=1e-6)
        gt_h = (gy_max - gy_min).clamp(min=1e-6)

        sum_obj = torch.zeros(1, device=device)
        sum_box = torch.zeros(1, device=device)
        sum_kpt = torch.zeros(1, device=device)
        sum_kpt_conf = torch.zeros(1, device=device)
        sum_consist = torch.zeros(1, device=device)
        total_num_pos = 0

        for raw in raw_preds:
            _, _, H, W = raw.shape

            gy_grid, gx_grid = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij',
            )

            gt_gj = gt_cx * W
            gt_gi = gt_cy * H

            pos_mask = torch.zeros(B, H, W, device=device, dtype=torch.bool)
            for b in range(B):
                dist = torch.sqrt(
                    (gx_grid - gt_gj[b]) ** 2 + (gy_grid - gt_gi[b]) ** 2
                )
                pos_mask[b] = dist <= self.center_radius

            num_pos = max(pos_mask.sum().item(), 1)
            total_num_pos += num_pos

            # ---- Decode predictions (full grid) ----
            dec_cx = (torch.sigmoid(raw[:, 0]) * 2.0 - 0.5 + gx_grid) / W
            dec_cy = (torch.sigmoid(raw[:, 1]) * 2.0 - 0.5 + gy_grid) / H
            dec_bw = torch.sigmoid(raw[:, 2])
            dec_bh = torch.sigmoid(raw[:, 3])

            # ---- 1. Objectness BCE (all cells, normalised by #pos) ----
            obj_loss = F.binary_cross_entropy_with_logits(
                raw[:, 4], pos_mask.float(), reduction='sum',
            ) / num_pos
            sum_obj = sum_obj + obj_loss

            # ---- 2. Box CIoU (positives only) ----
            gcx_e = gt_cx.view(B, 1, 1).expand_as(dec_cx)[pos_mask]
            gcy_e = gt_cy.view(B, 1, 1).expand_as(dec_cy)[pos_mask]
            gw_e = gt_w.view(B, 1, 1).expand_as(dec_bw)[pos_mask]
            gh_e = gt_h.view(B, 1, 1).expand_as(dec_bh)[pos_mask]

            box_loss = self._ciou_elementwise(
                dec_cx[pos_mask], dec_cy[pos_mask],
                dec_bw[pos_mask], dec_bh[pos_mask],
                gcx_e, gcy_e, gw_e, gh_e,
            ).sum() / num_pos
            sum_box = sum_box + box_loss

            # ---- 3. Keypoint L1 (positives only) ----
            scale_kpt_l1 = torch.zeros(1, device=device)

            for k in range(self.num_kpt):
                # Decode keypoint (cell-relative offset)
                dec_kx = (raw[:, 5 + k * 3] * 2.0 + gx_grid) / W
                dec_ky = (raw[:, 5 + k * 3 + 1] * 2.0 + gy_grid) / H

                gt_kx = gt_corners[:, k, 0].view(B, 1, 1).expand(B, H, W)
                gt_ky = gt_corners[:, k, 1].view(B, 1, 1).expand(B, H, W)

                # L1 on normalised coordinates at positive cells
                scale_kpt_l1 = scale_kpt_l1 + (
                    torch.abs(dec_kx[pos_mask] - gt_kx[pos_mask])
                    + torch.abs(dec_ky[pos_mask] - gt_ky[pos_mask])
                ).sum()

            sum_kpt = sum_kpt + scale_kpt_l1 / (num_pos * self.num_kpt)

            # ---- 4. Keypoint confidence BCE (ALL cells) ----
            kpt_conf_target = pos_mask.float()  # [B, H, W]

            scale_kpt_conf = torch.zeros(1, device=device)
            for k in range(self.num_kpt):
                conf_logit = raw[:, 5 + k * 3 + 2]
                scale_kpt_conf = scale_kpt_conf + (
                    F.binary_cross_entropy_with_logits(
                        conf_logit, kpt_conf_target, reduction='sum',
                    )
                )
            scale_kpt_conf = scale_kpt_conf / (B * H * W * self.num_kpt)
            sum_kpt_conf = sum_kpt_conf + scale_kpt_conf

            # ---- 5. Box-corner consistency (positives only) ----
            # Derive an axis-aligned box from the decoded corners and
            # compare to the predicted box via IoU. Loss = 1 - IoU.
            all_dec_kx = []
            all_dec_ky = []
            for k in range(self.num_kpt):
                all_dec_kx.append((raw[:, 5 + k * 3] * 2.0 + gx_grid) / W)
                all_dec_ky.append((raw[:, 5 + k * 3 + 1] * 2.0 + gy_grid) / H)
            # Stack to [B, num_kpt, H, W]
            kpt_xs = torch.stack(all_dec_kx, dim=1)
            kpt_ys = torch.stack(all_dec_ky, dim=1)

            # Corner-derived box (at positive cells)
            kpt_x_min = kpt_xs.min(dim=1).values[pos_mask]  # [P]
            kpt_x_max = kpt_xs.max(dim=1).values[pos_mask]
            kpt_y_min = kpt_ys.min(dim=1).values[pos_mask]
            kpt_y_max = kpt_ys.max(dim=1).values[pos_mask]

            # Predicted box (at positive cells)
            pred_x1 = (dec_cx - dec_bw / 2)[pos_mask]
            pred_y1 = (dec_cy - dec_bh / 2)[pos_mask]
            pred_x2 = (dec_cx + dec_bw / 2)[pos_mask]
            pred_y2 = (dec_cy + dec_bh / 2)[pos_mask]

            # IoU between corner-derived box and predicted box
            eps = 1e-7
            ix1 = torch.max(kpt_x_min, pred_x1)
            iy1 = torch.max(kpt_y_min, pred_y1)
            ix2 = torch.min(kpt_x_max, pred_x2)
            iy2 = torch.min(kpt_y_max, pred_y2)
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            area_kpt = (kpt_x_max - kpt_x_min).clamp(min=eps) * (kpt_y_max - kpt_y_min).clamp(min=eps)
            area_box = (pred_x2 - pred_x1).clamp(min=eps) * (pred_y2 - pred_y1).clamp(min=eps)
            iou = inter / (area_kpt + area_box - inter + eps)

            consist_loss = (1.0 - iou).sum() / num_pos
            sum_consist = sum_consist + consist_loss

        # Average across scales
        ns = float(len(raw_preds))
        avg_obj = sum_obj / ns
        avg_box = sum_box / ns
        avg_kpt = sum_kpt / ns
        avg_kpt_conf = sum_kpt_conf / ns
        avg_consist = sum_consist / ns

        total = (
            avg_obj * self.w_obj
            + avg_box * self.w_box
            + avg_kpt * self.w_kpt
            + avg_kpt_conf * self.w_kpt_conf
            + avg_consist * self.w_consist
        )

        return {
            'total': total,
            'obj': avg_obj.detach(),
            'box': avg_box.detach(),
            'kpt': avg_kpt.detach(),
            'kpt_conf': avg_kpt_conf.detach(),
            'consist': avg_consist.detach(),
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


class HeatmapLoss(nn.Module):
    """Gaussian heatmap loss using BCE with sum reduction for better gradient flow.
    """
    def __init__(self, sigma: float = 4.0, pos_weight: float = 200.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.pos_weight = pos_weight

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
        # Map [0, 1] normalized to [0, W-1] pixel indices
        gt_px = gt_corners.view(B, C, 1, 1, 2) * torch.tensor([W - 1, H - 1], device=device, dtype=torch.float32).view(1, 1, 1, 1, 2)
        gt_px_int = torch.round(gt_px)
        # Avoid in-place assignments to prevent autograd errors
        max_bounds = torch.tensor([W - 1, H - 1], device=device, dtype=torch.float32).view(1, 1, 1, 1, 2)
        gt_px_int = torch.clamp(gt_px_int, torch.zeros_like(max_bounds), max_bounds)
        
        dist_sq = torch.sum((grid.view(1, 1, H, W, 2) - gt_px_int) ** 2, dim=-1)
        target = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        # 2. Weighted BCE Loss (on raw logits)
        # Using reduction='sum' and dividing by (B*C) ensures the gradients 
        # are strong enough to force peak formation.
        loss = F.binary_cross_entropy_with_logits(
            pred_heatmaps, target, 
            pos_weight=torch.tensor([self.pos_weight], device=device),
            reduction='sum'
        )
        
        return loss / (B * C)
