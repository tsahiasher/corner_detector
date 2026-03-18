import torch
from typing import Tuple, Dict

def compute_pixel_error(pred_corners: torch.Tensor, target_corners: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Computes the Euclidean pixel error distance per corner vertex.
    
    Args:
        pred_corners (torch.Tensor): Predicted normalized corners [B, 4, 2].
        target_corners (torch.Tensor): Target mapped normalized corners [B, 4, 2].
        width (float): Original image coordinate width.
        height (float): Original image coordinate height.
        
    Returns:
        torch.Tensor: Evaluated Euclidean distances in pixels of shape [B, 4].
    """
    scale = torch.tensor([width, height], dtype=torch.float32, device=pred_corners.device)
    pred_px = pred_corners * scale
    target_px = target_corners * scale
    
    diff = pred_px - target_px
    dist = torch.norm(diff, dim=-1) # [B, 4]
    return dist

def calculate_accuracy_metrics(errors: torch.Tensor, thresholds: Tuple[int, ...] = (2, 3, 5, 10)) -> Dict[str, float]:
    """Calculates evaluation summary accuracy metrics from pixel distances.
    
    Args:
        errors (torch.Tensor): Tensor of scalar distance errors. Shape expected [B, 4].
        thresholds (Tuple[int, ...], optional): Distance pass-thresholds in pixels. Defaults to (2, 3, 5, 10).
        
    Returns:
        Dict[str, float]: Performance summary statistics including mean, median, per-corner, and threshold coverage rates.
    """
    if errors.numel() == 0:
        return {}
        
    flat_errors = errors.reshape(-1)
    
    metrics = {
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
