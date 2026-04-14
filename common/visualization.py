import os
import time
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def save_diagnostic_visualization(
    image_tensor: torch.Tensor,
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    edge_tensor: Optional[torch.Tensor],
    img_path: str,
    output_dir: str,
    secondary_corners: Optional[torch.Tensor] = None
) -> None:
    """Saves a rich diagnostic image for multi-stage debugging."""
    try:
        import cv2
    except ImportError:
        return

    from common.transforms import denormalize_image
    
    # 1. Base image (BGR)
    img = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # 2. Draw points/quads
    pred_np = pred_corners.cpu().numpy() if isinstance(pred_corners, torch.Tensor) else np.array(pred_corners)
    tgt_np = target_corners.cpu().numpy() if isinstance(target_corners, torch.Tensor) else np.array(target_corners)
    
    pred_px = (pred_np * [w, h]).astype(np.int32)
    tgt_px = (tgt_np * [w, h]).astype(np.int32)
    
    # Draw secondary (coarse) corners if provided
    if secondary_corners is not None:
        sec_np = secondary_corners.cpu().numpy() if isinstance(secondary_corners, torch.Tensor) else np.array(secondary_corners)
        sec_px = (sec_np * [w, h]).astype(np.int32)
        for px, py in sec_px:
            cv2.circle(img, (int(px), int(py)), 3, (255, 0, 0), -1) # Blue Coarse

    # If exactly 4 corners, draw a closed quad
    if len(tgt_px) == 4:
        cv2.polylines(img, [tgt_px.reshape(-1, 1, 2)], True, (0, 255, 0), 2)  # Green GT
        cv2.polylines(img, [pred_px.reshape(-1, 1, 2)], True, (0, 165, 255), 2)  # Orange Pred
    
    for i, (px, py) in enumerate(pred_px):
        cv2.circle(img, (int(px), int(py)), 4, (0, 0, 255), -1)
        # Pred label
        cv2.putText(img, str(i), (int(px) + 5, int(py) + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # GT circle
        tpx, tpy = tgt_px[i]
        cv2.circle(img, (int(tpx), int(tpy)), 4, (0, 255, 0), 1)
        # GT label (optional, but helps if they differ much)
        cv2.putText(img, str(i), (int(tpx) - 12, int(tpy) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 3. Mask and Edges overlays (Optional)
    strips = [img]
    if mask_tensor is not None:
        mask = (mask_tensor.cpu().numpy()[0] * 255).astype(np.uint8)
        mask_vis = cv2.resize(cv2.applyColorMap(mask, cv2.COLORMAP_JET), (w, h))
        strips.append(mask_vis)
    
    if edge_tensor is not None:
        edges = (edge_tensor.cpu().numpy()[0] * 255).astype(np.uint8)
        edge_vis = cv2.resize(cv2.applyColorMap(edges, cv2.COLORMAP_HOT), (w, h))
        strips.append(edge_vis)

    # Combine into a horizontal strip
    combined = np.hstack(strips)
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, fname), combined)
