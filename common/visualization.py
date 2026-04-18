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
    
    # Squeeze batch dimension if present (e.g. from [1, 4, 2] to [4, 2])
    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = pred_np[0]
    if tgt_np.ndim == 3 and tgt_np.shape[0] == 1:
        tgt_np = tgt_np[0]

    pred_px = (pred_np * [w, h]).astype(np.int32)
    tgt_px = (tgt_np * [w, h]).astype(np.int32)

    
    # Draw secondary (boundingbox) corners if provided
    if secondary_corners is not None:
        sec_np = secondary_corners.cpu().numpy() if isinstance(secondary_corners, torch.Tensor) else np.array(secondary_corners)
        sec_px = (sec_np * [w, h]).astype(np.int32)
        for px, py in sec_px:
            cv2.circle(img, (int(px), int(py)), 3, (255, 0, 0), -1) # Blue BoundingBox

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


def save_local_refining_debug(
    patches: torch.Tensor,
    heatmaps: torch.Tensor,
    img_path: str,
    output_dir: str
) -> None:
    """Saves a grid showing the 4 corner patches and their local heatmaps."""
    try:
        import cv2
    except ImportError:
        return

    # patches: [4, C, H, W], heatmaps: [4, 1, Hh, Wh]
    p_np = patches.detach().cpu().numpy()
    h_np = torch.sigmoid(heatmaps).detach().cpu().numpy()
    
    B, C, H, W = p_np.shape
    _, _, Hh, Wh = h_np.shape
    
    vis_patches = []
    for i in range(4):
        # 1. Patch Visualization
        p = p_np[i] # [C, H, W]
        if C == 3:
            p = p.transpose(1, 2, 0)
            p = (np.clip(p, 0, 1) * 255).astype(np.uint8)
            p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        else:
            # For 64-channel features, visualize mean energy
            p = np.mean(p, axis=0) # [H, W]
            p = (np.clip(p, 0, 1) * 255).astype(np.uint8)
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
            
        p = cv2.resize(p, (128, 128), interpolation=cv2.INTER_NEAREST)

        
        # 2. Heatmap overlay
        h = h_np[i, 0]
        h_vis = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
        h_vis = cv2.resize(h_vis, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Add labels
        label = f"Corner {i}"
        cv2.putText(p, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combine patch and heatmap
        vis_patches.append(np.hstack([p, h_vis]))

    # Stack into 2x2 grid
    top = np.hstack([vis_patches[0], vis_patches[1]])
    bottom = np.hstack([vis_patches[3], vis_patches[2]]) # TL, TR, BL, BR sequence matches clockwise-ish if needed
    grid = np.vstack([top, bottom])
    
    os.makedirs(output_dir, exist_ok=True)
    fname = "refine_" + os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, fname), grid)

