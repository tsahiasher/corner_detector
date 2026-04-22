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


def save_refiner_global_debug(
    image_tensor: torch.Tensor,
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    heatmaps: torch.Tensor,
    img_path: str,
    output_dir: str
) -> None:
    """Saves a rich diagnostic image for Global Heatmap debugging.
    
    Shows:
    1. Main image with GT, Pred (SoftArgmax), and Raw Heatmap Peaks.
    2. The 4 individual heatmap channels.
    """
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

    # 2. Extract Raw Peaks from Heatmaps
    if heatmaps.ndim == 3:
        heatmaps = heatmaps.unsqueeze(0)
    B, C, H, W = heatmaps.shape
    heatmaps_sig = torch.sigmoid(heatmaps)
    
    # [C, 2] in [0, 1] space (Argmax peaks)
    peaks_flat = heatmaps.view(C, -1).argmax(dim=-1)
    peaks_y = (peaks_flat // W).float() / (H - 1)
    peaks_x = (peaks_flat % W).float() / (W - 1)
    peaks_np = torch.stack([peaks_x, peaks_y], dim=-1).cpu().numpy()

    # 3. Points for drawing
    pred_np = pred_corners.cpu().numpy() if isinstance(pred_corners, torch.Tensor) else np.array(pred_corners)
    tgt_np = target_corners.cpu().numpy() if isinstance(target_corners, torch.Tensor) else np.array(target_corners)
    
    if pred_np.ndim == 3: pred_np = pred_np[0]
    if tgt_np.ndim == 3: tgt_np = tgt_np[0]

    pred_px = (pred_np * [w, h]).astype(np.int32)
    tgt_px = (tgt_np * [w, h]).astype(np.int32)
    peaks_px = (peaks_np * [w, h]).astype(np.int32)

    # 4. Draw on Main Image
    # GT (Green)
    cv2.polylines(img, [tgt_px.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    # Pred SoftArgmax (Orange)
    cv2.polylines(img, [pred_px.reshape(-1, 1, 2)], True, (0, 165, 255), 2)
    
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)] # Blue, Yellow, Magenta, Red
    names = ["TL", "TR", "BR", "BL"]
    
    for i in range(4):
        # GT point
        cv2.circle(img, (int(tgt_px[i, 0]), int(tgt_px[i, 1])), 5, (0, 255, 0), -1)
        # Pred SoftArgmax point
        cv2.circle(img, (int(pred_px[i, 0]), int(pred_px[i, 1])), 5, (0, 165, 255), -1)
        # Raw Peak point (Cross)
        px, py = int(peaks_px[i, 0]), int(peaks_px[i, 1])
        cv2.line(img, (px - 8, py), (px + 8, py), colors[i], 2)
        cv2.line(img, (px, py - 8), (px, py + 8), colors[i], 2)
        
        cv2.putText(img, names[i], (px + 10, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    # 5. Create Heatmap Overlay Grid (2x2)
    # Each cell shows the image with a single heatmap channel overlaid
    vis_strips = []
    names = ["TL", "TR", "BR", "BL"]
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)] # Blue, Yellow, Magenta, Red
    
    for i in range(4):
        # Create base for this corner's visualization
        vis = img.copy()
        
        # Heatmap overlay
        hm = (heatmaps_sig[0, i].cpu().numpy() * 255).astype(np.uint8)
        hm_vis = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_vis = cv2.resize(hm_vis, (w, h))
        
        # Alpha blend (50% heatmap, 50% image)
        vis = cv2.addWeighted(vis, 0.5, hm_vis, 0.5, 0)
        
        # Draw GT, Pred, and Peak for this specific corner
        # GT (Green circle)
        cv2.circle(vis, (int(tgt_px[i, 0]), int(tgt_px[i, 1])), 6, (0, 255, 0), 2)
        # Pred SoftArgmax (Orange circle)
        cv2.circle(vis, (int(pred_px[i, 0]), int(pred_px[i, 1])), 6, (0, 165, 255), 2)
        # Raw Peak (Cross in specific color)
        px, py = int(peaks_px[i, 0]), int(peaks_px[i, 1])
        cv2.line(vis, (px - 10, py), (px + 10, py), colors[i], 2)
        cv2.line(vis, (px, py - 10), (px, py + 10), colors[i], 2)
        
        # Corner label
        cv2.putText(vis, f"Corner {i} ({names[i]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        vis_strips.append(vis)
        
    # Combine into 2x2 grid
    top = np.hstack([vis_strips[0], vis_strips[1]])
    bottom = np.hstack([vis_strips[3], vis_strips[2]]) # Clockwise
    grid = np.vstack([top, bottom])
    
    os.makedirs(output_dir, exist_ok=True)
    fname = "global_" + os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, fname), grid)

