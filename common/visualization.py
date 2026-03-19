import os
import time
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def draw_corners_on_image(image_tensor: torch.Tensor, corners_norm: torch.Tensor) -> np.ndarray:
    """Draws predicted normalized corners with bounding polygon edges onto the standard PyTorch tensor image.

    Args:
        image_tensor (torch.Tensor): A [3, H, W] normalized image tensor fed to network.
        corners_norm (torch.Tensor): A [4, 2] tensor of network predicted coordinate locations mapped to [0,1].

    Returns:
        np.ndarray: An RGB HxWxC np.uint8 matrix with drawn identifier geometry overlays.

    Raises:
        ImportError: Escaped if OpenCV dependency is missing.
    """
    try:
        import cv2
    except ImportError as e:
        logger.error("cv2 module not found, skipping visualization generation.")
        raise e

    from common.transforms import denormalize_image

    # Denormalize map to visualization [0,1] plane.
    img = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    # Translate float map to internal C byte format
    img = (img * 255).astype(np.uint8)
    # Allocate BGR channel frame
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    corners_px = (corners_norm.cpu().numpy() * [w, h]).astype(np.int32)

    # Render corners and explicit connected topology polygon
    for i in range(4):
        cv2.circle(img, tuple(corners_px[i]), 5, (0, 0, 255), -1)
        next_i = (i + 1) % 4
        cv2.line(img, tuple(corners_px[i]), tuple(corners_px[next_i]), (0, 255, 0), 2)

    # Flush to standard RGB buffer representation.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_indexed_corners(
    image_tensor: torch.Tensor,
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    orig_width: float,
    orig_height: float,
) -> np.ndarray:
    """Draws predicted and target corners with index labels on the image.

    Predicted corners are drawn in Orange with their index (0-3).
    Target corners are drawn in Green with their index (0-3).
    Each set is connected by a colored polygon outline.

    Coordinates are mapped through (orig_width, orig_height) into the display
    image's pixel grid so the labels land in the correct place even when the
    original image was non-square.

    Args:
        image_tensor (torch.Tensor): A [3, H, W] normalized image tensor.
        pred_corners (torch.Tensor): A [4, 2] tensor of predicted normalized coords.
        target_corners (torch.Tensor): A [4, 2] tensor of target normalized coords.
        orig_width (float): Original image width in pixels.
        orig_height (float): Original image height in pixels.

    Returns:
        np.ndarray: An RGB HxWxC np.uint8 image with indexed visualizations.
    """
    try:
        import cv2
    except ImportError as e:
        logger.error("cv2 module not found, skipping visualization generation.")
        raise e

    from common.transforms import denormalize_image

    # Denormalize and convert to uint8 BGR
    img = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    disp_h, disp_w, _ = img.shape

    # Scale factors: normalized -> orig pixels -> display pixels
    sx = disp_w / orig_width * orig_width   # = disp_w (since norm * orig = px, then px * disp/orig = disp)
    sy = disp_h / orig_height * orig_height  # = disp_h

    # Actually it simplifies: norm * disp_size = display pixels
    pred_px = (pred_corners.cpu().numpy() * [disp_w, disp_h]).astype(np.int32)
    tgt_px = (target_corners.cpu().numpy() * [disp_w, disp_h]).astype(np.int32)

    # Colors in BGR: Orange for predictions, Green for targets
    color_pred = (0, 165, 255)   # Orange
    color_tgt = (0, 200, 0)      # Green
    # Semantic order from YOLO annotations: TL, TR, BR, BL
    label_names = ['TL', 'TR', 'BR', 'BL']

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    # Draw target polygon + labels (draw first so predictions overlay on top)
    for i in range(4):
        pt = tuple(tgt_px[i])
        next_pt = tuple(tgt_px[(i + 1) % 4])
        cv2.line(img, pt, next_pt, color_tgt, 2)
        cv2.circle(img, pt, 6, color_tgt, -1)
        label = f"{i}:{label_names[i]}"
        # Offset text slightly to avoid overlap with circle
        cv2.putText(img, label, (pt[0] + 8, pt[1] - 8), font, font_scale, color_tgt, thickness)

    # Draw predicted polygon + labels
    for i in range(4):
        pt = tuple(pred_px[i])
        next_pt = tuple(pred_px[(i + 1) % 4])
        cv2.line(img, pt, next_pt, color_pred, 2)
        cv2.circle(img, pt, 6, color_pred, -1)
        label = f"{i}:{label_names[i]}"
        cv2.putText(img, label, (pt[0] + 8, pt[1] + 18), font, font_scale, color_pred, thickness)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_indexed_visualization(
    image_tensor: torch.Tensor,
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    orig_width: float,
    orig_height: float,
    save_path: str,
    img_path: Optional[str] = None,
) -> None:
    """Draws indexed corners and saves the result to disk.

    Args:
        image_tensor (torch.Tensor): A [3, H, W] normalized image tensor.
        pred_corners (torch.Tensor): A [4, 2] tensor of predicted normalized coords.
        target_corners (torch.Tensor): A [4, 2] tensor of target normalized coords.
        orig_width (float): Original image width in pixels.
        orig_height (float): Original image height in pixels.
        save_path (str): Output file path (e.g. .jpg or .png).
        img_path (Optional[str]): Original image path for logging.
    """
    try:
        import cv2
    except ImportError:
        logger.error("cv2 module not found, skipping visualization save.")
        return

    vis = draw_indexed_corners(image_tensor, pred_corners, target_corners, orig_width, orig_height)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert RGB back to BGR for cv2 save
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    logger.debug(f"Saved visualization for {src} -> {save_path}")


def draw_quadrilateral(img: np.ndarray, corners_px: np.ndarray, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draws a closed quadrilateral on a BGR image."""
    try:
        import cv2
    except ImportError:
        return img
    for i in range(4):
        pt1 = tuple(corners_px[i].astype(int))
        pt2 = tuple(corners_px[(i+1)%4].astype(int))
        cv2.line(img, pt1, pt2, color, thickness)
    return img


def save_diagnostic_visualization(
    image_tensor: torch.Tensor,
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    edge_tensor: Optional[torch.Tensor],
    img_path: str,
    output_dir: str
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
