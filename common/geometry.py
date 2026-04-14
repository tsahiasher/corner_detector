import torch
import numpy as np
from typing import Tuple, List

def get_visual_orientation(keypoints: List[List[float]]) -> int:
    """Derives the visual rotation class (0=0°, 1=90°, 2=180°, 3=270°) of the card 
    based on the geometric quadrant of the Physical Top-Left corner (index 0).
    """
    if len(keypoints) < 4:
        return 0
        
    p0 = keypoints[0]
    cx = sum(p[0] for p in keypoints) / 4.0
    cy = sum(p[1] for p in keypoints) / 4.0
    
    dx = p0[0] - cx
    dy = p0[1] - cy
    
    if dx <= 0 and dy <= 0:
        return 0  # Physical TL is at Visual TL (0 deg)
    elif dx > 0 and dy <= 0:
        return 1  # Physical TL is at Visual TR (90 deg CW)
    elif dx > 0 and dy > 0:
        return 2  # Physical TL is at Visual BR (180 deg)
    else:
        return 3  # Physical TL is at Visual BL (270 deg CW)

def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Computes a 3x3 perspective transform matrix from 4 source points to 4 destination points.

    Args:
        src_pts (np.ndarray): Source quadrilateral corners, shape [4, 2], float32.
        dst_pts (np.ndarray): Destination quadrilateral corners, shape [4, 2], float32.

    Returns:
        np.ndarray: 3x3 homography matrix, float64.

    Raises:
        ImportError: If OpenCV is not available.
        ValueError: If points are not the correct shape.
    """
    import cv2

    src = np.asarray(src_pts, dtype=np.float32).reshape(4, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(4, 2)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(f"Expected 4 points each, got src={src.shape}, dst={dst.shape}")

    H = cv2.getPerspectiveTransform(src, dst)
    return H


def warp_image(image: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Warps an image using a 3x3 homography matrix.

    Args:
        image (np.ndarray): Input image, HxWxC or HxW, uint8 or float.
        H (np.ndarray): 3x3 perspective transform matrix.
        output_size (Tuple[int, int]): Output (width, height).

    Returns:
        np.ndarray: Warped image with the requested output size.
    """
    import cv2

    return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sort_corners_clockwise(corners: torch.Tensor) -> torch.Tensor:
    """Semantically sorts a batch of 4 corners in clockwise order starting from Visual Top-Left.
    
    Uses atan2(dy, dx) relative to the centroid. To ensure stability for axis-aligned cards
    where multiple corners might have identical angles (e.g. perfect vertical/horizontal), 
    a tiny spatial tiebreaker is added to the angles.

    Args:
        corners (torch.Tensor): Tensor of shape [B, 4, 2] or [4, 2].

    Returns:
        torch.Tensor: Sorted corners in the same shape as input.
    """
    is_single = corners.ndim == 2
    if is_single:
        corners = corners.unsqueeze(0)
    
    B = corners.size(0)
    device = corners.device
    
    # 1. Compute Centroid
    centroid = corners.mean(dim=1, keepdim=True) # [B, 1, 2]
    diffs = corners - centroid # [B, 4, 2]
    
    # 2. Compute Angles
    # atan2 returns values in [-pi, pi]
    angles = torch.atan2(diffs[:, :, 1], diffs[:, :, 0]) # [B, 4]
    
    # Stability Tiebreaker (v8.1): 
    # Add a tiny bias (1e-4 * dx) to the angle to ensure deterministic sorting 
    # even when corners are perfectly spatially aligned on a grid.
    angles = angles + diffs[:, :, 0] * 1e-4
    
    # 3. Sort
    sort_idx = torch.argsort(angles, dim=1) # [B, 4]
    
    # 4. Gather
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, 4)
    sorted_corners = corners[b_idx, sort_idx]
    
    if is_single:
        return sorted_corners.squeeze(0)
    return sorted_corners
