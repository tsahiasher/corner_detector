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

def normalize_corners(corners: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Normalizes absolute image coordinate corners to [0, 1].

    Args:
        corners (torch.Tensor): Coordinates of shape [..., 2].
        width (int): Image spatial width.
        height (int): Image spatial height.

    Returns:
        torch.Tensor: Normalized floating point coordinates.
    """
    norm = torch.tensor([width, height], dtype=torch.float32, device=corners.device)
    return corners / norm


def denormalize_corners(corners: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Denormalizes [0, 1] relative coordinates back to absolute image pixel scale.

    Args:
        corners (torch.Tensor): Normalized coordinates [..., 2].
        width (int): Target image width.
        height (int): Target image height.

    Returns:
        torch.Tensor: Absolute scale pixel coordinates.
    """
    norm = torch.tensor([width, height], dtype=torch.float32, device=corners.device)
    return corners * norm


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


def crop_corner_patches(rectified_img: np.ndarray, canonical_size: Tuple[int, int],
                        patch_size: int = 96) -> List[np.ndarray]:
    """Extracts 4 square patches from the corners of a rectified card image.

    The patches are centered on each corner of the canonical card. For corners
    near the image boundary, the patch center is clamped so the patch stays
    within the image bounds.

    Corner order: top-left, top-right, bottom-right, bottom-left.

    Args:
        rectified_img (np.ndarray): Rectified card image, shape [H, W, C].
        canonical_size (Tuple[int, int]): (width, height) of the canonical card.
        patch_size (int, optional): Side length of the square patch. Defaults to 96.

    Returns:
        List[np.ndarray]: Four patches, each of shape [patch_size, patch_size, C].
    """
    cw, ch = canonical_size
    half = patch_size // 2
    img_h, img_w = rectified_img.shape[:2]

    # Corner centers in the canonical card coordinate system
    corner_centers = [
        (0, 0),          # top-left
        (cw, 0),         # top-right
        (cw, ch),        # bottom-right
        (0, ch),         # bottom-left
    ]

    patches = []
    for cx, cy in corner_centers:
        # Clamp so the patch stays within the warped image bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        # If we hit the right/bottom edge, shift back
        if x2 > img_w:
            x2 = img_w
            x1 = max(0, x2 - patch_size)
        if y2 > img_h:
            y2 = img_h
            y1 = max(0, y2 - patch_size)

        patch = rectified_img[y1:y2, x1:x2]

        # Pad if patch is still too small (edge case for very small images)
        ph, pw = patch.shape[:2]
        if ph < patch_size or pw < patch_size:
            if rectified_img.ndim == 3:
                padded = np.zeros((patch_size, patch_size, rectified_img.shape[2]), dtype=rectified_img.dtype)
            else:
                padded = np.zeros((patch_size, patch_size), dtype=rectified_img.dtype)
            padded[:ph, :pw] = patch
            patch = padded

        patches.append(patch)

    return patches


def get_patch_origin(corner_idx: int, canonical_size: Tuple[int, int],
                     patch_size: int = 96) -> Tuple[int, int]:
    """Returns the top-left pixel coordinate of the patch for a given corner index.

    This must be perfectly consistent with crop_corner_patches so that
    patch-local coordinates can be mapped back to canonical-space coordinates.

    Args:
        corner_idx (int): Index 0-3 (TL, TR, BR, BL).
        canonical_size (Tuple[int, int]): (width, height) of the canonical card.
        patch_size (int, optional): Patch side length. Defaults to 96.

    Returns:
        Tuple[int, int]: (x_origin, y_origin) in canonical image coordinates.
    """
    cw, ch = canonical_size
    half = patch_size // 2

    corners = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    cx, cy = corners[corner_idx]

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x2 > cw:
        x1 = max(0, cw - patch_size)
    if y2 > ch:
        y1 = max(0, ch - patch_size)

    return (x1, y1)


def backproject_corners(refined_pts_canonical: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Maps refined canonical-space corner coordinates back to original image coordinates.

    Args:
        refined_pts_canonical (np.ndarray): Refined corner positions in canonical space, shape [N, 2].
        H_inv (np.ndarray): Inverse homography (canonical → original), 3x3.

    Returns:
        np.ndarray: Corner positions in original image coordinates, shape [N, 2].
    """
    import cv2

    pts = np.asarray(refined_pts_canonical, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H_inv)
    return transformed.reshape(-1, 2)
