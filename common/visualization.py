import numpy as np
import torch
import logging

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
