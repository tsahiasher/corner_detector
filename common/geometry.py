import torch

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
        width (int): Target Image width.
        height (int): Target Image height.
        
    Returns:
        torch.Tensor: Absolute scale pixel coordinates.
    """
    norm = torch.tensor([width, height], dtype=torch.float32, device=corners.device)
    return corners * norm
