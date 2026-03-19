import math
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Tuple, Any

class Compose:
    """Composes multiple transforms together.
    
    Args:
        transforms (List[Any]): List of transform operations.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms
        
    def __call__(self, img: Image.Image, keypoints: List[List[float]]) -> Tuple[torch.Tensor, List[List[float]]]:
        """Applies sequence of transforms.
        
        Args:
            img (Image.Image): Input PIL Image.
            keypoints (List[List[float]]): Input normalized keypoints.
            
        Returns:
            Tuple[torch.Tensor, List[List[float]]]: Transformed image tensor and geometry-tracked keypoints.
        """
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints

class ResizeImage:
    """Resizes image to target size. Modifies image resolution but keeps normalized keypoints unchanged.
    
    Args:
        size (int): Target width and height.
    """
    def __init__(self, size: int) -> None:
        self.size = size
        
    def __call__(self, img: Image.Image, keypoints: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
        """Forward pass.
        
        Args:
            img (Image.Image): Input Image.
            keypoints (List[List[float]]): Normalized keypoints mapped in [0,1].
            
        Returns:
            Tuple[Image.Image, List[List[float]]]: Resized Image and unchanged relative keypoints.
        """
        img = TF.resize(img, [self.size, self.size])
        return img, keypoints

class ToTensor:
    """Converts PIL image to float tensor scaled to [0,1]."""
    def __call__(self, img: Image.Image, keypoints: List[List[float]]) -> Tuple[torch.Tensor, List[List[float]]]:
        """Forward pass.
        
        Args:
            img (Image.Image): Input Image.
            keypoints (List[List[float]]): Normalized keypoints.
            
        Returns:
            Tuple[torch.Tensor, List[List[float]]]: Tensor image and keypoints.
        """
        return TF.to_tensor(img), keypoints

class Normalize:
    """Normalizes a tensor image with standard ImageNet mean and variance."""
    def __call__(self, img: torch.Tensor, keypoints: List[List[float]]) -> Tuple[torch.Tensor, List[List[float]]]:
        """Forward pass.
        
        Args:
            img (torch.Tensor): Input tensor image.
            keypoints (List[List[float]]): Normalized keypoints.
            
        Returns:
            Tuple[torch.Tensor, List[List[float]]]: Normalized image tensor, unchanged keypoints.
        """
        return TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), keypoints


class RandomRotation:
    """Randomly rotates the image and keypoints by a small angle.

    The same rotation is applied to both the image and its normalized keypoint
    coordinates so that semantic keypoint identity is preserved (index 0 stays
    the card's physical Top-Left).

    Keypoints are rotated around the image center (0.5, 0.5) in normalized
    coordinates and clamped to [0, 1] afterwards.

    Args:
        max_angle (float): Maximum absolute rotation angle in degrees.
    """
    def __init__(self, max_angle: float = 15.0) -> None:
        self.max_angle = max_angle

    def __call__(self, img: Image.Image, keypoints: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
        """Forward pass.

        Args:
            img (Image.Image): Input Image.
            keypoints (List[List[float]]): Normalized keypoints in [0, 1].

        Returns:
            Tuple[Image.Image, List[List[float]]]: Rotated image and keypoints.
        """
        angle = random.uniform(-self.max_angle, self.max_angle)

        # Rotate image (PIL rotates counter-clockwise for positive angles)
        img = TF.rotate(img, angle, fill=0)

        # Rotate keypoints around center (0.5, 0.5)
        rad = math.radians(-angle)  # negate because PIL rotates CCW
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        rotated_kps: List[List[float]] = []
        for x, y in keypoints:
            # Translate to origin at center
            dx = x - 0.5
            dy = y - 0.5
            # Apply rotation
            nx = cos_a * dx - sin_a * dy + 0.5
            ny = sin_a * dx + cos_a * dy + 0.5
            # Clamp to valid range
            rotated_kps.append([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])

        return img, rotated_kps


def get_train_transforms(image_size: int, is_train: bool = True) -> Compose:
    """Gets the transformation pipeline.

    Args:
        image_size (int): Target square dimension.
        is_train (bool, optional): Whether this is for training (enables augmentations). Defaults to True.

    Returns:
        Compose: The composition of transforms.
    """
    transforms = [ResizeImage(image_size)]

    if is_train:
        transforms.append(RandomRotation(max_angle=15.0))

    transforms.extend([ToTensor(), Normalize()])
    return Compose(transforms)

def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Reverts normalization of a tensor image back to standard [0,1] range for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized input tensor [3, H, W].
        
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean
