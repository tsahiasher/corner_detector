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



def get_train_transforms(image_size: int, is_train: bool = True) -> Compose:
    """Gets the transformation pipeline.

    Args:
        image_size (int): Target square dimension.
        is_train (bool, optional): Whether this is for training (enables augmentations). Defaults to True.

    Returns:
        Compose: The composition of transforms.
    """
    transforms = [ResizeImage(image_size)]

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
