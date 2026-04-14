import os
import glob
import logging
import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Tuple

from common.yolo_labels import parse_yolo_keypoint_line
from common.transforms import get_train_transforms
from common.geometry import get_visual_orientation

logger = logging.getLogger(__name__)

class YOLOKeypointDataset(Dataset):
    """Dataset class for loading ID card images and YOLO format coordinate annotations.
    
    Args:
        images_dir (str): Root directory path containing training or validation images.
        image_size (int, optional): Target static size to resize images to. Defaults to None.
        min_size (int, optional): Minimum edge for dynamic resize. Defaults to 800.
        max_size (int, optional): Maximum edge for dynamic resize. Defaults to 1333.
        is_train (bool, optional): If True, applies training augmentations. Defaults to True.
    """
    def __init__(self, images_dir: str, image_size: int = None, min_size: int = 800, max_size: int = 1333, is_train: bool = True) -> None:
        self.images_dir = images_dir
        
        extensions = ('*.jpg', '*.jpeg', '*.png')
        self.image_paths: List[str] = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()
            
        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
        self.is_train = is_train
        self.transforms = get_train_transforms(image_size=image_size, min_size=min_size, max_size=max_size, is_train=is_train)
        
    def _get_label_path(self, image_path: str) -> str:
        # In standard YOLO setups, labels are usually in a sibling "labels" directory
        # Example: images/train/foo.jpg -> labels/train/foo.txt
        img_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        
        # Traverse up from 'images/train' -> 'labels/train'
        parent_dir = os.path.dirname(img_dir)            # e.g., 'crop-dataset-eitan-yolo/images'
        root_dir = os.path.dirname(parent_dir)           # e.g., 'crop-dataset-eitan-yolo'
        split_name = os.path.basename(img_dir)           # e.g., 'train' or 'val'
        
        # Format the true path
        label_dir = os.path.join(root_dir, 'labels', split_name)
        return os.path.join(label_dir, name + ".txt")

    def __len__(self) -> int:
        """Returns bounds configuration size limits."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns an image and its corresponding annotations.
        
        Args:
            idx (int): Dataset index.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'image': Preprocessed image tensor.
                - 'corners': Tensor of normalized keypoint coordinates.
                - 'orient': Orientation class.
                - 'img_path': Original image path.
                - 'orig_width', 'orig_height': Original image size.
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Dataset access read stream error at {img_path}: {e}")
            raise e
            
        label_path = self._get_label_path(img_path)
        keypoints: List[List[float]] = []
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        parsed = parse_yolo_keypoint_line(lines[0])
                        if parsed:
                            keypoints = parsed
            except Exception as e:
                logger.error(f"Failed to read/parse YOLO annotation file at {label_path}: {e}")
        
        if not keypoints:
            raise ValueError(f"Missing or malformed YOLO annotation for image: {img_path}.")

        image_t, keypoints_t = self.transforms(image, keypoints)
        
        # Calculate absolute orientation based on augmented geometric positions
        orient_class = get_visual_orientation(keypoints_t)
        
        return {
            'index': idx,
            'image': image_t,
            'corners': torch.tensor(keypoints_t, dtype=torch.float32),
            'orient': torch.tensor(orient_class, dtype=torch.long),
            'img_path': img_path,
            'orig_width': torch.tensor(float(orig_w)),
            'orig_height': torch.tensor(float(orig_h))
        }

def collate_fn_pad(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a batch of images and dynamically pads them to max size in batch."""
    images = [item['image'] for item in batch]
    corners = [item['corners'] for item in batch]
    orients = [item['orient'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    orig_widths = [item['orig_width'] for item in batch]
    orig_heights = [item['orig_height'] for item in batch]
    indices = [item['index'] for item in batch]
    
    scaled_widths = [img.shape[2] for img in images]
    scaled_heights = [img.shape[1] for img in images]
    
    max_h = max(scaled_heights)
    max_w = max(scaled_widths)
    
    # Must be divisible by 32 for backbone downsampling
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    padded_images = []
    adjusted_corners = []
    
    for img, corner in zip(images, corners):
        c, h, w = img.shape
        # Padding format: (left, right, top, bottom)
        pad_len = (0, max_w - w, 0, max_h - h)
        padded_img = torch.nn.functional.pad(img, pad_len, value=0.0)
        padded_images.append(padded_img)
        
        # Adjust normalized coordinates.
        corner_adjusted = corner.clone()
        corner_adjusted[:, 0] = corner_adjusted[:, 0] * (w / max_w)
        corner_adjusted[:, 1] = corner_adjusted[:, 1] * (h / max_h)
        adjusted_corners.append(corner_adjusted)
        
    return {
        'index': torch.tensor(indices, dtype=torch.long),
        'image': torch.stack(padded_images, dim=0),
        'corners': torch.stack(adjusted_corners, dim=0),
        'orient': torch.stack(orients, dim=0),
        'img_path': img_paths,
        'orig_width': torch.stack(orig_widths, dim=0),
        'orig_height': torch.stack(orig_heights, dim=0),
        'scaled_width': torch.tensor(scaled_widths, dtype=torch.float32),
        'scaled_height': torch.tensor(scaled_heights, dtype=torch.float32)
    }
