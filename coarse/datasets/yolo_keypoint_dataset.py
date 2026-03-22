import os
import glob
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Tuple

from common.yolo_labels import parse_yolo_keypoint_line
from common.transforms import get_train_transforms

logger = logging.getLogger(__name__)

class YOLOKeypointDataset(Dataset):
    """Dataset class for loading ID card images and YOLO format coordinate annotations.
    
    Args:
        images_dir (str): Root directory path containing training or validation images.
        image_size (int, optional): Target size to resize images to. Defaults to 384.
        is_train (bool, optional): If True, applies training augmentations. Defaults to True.
    """
    def __init__(self, images_dir: str, image_size: int = 384, is_train: bool = True) -> None:
        self.images_dir = images_dir
        
        extensions = ('*.jpg', '*.jpeg', '*.png')
        self.image_paths: List[str] = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()
            
        self.image_size = image_size
        self.is_train = is_train
        self.transforms = get_train_transforms(image_size, is_train=is_train)
        
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

    def generate_mask_and_edges(self, keypoints: List[List[float]], size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a binary mask and a Gaussian boundary edge map from 4 corners.
        
        Args:
            keypoints (List[List[float]]): Normalized corners [4, 2].
            size (int): Target spatial resolution (e.g. 96).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mask, Gaussian edges) tensors of shape [1, size, size].
        """
        import cv2
        import numpy as np
        
        mask = np.zeros((size, size), dtype=np.uint8)
        edges = np.zeros((size, size), dtype=np.uint8)
        
        # Scale to target size
        pts = (np.array(keypoints) * size).astype(np.int32)
        
        # 1. Generate filled polygon mask
        cv2.fillPoly(mask, [pts], 255)
        
        # 2. Generate Gaussian boundary (Performance Boost v2)
        # Draw 1px binary line first
        cv2.polylines(edges, [pts], isClosed=True, color=255, thickness=1)
        # Apply Gaussian blur to create a smooth falloff (3px effective width)
        edges_f = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), sigmaX=0.8)
        # Normalize back to [0, 255] for consistency if needed, but we divide by 255 later
        
        # Convert to float tensors [1, H, W] in [0, 1]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        edges_t = torch.from_numpy(edges_f).float().unsqueeze(0) / 255.0
        
        return mask_t, edges_t

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
                - 'mask': Card area mask [1, 48, 48].
                - 'edges': Card boundary edges [1, 48, 48].
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

        # Generate dense geometric targets (96x96 for v2 boost)
        mask_t, edges_t = self.generate_mask_and_edges(keypoints_t, size=96)
        
        return {
            'index': idx,
            'image': image_t,
            'corners': torch.tensor(keypoints_t, dtype=torch.float32),
            'mask': mask_t,
            'edges': edges_t,
            'img_path': img_path,
            'orig_width': torch.tensor(float(orig_w)),
            'orig_height': torch.tensor(float(orig_h))
        }
