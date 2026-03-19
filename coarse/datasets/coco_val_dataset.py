import os
import json
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Tuple

from common.transforms import get_train_transforms

logger = logging.getLogger(__name__)

class COCOValDataset(Dataset):
    """Dataset class for loading validation images and COCO format keypoint annotations.
    
    Args:
        images_dir (str): Directory containing validation images.
        annotation_file (str): Path to the COCO format validation JSON file.
        image_size (int, optional): Target size to resize images to. Defaults to 384.
    """
    def __init__(self, images_dir: str, annotation_file: str, image_size: int = 384) -> None:
        self.images_dir = images_dir
        self.image_size = image_size
        self.transforms = get_train_transforms(image_size, is_train=False)
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                self.coco_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse COCO annotation file {annotation_file}: {e}")
            raise e
            
        self.images = {item['id']: item for item in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
    def __len__(self) -> int:
        """Size limit mapping blocks count access index struct return size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a validation image and its corresponding annotations.
        
        Args:
            idx (int): Dataset index.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'image': Preprocessed image tensor.
                - 'corners': Tensor of normalized keypoint coordinates.
                - 'img_path': Original image path for debugging/logging.
                - 'orig_size': Original image dimensions (width, height) tuple.
                
        Raises:
            Exception: If an underlying unrecoverable file error occurs.
        """
        ann = self.annotations[idx]
        img_info = self.images[ann['image_id']]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            raise e
            
        w, h = image.size
        kps = ann['keypoints']
        keypoints: List[List[float]] = []
        
        for i in range(4):
            x = kps[i*3] / w
            y = kps[i*3 + 1] / h
            keypoints.append([x, y])
            
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
            'orig_width': torch.tensor(float(w), dtype=torch.float32),
            'orig_height': torch.tensor(float(h), dtype=torch.float32)
        }

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
        cv2.polylines(edges, [pts], isClosed=True, color=255, thickness=1)
        edges_f = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), sigmaX=0.8)
        
        # Convert to float tensors [1, H, W] in [0, 1]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        edges_t = torch.from_numpy(edges_f).float().unsqueeze(0) / 255.0
        
        return mask_t, edges_t
