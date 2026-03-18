import os
import json
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List

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
            
        import math
        cx = sum([pt[0] for pt in keypoints]) / 4.0
        cy = sum([pt[1] for pt in keypoints]) / 4.0
        def angle_from_center(pt: List[float]) -> float:
            return math.atan2(pt[1] - cy, pt[0] - cx)
            
        keypoints = sorted(keypoints, key=angle_from_center)
            
        image_t, keypoints_t = self.transforms(image, keypoints)
        
        return {
            'image': image_t,
            'corners': torch.tensor(keypoints_t, dtype=torch.float32),
            'img_path': img_path,
            'orig_size': (w, h)
        }
