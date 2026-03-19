import os
import glob
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List

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
                - 'img_path': Original image path for debugging/logging.
                
        Raises:
            Exception: If an underlying unrecoverable file error occurs.
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
            raise ValueError(f"Missing or malformed YOLO annotation for image: {img_path}. Failing loudly to protect dataset bounding quality limits.")
            
        image_t, keypoints_t = self.transforms(image, keypoints)
        
        return {
            'image': image_t,
            'corners': torch.tensor(keypoints_t, dtype=torch.float32),
            'img_path': img_path,
            'orig_width': torch.tensor(float(orig_w)),
            'orig_height': torch.tensor(float(orig_h))
        }
