import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import logging
import math
import torch
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Tuple
import torchvision.transforms as T

from common.yolo_labels import parse_yolo_keypoint_line
from common.geometry import get_visual_orientation
from common.transforms import get_train_transforms

logger = logging.getLogger(__name__)




class OrientDataset(Dataset):
    """Dataset for card orientation classification (Stage 2.5).

    Design (matches inference exactly):
    1. Load image + YOLO keypoints (physical corners [TL, TR, BR, BL] in [0,1]).
    2. Sort the 4 physical corners by atan2(dy, dx) ascending — the SAME sort
       the coarse model applies to its predicted corners at inference time.
    3. Warp the card quad using the *atan2-sorted* corners as source.
       The resulting canonical crop may look upside-down, rotated, etc.
    4. Label  =  get_visual_orientation(keypoints)
              =  which slot in the atan2-sorted sequence the physical TL occupies.
       This is exactly the cyclic shift 's' needed at inference so that
       corners_final[0] = corners_coarse[s] = physical TL.

    Returns a dict with:
        - ``image``   : [3, H, W] normalised tensor
        - ``label``   : long scalar (0/1/2/3)
        - ``img_path``: str
    """

    def __init__(self, images_dir: str, min_size: int = 800, max_size: int = 1333,
                 is_train: bool = True) -> None:
        self.images_dir = images_dir
        self.min_size    = min_size
        self.max_size    = max_size
        self.is_train    = is_train

        extensions = ('*.jpg', '*.jpeg', '*.png')
        self.image_paths: List[str] = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()
        self.transforms = get_train_transforms(image_size=None, min_size=min_size, max_size=max_size, is_train=is_train)

    def _get_label_path(self, image_path: str) -> str:
        img_dir    = os.path.dirname(image_path)
        base_name  = os.path.basename(image_path)
        name, _    = os.path.splitext(base_name)
        parent_dir = os.path.dirname(img_dir)
        root_dir   = os.path.dirname(parent_dir)
        split_name = os.path.basename(img_dir)
        label_dir  = os.path.join(root_dir, 'labels', split_name)
        return os.path.join(label_dir, name + '.txt')

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Cannot open image {img_path}: {e}")

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
                logger.error(f"Failed to read annotation {label_path}: {e}")

        if not keypoints:
            raise ValueError(f"Missing/malformed annotation for {img_path}")

        # ── Label ─────────────────────────────────────────────────────────
        # get_visual_orientation returns the quadrant index of physical TL
        # relative to the centroid, which equals its position in the atan2-
        # sorted sequence.  This is the cyclic shift the inference code needs.
        orient_class = get_visual_orientation(keypoints)  # 0..3

        # We now use the complete, dynamically resized image exactly like the Coarse stage.
        image_t, _ = self.transforms(image, keypoints)

        return {
            'image':    image_t,
            'label':    torch.tensor(orient_class, dtype=torch.long),
            'img_path': img_path,
        }

def collate_fn_pad(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_len = (0, max_w - w, 0, max_h - h)
        padded_img = torch.nn.functional.pad(img, pad_len, value=0.0)
        padded_images.append(padded_img)
        
    return {
        'image': torch.stack(padded_images, dim=0),
        'label': torch.stack(labels, dim=0),
        'img_path': img_paths
    }
