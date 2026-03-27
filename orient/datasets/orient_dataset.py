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
from common.geometry import get_visual_orientation, compute_homography, warp_image

logger = logging.getLogger(__name__)

# ImageNet normalisation (matches coarse pipeline)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def _build_canonical_dst(crop_size: int) -> List[Tuple[float, float]]:
    """Returns the 4 canonical card corner destinations in visual TL→TR→BR→BL order."""
    s = float(crop_size)
    return [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]


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
        - ``image``   : [3, crop_size, crop_size] normalised tensor
        - ``label``   : long scalar (0/1/2/3)
        - ``img_path``: str
    """

    def __init__(self, images_dir: str, crop_size: int = 128,
                 is_train: bool = True) -> None:
        self.images_dir = images_dir
        self.crop_size   = crop_size
        self.is_train    = is_train

        extensions = ('*.jpg', '*.jpeg', '*.png')
        self.image_paths: List[str] = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()

        self.normalize = T.Normalize(mean=MEAN, std=STD)

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

        # ── Sort corners by atan2 (mirrors coarse model inference) ────────
        # Absolute pixel positions
        pts_px = [[kp[0] * orig_w, kp[1] * orig_h] for kp in keypoints]
        cx = sum(p[0] for p in pts_px) / 4.0
        cy = sum(p[1] for p in pts_px) / 4.0
        # Sort by atan2(dy, dx) ascending — identical to coarse model forward()
        pts_sorted = sorted(pts_px, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        # ── Warp using atan2-sorted corners (= inference warp source) ─────
        s = float(self.crop_size)
        src_np = np.array(pts_sorted, dtype=np.float32)          # [4,2] atan2 order
        dst_np = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
        H      = compute_homography(src_np, dst_np)
        img_np = np.array(image)
        warped = warp_image(img_np, H, (self.crop_size, self.crop_size))
        warped_pil = Image.fromarray(warped)
        
        os.makedirs("orient_debug", exist_ok=True)
        fname = os.path.basename(img_path)
        warped_pil.save(f"orient_debug/{fname}")

        # ── To tensor + normalise ─────────────────────────────────────────
        tensor = TF.to_tensor(warped_pil)
        tensor = self.normalize(tensor)

        return {
            'image':    tensor,
            'label':    torch.tensor(orient_class, dtype=torch.long),
            'img_path': img_path,
        }
