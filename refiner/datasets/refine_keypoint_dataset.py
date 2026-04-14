import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from common.yolo_labels import parse_yolo_keypoint_line
from common.transforms import get_train_transforms

class RefineKeypointDataset(Dataset):
    """
    Stage 2 Dataset: Extracts high-res patches around coarse corner estimates.
    Simulates Stage 1 errors using random jitter to train the refiner for robustness.
    """
    def __init__(self, image_dir, is_train=True, jitter_px=15.0, patch_size=96, min_size=800, max_size=1333):
        self.image_dir = image_dir
        self.is_train = is_train
        self.jitter_px = jitter_px
        self.patch_size = patch_size
        self.min_size = min_size
        self.max_size = max_size
        
        # Load image paths and labels (reusing YOLO parsing from Stage 1)
        self.img_paths = []
        self.labels = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            import glob
            self.img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        self.img_paths = sorted(self.img_paths)
        for img_p in self.img_paths:
            lbl_p = img_p.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(lbl_p):
                try:
                    with open(lbl_p, 'r') as f:
                        line = f.readline()
                        if line:
                            pts = parse_yolo_keypoint_line(line)
                            if pts:
                                self.labels.append(pts)
                                continue
                except Exception:
                    pass
            self.labels.append(None)
                
        # Filter valid samples
        valid_indices = [i for i, lbl in enumerate(self.labels) if lbl is not None]
        self.img_paths = [self.img_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

        # Use the standard global transforms for spatial resize and normalization
        self.transforms = get_train_transforms(image_size=None, min_size=self.min_size, max_size=self.max_size, is_train=False)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_corners_norm = self.labels[idx] # [4, 2]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self))
            
        # Transform scales image to [800, 1333] bound natively and outputs Normalised [0,1] Tensor
        img_t, _ = self.transforms(image, gt_corners_norm)
        C, H, W = img_t.shape

        patches = []
        targets = []

        for i in range(4):
            # True corner in pixels
            tx = gt_corners_norm[i][0] * W
            ty = gt_corners_norm[i][1] * H
            
            # Simulate Coarse Stage 1 Error with Jitter
            if self.jitter_px > 0:
                # Use a mix of uniform and occasional large outliers to simulate real Stage 1 behavior
                if self.is_train:
                    # 90% normal jitter, 10% "hard" jitter
                    scale = self.jitter_px if np.random.random() < 0.9 else self.jitter_px * 1.5
                    jx = np.random.uniform(-scale, scale)
                    jy = np.random.uniform(-scale, scale)
                else:
                    # Deterministic jitter for validation consistency if jitter_px > 0
                    # (Uses index-based seed for reproducibility within the epoch)
                    rng = np.random.RandomState(idx * 10 + i)
                    jx = rng.uniform(-self.jitter_px, self.jitter_px)
                    jy = rng.uniform(-self.jitter_px, self.jitter_px)
            else:
                jx, jy = 0.0, 0.0
            
            # Estimated center (what Stage 1 would give us)
            cx, cy = tx + jx, ty + jy
            
            # Crop Patch
            half = self.patch_size // 2
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
            
            # Pad if out of bounds (Refinement needs consistent patch size)
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                patch = torch.zeros((3, self.patch_size, self.patch_size), dtype=torch.float32)
                # Normalize padding bounds
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                patch = (patch - mean) / std # Fill padded territory with 0-value image mean!
                
                # Calculate valid intersection
                sx1, sy1 = max(0, x1), max(0, y1)
                sx2, sy2 = min(W, x2), min(H, y2)
                dx1, dy1 = max(0, -x1), max(0, -y1)
                dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
                if sx2 > sx1 and sy2 > sy1:
                    patch[:, dy1:dy2, dx1:dx2] = img_t[:, sy1:sy2, sx1:sx2]
            else:
                patch = img_t[:, y1:y2, x1:x2].clone()
            
            # Target position inside the patch [0, 1]
            # Center of the patch is at (0.5, 0.5) in this space if jitter is 0.
            ox = (tx - x1) / self.patch_size
            oy = (ty - y1) / self.patch_size
            
            patches.append(patch)
            targets.append(torch.tensor([ox, oy], dtype=torch.float32))

        return {
            'patches': torch.stack(patches), # [4, 3, 96, 96]
            'targets': torch.stack(targets), # [4, 2]
            'img_path': img_path,
            'index': idx
        }
