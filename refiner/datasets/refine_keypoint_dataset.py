import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from common.yolo_labels import parse_yolo_keypoint_line

class FullCardRefinerDataset(Dataset):
    """
    Stage 2 Dataset: Full-Card Corner Refiner.
    Extracts a single crop of the full card based on the Stage 1 BBOX.
    
    Target Coordinate Convention:
    - Points are normalized [0, 1] relative to the CROPPED image bounds.
    """
    def __init__(self, image_dir, input_size=256, margin_ratio=0.15):
        self.image_dir = image_dir
        self.input_size = input_size
        self.margin_ratio = margin_ratio
        
        # Load image paths and labels
        self.img_paths = []
        self.labels = []
        import glob
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        self.img_paths = sorted(self.img_paths)
        for img_p in self.img_paths:
            lbl_p = img_p.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(lbl_p):
                try:
                    with open(lbl_p, 'r') as f:
                        line = f.readline()
                        if line:
                            # Use existing project utility for parsing
                            parsed = parse_yolo_keypoint_line(line)
                            if parsed:
                                self.labels.append(parsed)
                                continue
                except Exception:
                    pass
            self.labels.append(None)
                
        # Filter valid samples
        valid_indices = [i for i, lbl in enumerate(self.labels) if lbl is not None]
        self.img_paths = [self.img_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def reorder_corners(pts):
        """
        Reorders corners to [TL, TR, BR, BL] based on image coordinates.
        TL: min(x+y), TR: max(x-y), BR: max(x+y), BL: min(x-y)
        """
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Using DIFF(x,y) = y-x. 
        # TR: max(x-y) -> min(y-x)
        # BL: min(x-y) -> max(y-x)
        d = pts[:, 1] - pts[:, 0]
        
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        # label['bbox'] is [cx, cy, w, h] normalized in original image space
        cx, cy, w, h = label['bbox']
        gt_kpts = np.array(label['keypoints'], dtype=np.float32)

        try:
            image_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image_pil.size
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self))
            
        # 1. Crop calculations
        mw = w * self.margin_ratio
        mh = h * self.margin_ratio
        
        x1 = max(0, cx - w/2 - mw)
        y1 = max(0, cy - h/2 - mh)
        x2 = min(1, cx + w/2 + mw)
        y2 = min(1, cy + h/2 + mh)
        
        px1, py1 = int(round(x1 * orig_w)), int(round(y1 * orig_h))
        px2, py2 = int(round(x2 * orig_w)), int(round(y2 * orig_h))
        
        if px2 <= px1: px2 = px1 + 1
        if py2 <= py1: py2 = py1 + 1
        
        crop_w = px2 - px1
        crop_h = py2 - py1
        
        # 2. Isotropic resize
        scale = self.input_size / max(crop_w, crop_h)
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        
        crop = image_pil.crop((px1, py1, px2, py2))
        crop_resized = crop.resize((new_w, new_h), Image.BILINEAR)
        
        # 3. Padding to square
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        
        full_input = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
        full_input.paste(crop_resized, (pad_x, pad_y))
        
        # 4. Normalize image
        img_np = np.array(full_input)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        
        # 5. Transform Corners to Padded Input Space [0, 1]
        # gt_kpts are normalized [0, 1] in original image space
        transformed_kpts = []
        for kx, ky in gt_kpts:
            # Shift by crop origin
            kx_c = kx * orig_w - px1
            ky_c = ky * orig_h - py1
            
            # Scale and Shift by padding
            kx_final = (kx_c * scale + pad_x) / self.input_size
            ky_final = (ky_c * scale + pad_y) / self.input_size
            transformed_kpts.append([kx_final, ky_final])
            
        ordered_kpts = self.reorder_corners(transformed_kpts)
        
        return {
            'image': img_t,
            'targets': torch.from_numpy(ordered_kpts), 
            'img_path': img_path,
            'metadata': {
                'crop_box': torch.tensor([px1, py1, px2, py2], dtype=torch.float32),
                'scale': torch.tensor(scale, dtype=torch.float32),
                'padding': torch.tensor([pad_x, pad_y], dtype=torch.float32),
                'input_size': torch.tensor(self.input_size, dtype=torch.float32)
            }
        }


# Alias for backward compatibility in imports
RefineKeypointDataset = FullCardRefinerDataset
