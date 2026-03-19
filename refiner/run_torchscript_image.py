import os
import sys
import glob
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.device import add_device_args, resolve_device, log_device_info, sync_time
from common.visualization import save_diagnostic_visualization

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('refiner_inference')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    return logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TorchScript Patch Refiner on Stage 1 outputs.")
    parser.add_argument('--model', type=str, required=True, help="Path to Refiner TorchScript model (.pt).")
    parser.add_argument('--image', type=str, required=True, help="Path to image or directory.")
    parser.add_argument('--coarse_results', type=str, required=True, help="Path to Stage 1 JSON results directory.")
    parser.add_argument('--output_dir', type=str, default='./refiner_outputs', help="Output directory.")
    parser.add_argument('--patch_size', type=int, default=96, help="Patch size used during training.")
    add_device_args(parser, default='cpu')
    parser.add_argument('--save_vis', action='store_true', default=True)
    parser.add_argument('--save_json', action='store_true', default=True)
    return parser.parse_args()

def extract_patch(img_np: np.ndarray, cx: float, cy: float, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extracts a patch and returns (patch, top_left_px)."""
    H, W = img_np.shape[:2]
    half = patch_size // 2
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = x1 + patch_size, y1 + patch_size
    
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(W, x2), min(H, y2)
        dx1, dy1 = max(0, -x1), max(0, -y1)
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        if sx2 > sx1 and sy2 > sy1:
            patch[dy1:dy2, dx1:dx2] = img_np[sy1:sy2, sx1:sx2]
    else:
        patch = img_np[y1:y2, x1:x2]
    return patch, (x1, y1)

def preprocess_patch(patch_np: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = TF.to_tensor(patch_np)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0).to(device)

def process_image(img_path: str, model: Any, coarse_data: Dict, args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    t_start = sync_time()
    img = cv2.imread(img_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    coarse_corners_px = coarse_data['corners_pixel'] # [4, 2]
    refined_corners_px = []
    
    t_patches_start = sync_time()
    for i in range(4):
        cx, cy = coarse_corners_px[i]
        patch_np, (x1, y1) = extract_patch(img_rgb, cx, cy, args.patch_size)
        patch_t = preprocess_patch(patch_np, device)
        
        with torch.inference_mode():
            pred = model(patch_t) # [1, 2] in [0, 1] range
        
        ox, oy = pred[0].cpu().tolist()
        rx = x1 + ox * args.patch_size
        ry = y1 + oy * args.patch_size
        refined_corners_px.append([rx, ry])
    
    t_end = sync_time()
    total_ms = (t_end - t_start) * 1000
    
    # Visualization
    if args.save_vis:
        vis_img = img.copy()
        for i in range(4):
            c_pt = tuple(map(int, coarse_corners_px[i]))
            r_pt = tuple(map(int, refined_corners_px[i]))
            cv2.circle(vis_img, c_pt, 4, (0, 165, 255), -1) # Coarse Orange
            cv2.circle(vis_img, r_pt, 3, (0, 255, 0), -1)   # Refined Green
            cv2.line(vis_img, c_pt, r_pt, (255, 255, 255), 1)
        
        # Connect refined points
        pts = np.array(refined_corners_px, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
        
        out_path = os.path.join(args.output_dir, f"refined_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, vis_img)

    # JSON
    if args.save_json:
        result = {
            'image': img_path,
            'coarse_corners': coarse_corners_px,
            'refined_corners': refined_corners_px,
            'timing_ms': total_ms
        }
        json_path = os.path.join(args.output_dir, f"refined_{os.path.basename(img_path)}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
            
    return total_ms

def main():
    args = parse_args()
    logger = setup_logging()
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading Refiner Model: {args.model}")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    
    # Find images
    image_paths = []
    if os.path.isdir(args.image):
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
    else:
        image_paths = [args.image]
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    for img_p in image_paths:
        # Expected Stage 1 JSON: result_[basename].json
        base = os.path.basename(img_p)
        coarse_json = os.path.join(args.coarse_results, f"result_{base}.json")
        
        if not os.path.exists(coarse_json):
            logger.warning(f"Skipping {base}: Missing coarse result {coarse_json}")
            continue
            
        with open(coarse_json, 'r') as f:
            coarse_data = json.load(f)
            
        ms = process_image(img_p, model, coarse_data, args, device, logger)
        if ms:
            logger.info(f"Processed {base} in {ms:.1f}ms")

if __name__ == "__main__":
    main()
