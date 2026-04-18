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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.device import add_device_args, resolve_device, log_device_info, sync_time
from common.checkpoint import load_checkpoint
from refiner.models.patch_refiner import PatchRefinerNet

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
    parser = argparse.ArgumentParser(description="Run Full-Card Corner Refiner on Stage 1 outputs.")
    parser.add_argument('--model', type=str, required=True, help="Path to Refiner TorchScript model (.pt).")
    parser.add_argument('--image', type=str, required=True, help="Path to image or directory.")
    parser.add_argument('--boundingbox_results', type=str, required=True, help="Path to Stage 1 JSON results directory.")
    parser.add_argument('--output_dir', type=str, default='', help="Output directory.")
    parser.add_argument('--input_size', type=int, default=640, help="Input size used during training.")
    parser.add_argument('--margin_ratio', type=float, default=0.15, help="Margin to expand Stage 1 BBOX.")
    add_device_args(parser, default='cpu')

    parser.add_argument('--save_vis', action='store_true', default=True)
    parser.add_argument('--save_json', action='store_true', default=True)
    parser.add_argument('--pytorch', action='store_true', help="Load PyTorch checkpoint (.pt) instead of TorchScript model.")
    return parser.parse_args()

def preprocess_image(img_rgb: np.ndarray, bbox_px: List[float], margin_ratio: float, input_size: int, device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Crops full card based on bbox, resizes isotropically and pads to input_size."""
    H, W = img_rgb.shape[:2]
    cx, cy, bw, bh = bbox_px
    
    mw = bw * margin_ratio
    mh = bh * margin_ratio
    
    x1 = int(round(max(0, cx - bw/2 - mw)))
    y1 = int(round(max(0, cy - bh/2 - mh)))
    x2 = int(round(min(W, cx + bw/2 + mw)))
    y2 = int(round(min(H, cy + bh/2 + mh)))
    
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    
    crop = img_rgb[y1:y2, x1:x2]
    cw, ch = x2 - x1, y2 - y1
    
    # Isotropic Resize
    scale = input_size / max(cw, ch)
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    crop_resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Padding
    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2
    full_input = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    full_input[pad_y:pad_y+nh, pad_x:pad_x+nw] = crop_resized
    
    # Normalize
    img_t = torch.from_numpy(full_input).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    
    metadata = {
        'crop_box': [x1, y1, x2, y2],
        'scale': scale,
        'padding': [pad_x, pad_y],
        'input_size': input_size
    }
    
    return img_t.unsqueeze(0).to(device), metadata

def process_image(img_path: str, model: Any, boundingbox_data: Dict, args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    t_start = sync_time()
    img = cv2.imread(img_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get BBOX from Stage 1 JSON
    if 'bbox_pixel' in boundingbox_data:
        bbox_px = boundingbox_data['bbox_pixel']
    else:
        corners = np.array(boundingbox_data['corners_pixel'])
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox_px = [cx, cy, bw, bh]
    
    img_t, meta = preprocess_image(img_rgb, bbox_px, args.margin_ratio, args.input_size, device)
    
    with torch.no_grad():
        out = model(img_t)
        # Model returns (final, coarse)
        if isinstance(out, (list, tuple)):
            pred_coords_norm = out[0] # [1, 4, 2] - Final refined prediction
        else:
            pred_coords_norm = out

            
    # EXACT Inverse mapping to original pixels
    p_norm = pred_coords_norm[0].cpu().numpy() # [4, 2] in [0, 1] padded space
    scale = meta['scale']
    px, py = meta['padding']
    x1, y1, _, _ = meta['crop_box']
    
    refined_corners_px = []
    for i in range(4):
        # 1. Normalized -> Padded-input pixels
        curr_px = p_norm[i, 0] * args.input_size
        curr_py = p_norm[i, 1] * args.input_size
        # 2. Unpad -> Resized crop pixels
        curr_rx = curr_px - px
        curr_ry = curr_py - py
        # 3. Scale -> Crop pixels
        curr_cx = curr_rx / scale
        curr_cy = curr_ry / scale
        # 4. Offset -> Original pixels
        curr_orig_x = curr_cx + x1
        curr_orig_y = curr_cy + y1
        refined_corners_px.append([float(curr_orig_x), float(curr_orig_y)])

    
    t_end = sync_time()
    total_ms = (t_end - t_start) * 1000
    
    # Visualization
    if args.save_vis:
        vis_img = img.copy()
        for i in range(4):
            r_pt = tuple(map(int, refined_corners_px[i]))
            cv2.circle(vis_img, r_pt, 4, (0, 255, 0), -1)   # Refined Green
            
        # Connect refined points
        pts = np.array(refined_corners_px, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
        
        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis_img)

    # JSON
    if args.save_json:
        result = {
            'image': img_path,
            'boundingbox_bbox': bbox_px,
            'refined_corners': refined_corners_px,
            'timing_ms': total_ms
        }
        json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        json_path = os.path.join(args.output_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
            
    return total_ms

def main():
    args = parse_args()
    logger = setup_logging()
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    if not args.output_dir:
        input_p = os.path.abspath(args.image)
        if os.path.isdir(input_p):
            input_base = input_p.rstrip('\\/')
        else:
            input_base = os.path.dirname(input_p)
        args.output_dir = input_base + "_refined"
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    t0 = sync_time()
    try:
        if args.pytorch:
            logger.info(f"Loading Refiner model (PyTorch): {args.model}")
            model = PatchRefinerNet(input_size=args.input_size).to(device)
            load_checkpoint(model, None, None, args.model, device=device)
            model.eval()
        else:
            logger.info(f"Loading Refiner model (TorchScript): {args.model}")
            model = torch.jit.load(args.model, map_location=device)
            model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    t_load_model = sync_time() - t0
    
    logger.info(f"Model Path:         {args.model}")
    logger.info(f"Model Load Time:    {t_load_model*1000:.1f} ms\n")
    
    # Find images
    image_paths = []
    if os.path.isdir(args.image):
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
    else:
        image_paths = [args.image]
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    for img_p in image_paths:
        base = os.path.basename(img_p)
        name_no_ext = os.path.splitext(base)[0]
        boundingbox_json = os.path.join(args.boundingbox_results, f"{name_no_ext}.json")
        
        if not os.path.exists(boundingbox_json):
            logger.warning(f"Skipping {base}: Missing boundingbox result {boundingbox_json}")
            continue
            
        with open(boundingbox_json, 'r') as f:
            boundingbox_data = json.load(f)
            
        ms = process_image(img_p, model, boundingbox_data, args, device, logger)
        if ms:
            logger.info(f"Processed {base} in {ms:.1f}ms")

if __name__ == "__main__":
    main()
