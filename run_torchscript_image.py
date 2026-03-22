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

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.device import add_device_args, resolve_device, log_device_info, sync_time
from common.checkpoint import load_checkpoint
from coarse.models.coarse_quad_net import CoarseQuadNet
from refiner.models.patch_refiner import PatchRefinerNet

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('unified_inference')
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
    parser = argparse.ArgumentParser(description="Unified Coarse-to-Refined Corner Detection Inference.")
    parser.add_argument('--coarse_model', '--coarse_mode', type=str, required=True, help="Path to Stage 1 (Coarse) model.")
    parser.add_argument('--refiner_model', type=str, required=True, help="Path to Stage 2 (Refiner) TorchScript model.")
    parser.add_argument('--input', type=str, required=True, help="Path to image file or directory.")
    parser.add_argument('--output_dir', type=str, default='', help="Directory to save results. If empty, uses input path + '_cropped'.")
    
    parser.add_argument('--coarse_size', type=int, default=384, help="Input size for Coarse model.")
    parser.add_argument('--patch_size', type=int, default=96, help="Patch size for Refiner model.")
    
    add_device_args(parser, default='cpu')
    parser.add_argument('--save_json', action='store_true', default=True, help="Save numeric results to JSON.")
    parser.add_argument('--no_vis', action='store_true', default=False, help="Skip visualization generation.")
    parser.add_argument('--pytorch', action='store_true', help="Load PyTorch checkpoints (.pt) instead of TorchScript models.")
    
    return parser.parse_args()

def preprocess_coarse(image: Image.Image, target_size: int, device: torch.device) -> torch.Tensor:
    img_resized = TF.resize(image, [target_size, target_size])
    tensor = TF.to_tensor(img_resized)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = TF.normalize(tensor, mean=mean, std=std)
    return tensor.unsqueeze(0).to(device)

def extract_patch(img_np: np.ndarray, cx: float, cy: float, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = img_np.shape[:2]
    half = patch_size // 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
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
    # patch_np is RGB (from PIL or converted)
    tensor = TF.to_tensor(patch_np)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0).to(device)

def draw_results(img_bgr: np.ndarray, coarse_pts: List[List[float]], refined_pts: List[List[float]]) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    
    color_coarse = (0, 165, 255)  # Orange
    color_refined = (0, 255, 0)   # Green
    thickness = max(1, int(min(w, h) / 400))
    radius_c = max(3, int(min(w, h) / 200))
    radius_r = max(4, int(min(w, h) / 150))
    font_scale = max(0.4, min(w, h) / 1000.0)

    # Convert to integer tuples
    c_pts = [tuple(map(int, p)) for p in coarse_pts]
    r_pts = [tuple(map(int, p)) for p in refined_pts]

    # Draw coarse markers (optional, small circles)
    for pt in c_pts:
        cv2.circle(vis, pt, radius_c, color_coarse, 1)

    # Draw refined polygon and indices
    for i in range(4):
        p1 = r_pts[i]
        p2 = r_pts[(i + 1) % 4]
        cv2.line(vis, p1, p2, color_refined, thickness * 2)
        cv2.circle(vis, p1, radius_r, color_refined, -1)
        
        # Draw index
        label = str(i)
        cv2.putText(vis, label, (p1[0] + 10, p1[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_refined, thickness)

    return vis

def process_single_image(img_path: str, coarse_model: Any, refiner_model: Any, 
                         args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    t_start = sync_time()
    
    # 1. Load and Preprocess for Coarse
    try:
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_bgr = cv2.imread(img_path) # For visualization
    except Exception as e:
        logger.error(f"Failed to load image {img_path}: {e}")
        return None

    # Step 1: Coarse Detection
    t_c0 = sync_time()
    coarse_input = preprocess_coarse(pil_img, args.coarse_size, device)
    with torch.inference_mode():
        coarse_out = coarse_model(coarse_input)
    
    if 'corners' not in coarse_out:
        logger.error(f"Coarse model failed to return corners for {img_path}")
        return None
        
    corners_norm = coarse_out['corners'][0].cpu().tolist() # [4, 2]
    coarse_pts_px = []
    for (nx, ny) in corners_norm:
        px = nx * orig_w
        py = ny * orig_h
        coarse_pts_px.append([px, py])
    t_coarse = sync_time() - t_c0
    
    # Step 2: Patch Refinement
    t_r0 = sync_time()
    refined_pts_px = []
    img_rgb_np = np.array(pil_img)
    
    for i in range(4):
        cx, cy = coarse_pts_px[i]
        patch_np, (x1, y1) = extract_patch(img_rgb_np, cx, cy, args.patch_size)
        patch_t = preprocess_patch(patch_np, device)
        
        with torch.inference_mode():
            refine_out = refiner_model(patch_t)
            
        # Handle both (final, coarse) tuple and single tensor returns
        if isinstance(refine_out, (list, tuple)):
            pred_tensor = refine_out[0]
        else:
            pred_tensor = refine_out
            
        # pred_tensor is [1, 2], unpacking [0] gives (dx, dy)
        dx, dy = pred_tensor[0].cpu().tolist()
        rx = x1 + dx * args.patch_size
        ry = y1 + dy * args.patch_size
        refined_pts_px.append([rx, ry])
    t_refine = sync_time() - t_r0
    
    t_total = sync_time() - t_start
    
    # Visualization
    if not args.no_vis:
        vis = draw_results(img_bgr, coarse_pts_px, refined_pts_px)
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.output_dir, out_name), vis)
        
    # JSON
    if args.save_json:
        result = {
            'image': img_path,
            'coarse_corners': coarse_pts_px,
            'refined_corners': refined_pts_px,
            'times_ms': {
                'coarse': t_coarse * 1000,
                'refine': t_refine * 1000,
                'total': t_total * 1000
            }
        }
        json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        with open(os.path.join(args.output_dir, json_name), 'w') as f:
            json.dump(result, f, indent=4)
            
    return t_total

def main():
    args = parse_args()
    logger = setup_logging()
    
    # Resolve output directory if not provided
    if not args.output_dir:
        if os.path.isdir(args.input):
            input_base = args.input.rstrip('\\/')
        else:
            input_base = os.path.dirname(os.path.abspath(args.input))
        args.output_dir = input_base + "_cropped"
    
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Models
    if args.pytorch:
        logger.info(f"Loading Coarse model (PyTorch): {args.coarse_model}")
        coarse_model = CoarseQuadNet().to(device)
        load_checkpoint(coarse_model, None, None, args.coarse_model, device=device)
        coarse_model.eval()
        
        logger.info(f"Loading Refiner model (PyTorch): {args.refiner_model}")
        refiner_model = PatchRefinerNet().to(device)
        load_checkpoint(refiner_model, None, None, args.refiner_model, device=device)
        refiner_model.eval()
    else:
        logger.info(f"Loading Coarse model (TorchScript): {args.coarse_model}")
        coarse_model = torch.jit.load(args.coarse_model, map_location=device)
        coarse_model.eval()
        
        logger.info(f"Loading Refiner model (TorchScript): {args.refiner_model}")
        refiner_model = torch.jit.load(args.refiner_model, map_location=device)
        refiner_model.eval()
    
    # Identify images
    image_paths = []
    if os.path.isdir(args.input):
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths = sorted(list(set(image_paths)))
    else:
        image_paths = [args.input]
        
    logger.info(f"Processing {len(image_paths)} images...")
    
    start_time = time.time()
    success_count = 0
    for img_p in image_paths:
        t = process_single_image(img_p, coarse_model, refiner_model, args, device, logger)
        if t is not None:
            success_count += 1
            logger.info(f"  [{success_count}/{len(image_paths)}] {os.path.basename(img_p):30s} | Total: {t*1000:6.1f}ms")
            
    total_elapsed = time.time() - start_time
    logger.info("\n" + "="*50)
    logger.info(f"Done. Processed {success_count} images in {total_elapsed:.2f}s.")
    logger.info(f"Average time per image: {total_elapsed/max(1, success_count)*1000:.1f}ms")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
