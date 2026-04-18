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
from common.geometry import compute_homography, warp_image
from boundingbox.models.boundingbox_quad_net import BoundingBoxQuadNet
from refiner.models.patch_refiner import PatchRefinerNet
from orient.models.orient_net import OrientNet

ORIENT_LABELS = {0: '0°', 1: '90°', 2: '180°', 3: '270°'}

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
    parser = argparse.ArgumentParser(description="Unified BoundingBox → Orient → Refined Corner Detection Inference.")
    parser.add_argument('--boundingbox_model', '--boundingbox_mode', type=str, required=True, help="Path to Stage 1 (BoundingBox) model.")
    parser.add_argument('--refiner_model', type=str, required=True, help="Path to Stage 2 (Refiner) TorchScript model.")
    parser.add_argument('--orient_model', type=str, default='',
                        help="(Optional) Path to Stage 2.5 (Orient) model. If omitted, orientation correction is skipped.")
    parser.add_argument('--input', type=str, required=True, help="Path to image file or directory.")
    parser.add_argument('--output_dir', type=str, default='', help="Directory to save results. If empty, uses input path + '_cropped'.")
    
    parser.add_argument('--refiner_size', type=int, default=640, help="Input size for Refiner model.")
    parser.add_argument('--margin_ratio', type=float, default=0.15, help="Margin to expand Stage 1 BBOX for refinement.")
    parser.add_argument('--orient_crop_size', type=int, default=128,
                        help="Canonical crop size fed to OrientNet.")
    
    add_device_args(parser, default='cpu')
    parser.add_argument('--save_json', action='store_true', default=True, help="Save numeric results to JSON.")
    parser.add_argument('--no_vis', action='store_true', default=False, help="Skip visualization generation.")
    parser.add_argument('--pytorch', action='store_true', help="Load PyTorch checkpoints (.pt) instead of TorchScript models.")
    
    return parser.parse_args()

def preprocess_boundingbox(image: Image.Image, target_size: int, device: torch.device) -> torch.Tensor:
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

def preprocess_refiner(img_rgb: np.ndarray, corners_px: List[List[float]], margin_ratio: float, input_size: int, device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Crops full card based on quad extent, resizes isotropically and pads to input_size."""
    H, W = img_rgb.shape[:2]
    pts = np.array(corners_px)
    x1_orig, y1_orig = pts.min(axis=0)
    x2_orig, y2_orig = pts.max(axis=0)
    bw, bh = x2_orig - x1_orig, y2_orig - y1_orig
    cx, cy = (x1_orig + x2_orig) / 2, (y1_orig + y2_orig) / 2

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

def warp_card_crop(img_rgb_np: np.ndarray, corners_px: List[List[float]],
                   crop_size: int) -> np.ndarray:
    """Warps the 4-corner card quad to a canonical square crop."""
    s   = float(crop_size)
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    src = np.array(corners_px, dtype=np.float32)
    H   = compute_homography(src, dst)
    return warp_image(img_rgb_np, H, (crop_size, crop_size))


def draw_results(img_bgr: np.ndarray, boundingbox_pts: List[List[float]], refined_pts: List[List[float]]) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    
    color_boundingbox = (0, 165, 255)  # Orange
    color_refined = (0, 255, 0)   # Green
    thickness = max(1, int(min(w, h) / 400))
    radius_c = max(3, int(min(w, h) / 200))
    radius_r = max(4, int(min(w, h) / 150))
    font_scale = max(0.4, min(w, h) / 1000.0)

    # Convert to integer tuples
    c_pts = [tuple(map(int, p)) for p in boundingbox_pts]
    r_pts = [tuple(map(int, p)) for p in refined_pts]

    # Draw boundingbox markers (optional, small circles)
    for pt in c_pts:
        cv2.circle(vis, pt, radius_c, color_boundingbox, 1)

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

def process_single_image(img_path: str, boundingbox_model: Any, refiner_model: Any,
                         orient_model: Optional[Any],
                         args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    t_start = sync_time()
    
    # 1. Load and Preprocess for BoundingBox
    try:
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_bgr = cv2.imread(img_path) # For visualization
    except Exception as e:
        logger.error(f"Failed to load image {img_path}: {e}")
        return None

    # Step 1: BoundingBox Detection
    t_c0 = sync_time()
    boundingbox_input = preprocess_boundingbox(pil_img, args.boundingbox_size, device)
    with torch.inference_mode():
        boundingbox_out = boundingbox_model(boundingbox_input)
    
    if 'corners' not in boundingbox_out:
        logger.error(f"BoundingBox model failed to return corners for {img_path}")
        return None
        
    corners_norm = boundingbox_out['corners'][0].cpu().tolist() # [4, 2]
    boundingbox_pts_px = []
    for (nx, ny) in corners_norm:
        px = nx * orig_w
        py = ny * orig_h
        boundingbox_pts_px.append([px, py])
    t_boundingbox = sync_time() - t_c0

    # Step 2 (optional): Orientation Classification
    orient_class = None
    t_orient = 0.0
    img_rgb_np = np.array(pil_img)
    if orient_model is not None:
        t_o0 = sync_time()
        crop_np = warp_card_crop(img_rgb_np, boundingbox_pts_px, args.orient_crop_size)
        crop_pil = Image.fromarray(crop_np)
        crop_t = TF.to_tensor(crop_pil)
        crop_t = TF.normalize(crop_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        crop_t = crop_t.unsqueeze(0).to(device)
        with torch.inference_mode():
            orient_logits = orient_model(crop_t)
        orient_class = int(orient_logits.argmax(dim=-1).item())
        # Cyclically shift corners so corners[0] is the physical TL
        n = len(boundingbox_pts_px)
        boundingbox_pts_px = [boundingbox_pts_px[(i + orient_class) % n] for i in range(n)]
        t_orient = sync_time() - t_o0
    
    # Step 3: Full-Card Refinement
    t_r0 = sync_time()
    
    refine_input, r_meta = preprocess_refiner(img_rgb_np, boundingbox_pts_px, 
                                             args.margin_ratio, args.refiner_size, device)
    
    with torch.inference_mode():
        refine_out = refiner_model(refine_input)
        
    if isinstance(refine_out, (list, tuple)):
        p_norm = refine_out[0][0].cpu().numpy() # [4, 2]
    else:
        p_norm = refine_out[0].cpu().numpy()
        
    # Inverse mapping to original pixels
    r_scale = r_meta['scale']
    r_px, r_py = r_meta['padding']
    r_x1, r_y1, _, _ = r_meta['crop_box']
    
    refined_pts_px = []
    for i in range(4):
        # 1. Normalized -> Padded-input pixels
        curr_px = p_norm[i, 0] * args.refiner_size
        curr_py = p_norm[i, 1] * args.refiner_size
        # 2. Unpad -> Resized crop pixels
        curr_rx = curr_px - r_px
        curr_ry = curr_py - r_py
        # 3. Scale -> Crop pixels
        curr_cx = curr_rx / r_scale
        curr_cy = curr_ry / r_scale
        # 4. Offset -> Original pixels
        curr_orig_x = curr_cx + r_x1
        curr_orig_y = curr_cy + r_y1
        refined_pts_px.append([float(curr_orig_x), float(curr_orig_y)])
        
    t_refine = sync_time() - t_r0
    
    t_total = sync_time() - t_start
    
    # Visualization
    if not args.no_vis:
        vis = draw_results(img_bgr, boundingbox_pts_px, refined_pts_px)
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.output_dir, out_name), vis)
        
    # JSON
    if args.save_json:
        result = {
            'image': img_path,
            'boundingbox_corners': boundingbox_pts_px,
            'refined_corners': refined_pts_px,
            'orient_class': orient_class,
            'orient_label': ORIENT_LABELS.get(orient_class, 'n/a') if orient_class is not None else 'n/a',
            'times_ms': {
                'boundingbox': t_boundingbox * 1000,
                'orient': t_orient * 1000,
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
    orient_model = None
    if args.pytorch:
        logger.info(f"Loading BoundingBox model (PyTorch): {args.boundingbox_model}")
        boundingbox_model = BoundingBoxQuadNet().to(device)
        load_checkpoint(boundingbox_model, None, None, args.boundingbox_model, device=device)
        boundingbox_model.eval()
        
        logger.info(f"Loading Refiner model (PyTorch): {args.refiner_model}")
        refiner_model = PatchRefinerNet(input_size=args.refiner_size).to(device)
        load_checkpoint(refiner_model, None, None, args.refiner_model, device=device)
        refiner_model.eval()

        if args.orient_model:
            logger.info(f"Loading Orient model (PyTorch): {args.orient_model}")
            orient_model = OrientNet(num_classes=4).to(device)
            load_checkpoint(orient_model, None, None, args.orient_model, device=device)
            orient_model.eval()
    else:
        logger.info(f"Loading BoundingBox model (TorchScript): {args.boundingbox_model}")
        boundingbox_model = torch.jit.load(args.boundingbox_model, map_location=device)
        boundingbox_model.eval()
        
        logger.info(f"Loading Refiner model (TorchScript): {args.refiner_model}")
        refiner_model = torch.jit.load(args.refiner_model, map_location=device)
        refiner_model.eval()

        if args.orient_model:
            logger.info(f"Loading Orient model (TorchScript): {args.orient_model}")
            orient_model = torch.jit.load(args.orient_model, map_location=device)
            orient_model.eval()

    if orient_model is None:
        logger.info('Orient model not provided — semantic corner ordering skipped.')
    
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
        t = process_single_image(img_p, boundingbox_model, refiner_model,
                                 orient_model, args, device, logger)
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
