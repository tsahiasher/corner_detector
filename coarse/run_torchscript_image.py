import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from common.device import add_device_args, resolve_device, log_device_info, sync_time
from common.checkpoint import load_checkpoint
from coarse.models.coarse_quad_net import CoarseQuadNet

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configures explicit, clean logging for deterministic inference."""
    logger = logging.getLogger('inference_pipeline')
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
    parser = argparse.ArgumentParser(description="Run TorchScript global box regressor on single image or directory.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str, required=True, help="Path to the exported TorchScript model (.pt).")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image or directory of images.")
    parser.add_argument('--output_dir', type=str, default='', help="Directory to save execution artifacts (visualizations, JSON).")
    add_device_args(parser, default='cpu')
    parser.add_argument('--image_size', type=int, default=384, help="Fixed target inference dimension.")
    
    parser.add_argument('--save_json', action='store_true', default=True, help="Flag to export explicit numeric result limits coordinates to a JSON payload.")
    parser.add_argument('--save_vis', action='store_true', default=True, help="Flag to generate and save a visualization overlaid upon the absolute original image scale.")
    parser.add_argument('--no_vis', action='store_true', default=False, help="Suppresses any visual artifact generations.")
    parser.add_argument('--pytorch', action='store_true', help="Load PyTorch checkpoint (.pt) instead of TorchScript model.")
    
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Minimum global pool inference confidence required.")
    parser.add_argument('--print_coordinates', action='store_true', default=True, help="Dumps the raw parsed map scale geometry structs vector payloads explicitly into console bounds.")
    
    return parser.parse_args()

def preprocess_image(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    from common.transforms import ResizeImage
    resizer = ResizeImage(size=image_size)
    img_resized, _ = resizer(image, [])
    
    tensor = TF.to_tensor(img_resized)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = TF.normalize(tensor, mean=mean, std=std)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

def save_visualization(image: Image.Image, box_px: List[int], img_path: str, output_dir: str) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        logging.getLogger('inference_pipeline').warning("Skipping visualization because OpenCV (cv2) or NumPy is not installed.")
        return

    orig_w, orig_h = image.size
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    h, w = img_np.shape[:2]
    color_pred = (0, 165, 255)  # Orange
    thickness = max(2, int(min(w, h) / 400))

    cx, cy, pw, ph = box_to_poly(box_px)
    x1, y1, x2, y2 = box_px

    # Draw regular rectangle
    cv2.rectangle(img_np, (x1, y1), (x2, y2), color_pred, thickness)

    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, base_name)
    cv2.imwrite(out_path, img_np)

def box_to_poly(box_px: list):
     # Just dummy extraction
     x1, y1, x2, y2 = box_px
     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
     w, h = x2 - x1, y2 - y1
     return cx, cy, w, h
     
def process_image(img_path: str, model: Any, args: argparse.Namespace, device: torch.device, logger: logging.Logger, is_dir_mode: bool) -> Optional[Dict[str, float]]:
    t0 = sync_time()
    try:
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
    except Exception as e:
        logger.error(f"Failed loading Image: {e}")
        return None
    t_load_img = sync_time() - t0
    
    # Preprocess
    t0 = sync_time()
    input_tensor = preprocess_image(image, args.image_size, device)
    t_prep = sync_time() - t0
    
    # Inference
    t0 = sync_time()
    with torch.inference_mode():
        # Traced model gives tuple or dict depending on export method strict=False
        out = model(input_tensor)
        
    if isinstance(out, dict):
        pred_box = out['box'][0]
    else:
        # Tuple return case if trace simplified it
        pred_box = out[0][0] if isinstance(out, tuple) else out[0]
        
    score_val = 1.0 # Implicit object validation context
    logger.info(f"  Input Shape:  {list(input_tensor.shape)}")
    t_infer = sync_time() - t0
    
    t0 = sync_time()
    cx_n, cy_n, w_n, h_n = pred_box.cpu().tolist()
    
    x1_px = int(round((cx_n - w_n/2) * orig_w))
    y1_px = int(round((cy_n - h_n/2) * orig_h))
    x2_px = int(round((cx_n + w_n/2) * orig_w))
    y2_px = int(round((cy_n + h_n/2) * orig_h))
    
    x1_px = max(0, min(x1_px, orig_w - 1))
    y1_px = max(0, min(y1_px, orig_h - 1))
    x2_px = max(0, min(x2_px, orig_w - 1))
    y2_px = max(0, min(y2_px, orig_h - 1))
    box_px = [x1_px, y1_px, x2_px, y2_px]
    t_post = sync_time() - t0

    # Actions
    if not args.no_vis and args.save_vis:
        save_visualization(image, box_px, img_path, args.output_dir)
        
    json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    json_path = os.path.join(args.output_dir, json_name)
    if args.save_json:
        out_data = {
            'image_file': img_path,
            'original_width': orig_w,
            'original_height': orig_h,
            'confidence_score': score_val,
            'bounding_box_normalized': [cx_n, cy_n, w_n, h_n],
            'bounding_box_pixel': box_px,
            'timing_ms': {
                'load_image': t_load_img * 1000,
                'preprocess': t_prep * 1000,
                'inference': t_infer * 1000,
                'postprocess': t_post * 1000
            }
        }
        with open(json_path, 'w') as f:
            json.dump(out_data, f, indent=4)

    total_time = t_load_img + t_prep + t_infer + t_post

    if not is_dir_mode:
        logger.info(f"Image Path:         {img_path}")
        logger.info(f"Original Size:      {orig_w}x{orig_h}")
        logger.info(f"Input Shape:        {list(input_tensor.shape)}")
        logger.info("\n--- Results ---")
        logger.info(f"Target Confidence:  {score_val:.4f}")
        
        if score_val < args.score_threshold:
            logger.warning(f"Note: Score is below confidence threshold ({args.score_threshold})")
            
        if args.print_coordinates:
            logger.info("\nNormalized Box (cx, cy, w, h):")
            logger.info(f"  ({cx_n:.4f}, {cy_n:.4f}, {w_n:.4f}, {h_n:.4f})")
                
        logger.info("\nDecoded Pixel Bounding Box [x1, y1, x2, y2]:")
        logger.info(f"  {box_px}")
            
        if not args.no_vis and args.save_vis:
            logger.info(f"\nSaved visualization to: {os.path.join(args.output_dir, os.path.basename(img_path))}")
        if args.save_json:
            logger.info(f"Saved coarse results for refiner Stage 2 Stage to: {json_path}")

        logger.info("\n--- Edge Deployment Telemetry ---")
        logger.info(f"  Load image:  {t_load_img*1000:>6.1f} ms")
        logger.info(f"  Preprocess:  {t_prep*1000:>6.1f} ms")
        logger.info(f"  Inference:   {t_infer*1000:>6.1f} ms")
        logger.info(f"  Postprocess: {t_post*1000:>6.1f} ms")
        logger.info(f"  Total:       {total_time*1000:>6.1f} ms")

    return {
        'total': total_time,
        'prep': t_prep,
        'infer': t_infer,
        'post': t_post,
    }

def main() -> None:
    args = parse_args()
    
    if not args.output_dir:
        input_p = os.path.abspath(args.image)
        if os.path.isdir(input_p):
            input_base = input_p.rstrip('\\/')
        else:
            input_base = os.path.dirname(input_p)
        args.output_dir = input_base + "_cropped"
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging in the output dir
    log_file = os.path.join(args.output_dir, 'inference.log')
    logger = setup_logging(log_file)
    logger.info("=== TorchScript Inference Pipeline ===")
    logger.info(f"Output directory: {args.output_dir}")
        
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    # Identify images (File vs Directory)
    image_paths = []
    is_dir_mode = os.path.isdir(args.image)
    if is_dir_mode:
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
        image_paths = sorted(list(set(image_paths)))
    else:
        if not os.path.exists(args.image):
            logger.error(f"Image File Missing: {args.image}")
            return
        image_paths = [args.image]
        
    if not image_paths:
        logger.warning(f"No usable images found in: {args.image}")
        return
        
    logger.info(f"Found {len(image_paths)} images to process.")
    
    # Load Model (Once for all)
    t0 = sync_time()
    try:
        if args.pytorch:
            logger.info(f"Loading Coarse model (PyTorch): {args.model}")
            model = CoarseQuadNet().to(device)
            load_checkpoint(model, None, None, args.model, device=device)
            model.eval()
        else:
            logger.info(f"Loading Coarse model (TorchScript): {args.model}")
            model = torch.jit.load(args.model, map_location=device)
            model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    t_load_model = sync_time() - t0
    
    logger.info(f"Model Path:         {args.model}")
    logger.info(f"Model Load Time:    {t_load_model*1000:.1f} ms\n")

    accum_infer = 0.0
    accum_total = 0.0
    
    for idx, img_path in enumerate(image_paths):
        res = process_image(img_path, model, args, device, logger, is_dir_mode)
        if res:
            if is_dir_mode:
                name = os.path.basename(img_path)
                logger.info(f"  [{idx+1}/{len(image_paths)}] {name:30s} Time: {res['total']*1000:>6.1f} ms")
            accum_infer += res['infer']
            accum_total += res['total']

    # Directory Summary
    if is_dir_mode and len(image_paths) > 0:
        logger.info("\n" + "="*50)
        logger.info("   Directory Processing Summary")
        logger.info("="*50)
        logger.info(f"Total Images Processed: {len(image_paths)}")
        logger.info(f"Total Processing Time:  {accum_total:.2f} s")
        logger.info(f"Average Total Time/Img: {accum_total/len(image_paths)*1000:.1f} ms")
        logger.info(f"Average Infer Time/Img: {accum_infer/len(image_paths)*1000:.1f} ms")
        logger.info("="*50)

if __name__ == "__main__":
    main()
