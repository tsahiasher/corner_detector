import os
import sys
import os
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
from common.visualization import save_diagnostic_visualization
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
    """Parses command line config arguments for TorchScript inference."""
    parser = argparse.ArgumentParser(description="Run TorchScript coarse corner detector on single image or directory.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str, required=True, help="Path to the exported TorchScript model (.pt).")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image or directory of images.")
    parser.add_argument('--output_dir', type=str, default='', help="Directory to save execution artifacts (visualizations, JSON).")
    add_device_args(parser, default='cpu')
    parser.add_argument('--min_size', type=int, default=800, help="Minimum image dimension for resizing.")
    parser.add_argument('--max_size', type=int, default=1333, help="Maximum image dimension for resizing.")
    
    parser.add_argument('--save_json', action='store_true', default=True, help="Flag to export explicit numeric result limits coordinates to a JSON payload.")
    parser.add_argument('--save_vis', action='store_true', default=True, help="Flag to generate and save a visualization overlaid upon the absolute original image scale.")
    parser.add_argument('--no_vis', action='store_true', default=False, help="Suppresses any visual artifact generations.")
    parser.add_argument('--pytorch', action='store_true', help="Load PyTorch checkpoint (.pt) instead of TorchScript model.")
    
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Minimum global pool inference bound target confidence required for bounds map output acceptance.")
    parser.add_argument('--print_coordinates', action='store_true', default=True, help="Dumps the raw parsed map scale geometry structs vector payloads explicitly into console bounds.")
    parser.add_argument('--run_dir', type=str, default='', help="Optional run directory space to group structural logs instead of using global standalone namespaces.")
    
    return parser.parse_args()

def preprocess_image(image: Image.Image, min_size: int, max_size: int, device: torch.device) -> torch.Tensor:
    from common.transforms import ResizeMinMax
    resizer = ResizeMinMax(min_size=min_size, max_size=max_size)
    # ResizeMinMax expects (image, keypoints), we pass empty keypoints
    img_resized, _ = resizer(image, [])
    
    tensor = TF.to_tensor(img_resized)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = TF.normalize(tensor, mean=mean, std=std)
    tensor = tensor.unsqueeze(0).to(device)
    
    # Ensure dimensions are multiples of 32 for the FPN backbone
    h, w = tensor.shape[2:]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    if new_h != h or new_w != w:
        tensor = torch.nn.functional.pad(tensor, (0, new_w - w, 0, new_h - h))
        
    return tensor

def save_visualization(image: Image.Image, corners_norm: list, corners_px: List[List[int]],
                       img_path: str, output_dir: str) -> None:
    """Draws indexed predicted corners on the original-scale image and saves it.

    Args:
        image (Image.Image): Original PIL image.
        corners_norm (list): Normalized corner coordinates [[x,y], ...].
        corners_px (List[List[int]]): Pixel-space corner coordinates.
        img_path (str): Original image path.
        output_dir (str): Destination directory.
        input_size (int): Model input resolution used for preprocessing.
    """
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
    label_names = ['TL', 'TR', 'BR', 'BL']
    color_pred = (0, 165, 255)  # Orange
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 800.0)
    thickness = max(1, int(min(w, h) / 400))
    radius = max(4, int(min(w, h) / 150))

    # Draw polygon and indexed labels
    for i in range(4):
        pt = tuple(corners_px[i])
        next_pt = tuple(corners_px[(i + 1) % 4])
        cv2.line(img_np, pt, next_pt, (0, 255, 0), thickness)
        cv2.circle(img_np, pt, radius, color_pred, -1)
        label = str(i)
        cv2.putText(img_np, label, (pt[0] + 12, pt[1] + 12), font, font_scale, color_pred, thickness)

    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, base_name)
    cv2.imwrite(out_path, img_np)

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
    input_tensor = preprocess_image(image, args.min_size, args.max_size, device)
    t_prep = sync_time() - t0
    
    # Inference
    t0 = sync_time()
    with torch.inference_mode():
        out_dict = model(input_tensor)
    
    if 'score' not in out_dict:
        return None
        
    logger.info(f"  Input Shape:  {list(input_tensor.shape)}")
    score_val = out_dict['score'][0, 0].item()
    corners_norm = out_dict['corners'][0].cpu().tolist()
    t_infer = sync_time() - t0
    
    t0 = sync_time()
    corners_px = []
    for (nx, ny) in corners_norm:
        x_px = int(round(nx * orig_w))
        y_px = int(round(ny * orig_h))
        x_px = max(0, min(x_px, orig_w - 1))
        y_px = max(0, min(y_px, orig_h - 1))
        corners_px.append([x_px, y_px])
    t_post = sync_time() - t0

    # Actions
    if not args.no_vis and args.save_vis:
        save_visualization(image, corners_norm, corners_px, img_path, args.output_dir)
        
        # Performance Boost v2: Save diagnostic triple-view if mask/edges available
        if 'mask' in out_dict and 'edges' in out_dict:
            # Dummy GT for inference-only vis
            dummy_gt = torch.tensor(corners_norm, device=device)
            save_diagnostic_visualization(
                input_tensor[0], out_dict['corners'][0], dummy_gt,
                out_dict['mask'][0], out_dict['edges'][0],
                img_path, args.output_dir
            )
        
    json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    json_path = os.path.join(args.output_dir, json_name)
    if args.save_json:
        out_data = {
            'image_file': img_path,
            'original_width': orig_w,
            'original_height': orig_h,
            'confidence_score': score_val,
            'corners_normalized': corners_norm,
            'corners_pixel': corners_px,
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
            logger.info("\nNormalized coordinates ([0,1]):")
            for i, (nx, ny) in enumerate(corners_norm):
                logger.info(f"  P{i+1}: ({nx:.4f}, {ny:.4f})")
                
        logger.info("\nDecoded Pixel coordinates:")
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (px, py) in enumerate(corners_px):
            logger.info(f"  {labels[i]:13s}: ({px}, {py})")
            
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
    t_start_total = sync_time()
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
