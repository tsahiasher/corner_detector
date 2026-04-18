"""Standalone Stage 2.5 inference: card orientation classification.

Usage:
    python orient/run_torchscript_image.py \\
        --orient_model orient/runs/<run>/checkpoints/orient_net.pt \\
        --boundingbox_model boundingbox/runs/<run>/checkpoints/best.pt \\
        --input /path/to/image_or_dir \\
        [--pytorch] [--output_dir results/]

The script:
  1. Runs the BoundingBox model to get 4 corners.
  2. Warps the card to a 128x128 canonical crop using the GT homography.
  3. Runs OrientNet to classify rotation (0°/90°/180°/270°).
  4. Applies the inverse cyclic shift so corners[0] is the physical TL.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import argparse
import logging
import time
from typing import List, Optional, Any

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from common.device import add_device_args, resolve_device, log_device_info, sync_time
from common.checkpoint import load_checkpoint
from common.geometry import compute_homography, warp_image
from boundingbox.models.boundingbox_quad_net import BoundingBoxQuadNet
from orient.models.orient_net import OrientNet

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
ORIENT_LABELS = {0: '  0°', 1: ' 90°', 2: '180°', 3: '270°'}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('orient_inference')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OrientNet standalone inference.')
    parser.add_argument('--orient_model', type=str, required=True,
                        help='Path to OrientNet model (.pt).')
    parser.add_argument('--boundingbox_model', '--boundingbox_mode', type=str, required=True,
                        help='Path to BoundingBox model (.pt).')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to image or directory.')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Results directory. Defaults to <input>_orient.')
    parser.add_argument('--boundingbox_size', type=int, default=384)
    parser.add_argument('--crop_size',  type=int, default=128,
                        help='Canonical crop size fed to OrientNet.')
    parser.add_argument('--save_json', action='store_true', default=True)
    parser.add_argument('--no_vis',   action='store_true', default=False)
    parser.add_argument('--pytorch',  action='store_true',
                        help='Load PyTorch checkpoints instead of TorchScript.')
    add_device_args(parser, default='cpu')
    return parser.parse_args()


def preprocess_boundingbox(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    img = TF.resize(image, [size, size])
    t   = TF.to_tensor(img)
    t   = TF.normalize(t, MEAN, STD)
    return t.unsqueeze(0).to(device)


def warp_crop(img_np: np.ndarray, corners_px: List[List[float]],
              crop_size: int) -> np.ndarray:
    """Warps the card quad to a canonical [crop_size x crop_size] square."""
    s    = float(crop_size)
    dst  = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    src  = np.array(corners_px, dtype=np.float32)
    H    = compute_homography(src, dst)
    return warp_image(img_np, H, (crop_size, crop_size))


def preprocess_orient(crop_np: np.ndarray, device: torch.device) -> torch.Tensor:
    pil = Image.fromarray(crop_np)
    t   = TF.to_tensor(pil)
    t   = TF.normalize(t, MEAN, STD)
    return t.unsqueeze(0).to(device)


def apply_orientation_shift(corners: List[List[float]], shift: int) -> List[List[float]]:
    """Cyclically rotates the corners list so that corners[0] is always the physical TL."""
    n = len(corners)
    return [corners[(i + shift) % n] for i in range(n)]


def draw_result(img_bgr: np.ndarray, corners_px: List[List[float]],
                orient_class: int) -> np.ndarray:
    vis       = img_bgr.copy()
    h, w      = vis.shape[:2]
    thickness = max(1, min(w, h) // 400)
    radius    = max(4, min(w, h) // 150)
    fs        = max(0.4, min(w, h) / 1000.0)
    pts       = [tuple(map(int, p)) for p in corners_px]
    color     = (0, 200, 80)  # green for physical-ordered corners

    for i in range(4):
        cv2.line(vis, pts[i], pts[(i + 1) % 4], color, thickness * 2)
        cv2.circle(vis, pts[i], radius, color, -1)
        cv2.putText(vis, str(i), (pts[i][0] + 10, pts[i][1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, thickness)

    label = f'Orient: {ORIENT_LABELS[orient_class].strip()}'
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4,
                (0, 255, 255), thickness + 1)
    return vis


def process_image(img_path: str, boundingbox_model: Any, orient_model: Any,
                  args: argparse.Namespace, device: torch.device,
                  logger: logging.Logger):
    t0 = sync_time()

    try:
        pil_img  = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_bgr  = cv2.imread(img_path)
        img_rgb  = np.array(pil_img)
    except Exception as e:
        logger.error(f'Cannot load {img_path}: {e}')
        return None

    # ── Stage 1: BoundingBox ─────────────────────────────────────────────────
    t1 = sync_time()
    boundingbox_in = preprocess_boundingbox(pil_img, args.boundingbox_size, device)
    with torch.inference_mode():
        boundingbox_out = boundingbox_model(boundingbox_in)
    corners_norm = boundingbox_out['corners'][0].cpu().tolist()   # [4, 2]
    corners_px   = [[nx * orig_w, ny * orig_h] for nx, ny in corners_norm]
    t_boundingbox = sync_time() - t1

    # ── Stage 2.5: Orient ────────────────────────────────────────────────
    t2 = sync_time()
    crop_np   = warp_crop(img_rgb, corners_px, args.crop_size)
    orient_in = preprocess_orient(crop_np, device)
    with torch.inference_mode():
        logits = orient_model(orient_in)
    orient_class = int(logits.argmax(dim=-1).item())
    # Apply inverse shift to make corners[0] the physical TL
    corners_final = apply_orientation_shift(corners_px, orient_class)
    t_orient = sync_time() - t2

    t_total = sync_time() - t0
    logger.info(f'  {os.path.basename(img_path):30s} | Orient: {ORIENT_LABELS[orient_class].strip():4s} '
                f'| BoundingBox: {t_boundingbox*1000:.1f}ms | Orient: {t_orient*1000:.1f}ms | Total: {t_total*1000:.1f}ms')

    # ── Visualisation ────────────────────────────────────────────────────
    if not args.no_vis:
        vis = draw_result(img_bgr, corners_final, orient_class)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(img_path)), vis)

    # ── JSON ─────────────────────────────────────────────────────────────
    if args.save_json:
        result = {
            'image': img_path,
            'orient_class': orient_class,
            'orient_label': ORIENT_LABELS[orient_class].strip(),
            'boundingbox_corners': corners_px,
            'final_corners':  corners_final,
            'times_ms': {'boundingbox': t_boundingbox*1000, 'orient': t_orient*1000, 'total': t_total*1000},
        }
        json_name = os.path.splitext(os.path.basename(img_path))[0] + '.json'
        with open(os.path.join(args.output_dir, json_name), 'w') as f:
            json.dump(result, f, indent=4)

    return t_total


def main() -> None:
    args   = parse_args()
    logger = setup_logging()

    if not args.output_dir:
        base = args.input.rstrip('/\\') if os.path.isdir(args.input) else \
               os.path.dirname(os.path.abspath(args.input))
        args.output_dir = base + '_orient'
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    # Load models
    if args.pytorch:
        boundingbox_model = BoundingBoxQuadNet().to(device)
        load_checkpoint(boundingbox_model, None, None, args.boundingbox_model, device=device)
        boundingbox_model.eval()
        orient_model = OrientNet(num_classes=4).to(device)
        load_checkpoint(orient_model, None, None, args.orient_model, device=device)
        orient_model.eval()
    else:
        boundingbox_model = torch.jit.load(args.boundingbox_model, map_location=device)
        boundingbox_model.eval()
        orient_model = torch.jit.load(args.orient_model, map_location=device)
        orient_model.eval()

    image_paths: List[str] = []
    if os.path.isdir(args.input):
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths = sorted(set(image_paths))
    else:
        image_paths = [args.input]

    logger.info(f'Processing {len(image_paths)} images...')
    t_start = time.time()
    ok = 0
    for p in image_paths:
        t = process_image(p, boundingbox_model, orient_model, args, device, logger)
        if t is not None:
            ok += 1

    elapsed = time.time() - t_start
    logger.info(f'\nProcessed {ok}/{len(image_paths)} in {elapsed:.2f}s '
                f'({elapsed/max(1,ok)*1000:.1f}ms/img avg). Results → {args.output_dir}')


if __name__ == '__main__':
    main()
