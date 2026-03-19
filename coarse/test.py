import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import logging
import csv
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coarse.models.coarse_quad_net import CoarseQuadNet
from coarse.datasets.coco_val_dataset import COCOValDataset
from common.checkpoint import load_checkpoint
from common.seed import set_seed
from common.metrics import compute_patch_recall
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.visualization import save_indexed_visualization


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and a specific file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('eval_pipeline')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(log_file)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    logger.info(f"Evaluation logging initialized. Writing logs to {log_file}")
    return logger


def parse_args() -> argparse.Namespace:
    """Parses command line configuration arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Coarse Quad network (Stage 1).")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained model checkpoint (.pt) or TorchScript model.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Directory containing validation images.")
    parser.add_argument('--val_annotations', type=str, default='../crop-dataset-eitan-yolo/annotations/val.json', help="Path to the COCO validation annotations JSON.")
    parser.add_argument('--input_size', type=int, default=384, help="Model input resolution (square dim).")
    parser.add_argument('--batch_size', type=int, default=1, help="Evaluation batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of PyTorch data loading workers.")
    add_device_args(parser, default='auto')
    parser.add_argument('--run_dir', type=str, default='', help="Explicit run directory. If empty, inferred from weights path.")
    parser.add_argument('--save_vis', action='store_true', default=True, help="Saves image visualizations of predictions overlaid on targets.")
    parser.add_argument('--save_patch_vis', action='store_true', default=True, help="Saves patch-recall diagnostic visualizations showing 64/80/96px patches.")
    parser.add_argument('--max_vis', type=int, default=20, help="Maximum number of visualizations to save (prioritizes the worst errors).")
    parser.add_argument('--save_csv', action='store_true', default=True, help="Saves per-image metrics to a CSV file.")
    parser.add_argument('--report_worst_k', type=int, default=10, help="Number of worst-performing images to highlight in the final report.")
    return parser.parse_args()




def save_patch_diagnostic(image_tensor: torch.Tensor, pred_norm: list, target_norm: list,
                          img_path: str, out_dir: str, orig_w: float, orig_h: float) -> None:
    """Draws refinement patches (64, 80, 96px) centered on predicted corners and shows
    whether each GT corner falls inside the patch.

    Args:
        image_tensor (torch.Tensor): Image tensor from the dataloader.
        pred_norm (list): Predicted normalized corner coordinates.
        target_norm (list): Ground truth normalized corner coordinates.
        img_path (str): Original file path.
        out_dir (str): Destination directory.
        orig_w (float): Original image width in pixels.
        orig_h (float): Original image height in pixels.
    """
    try:
        import cv2
        import numpy as np
        from common.transforms import denormalize_image
    except ImportError:
        return

    img = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    pred_px = np.array(pred_norm) * [orig_w, orig_h]
    target_px = np.array(target_norm) * [orig_w, orig_h]

    # Scale patches to the display image coordinate space
    scale_x = w / orig_w
    scale_y = h / orig_h

    pred_disp = (pred_px * [scale_x, scale_y]).astype(np.int32)
    target_disp = (target_px * [scale_x, scale_y]).astype(np.int32)

    patch_configs = [
        (64, (255, 200, 0)),   # cyan-ish
        (80, (200, 150, 0)),   # medium blue
        (96, (150, 100, 0)),   # dark blue
    ]
    corner_labels = ["TL", "TR", "BR", "BL"]

    for patch_size, color in patch_configs:
        half_w = int((patch_size / 2) * scale_x)
        half_h = int((patch_size / 2) * scale_y)
        for i in range(4):
            cx, cy = int(pred_disp[i][0]), int(pred_disp[i][1])
            top_left = (cx - half_w, cy - half_h)
            bottom_right = (cx + half_w, cy + half_h)
            cv2.rectangle(img, top_left, bottom_right, color, 1)

    # Draw GT as green dots, predicted as orange dots
    for i in range(4):
        cv2.circle(img, tuple(target_disp[i]), 4, (0, 255, 0), -1)
        cv2.circle(img, tuple(pred_disp[i]), 4, (0, 165, 255), -1)

        # Check recall for 64px patch (the primary target)
        dx = abs(pred_px[i][0] - target_px[i][0])
        dy = abs(pred_px[i][1] - target_px[i][1])
        recalled = dx <= 32 and dy <= 32
        status = "OK" if recalled else "MISS"
        status_color = (0, 255, 0) if recalled else (0, 0, 255)
        cv2.putText(img, f"{corner_labels[i]}:{status}", (int(pred_disp[i][0]) + 8, int(pred_disp[i][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    cv2.putText(img, "64px(cyan) 80px(med) 96px(dark)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"patch_{name}")
    cv2.imwrite(out_path, img)


def main() -> None:
    """Main evaluation function with patch recall diagnostics."""
    args = parse_args()

    # Configure run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        weights_dir = os.path.dirname(os.path.abspath(args.weights))
        parent_dir = os.path.dirname(weights_dir)
        if os.path.basename(weights_dir) == 'checkpoints':
            run_dir = parent_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join('.', 'runs', f"eval_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(os.path.join(log_dir, 'eval.log'))
    logger.info(f"Evaluation run directory: {run_dir}")

    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    logger.info("Initializing validation dataset...")
    val_dataset = COCOValDataset(args.val_images, args.val_annotations, image_size=args.input_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load Model
    is_torchscript = False
    try:
        model = torch.jit.load(args.weights, map_location=device)
        is_torchscript = True
        logger.info("Loaded TorchScript model.")
    except Exception:
        model = CoarseQuadNet().to(device)
        load_checkpoint(model, None, None, args.weights, device=device)
        logger.info("Loaded eager PyTorch checkpoint.")

    model.eval()

    results = []
    total_infer_time = 0.0
    total_num_images = 0
    # Accumulate patch recalls
    patch_recall_accum = {64: [], 80: [], 96: []}
    t_start_e2e = sync_time()

    logger.info("Running evaluation...")
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(val_pbar):
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            target_corners = batch['corners']
            paths = batch['img_path']

            w_list = batch.get('orig_width', torch.full((images.size(0),), 384.0, device=device))
            h_list = batch.get('orig_height', torch.full((images.size(0),), 384.0, device=device))

            t_infer = sync_time()
            score, pred_corners = model(images)
            infer_time = sync_time() - t_infer

            total_infer_time += infer_time
            total_num_images += images.size(0)

            for b_idx in range(images.size(0)):
                try:
                    w = w_list[b_idx].item()
                    h = h_list[b_idx].item()
                except Exception:
                    w, h = 384.0, 384.0

                p_norm = pred_corners[b_idx]
                t_norm = target_corners[b_idx]

                # Pixel errors
                p_px = p_norm * torch.tensor([w, h], device=device)
                t_px = t_norm * torch.tensor([w, h], device=device)
                diff = p_px - t_px
                dist = torch.norm(diff, dim=-1)

                # Patch recall
                pr = compute_patch_recall(
                    p_norm.unsqueeze(0), t_norm.unsqueeze(0), w, h,
                    patch_sizes=(64, 80, 96)
                )
                for ps in (64, 80, 96):
                    patch_recall_accum[ps].append(pr[f'patch_recall_{ps}px'])

                results.append({
                    'img_path': paths[b_idx] if isinstance(paths, list) else paths[b_idx],
                    'w': w, 'h': h,
                    'err_tl': dist[0].item(),
                    'err_tr': dist[1].item(),
                    'err_br': dist[2].item(),
                    'err_bl': dist[3].item(),
                    'mean_err': dist.mean().item(),
                    'p_norm': p_norm.cpu().tolist(),
                    't_norm': t_norm.cpu().tolist(),
                    'img_tensor': images[b_idx].cpu(),
                    'pr_64': pr['patch_recall_64px'],
                    'pr_80': pr['patch_recall_80px'],
                    'pr_96': pr['patch_recall_96px'],
                })

    total_eval_time = sync_time() - t_start_e2e
    avg_infer_ms = (total_infer_time / total_num_images) * 1000 if total_num_images > 0 else 0
    avg_e2e_ms = (total_eval_time / total_num_images) * 1000 if total_num_images > 0 else 0

    all_mean_errs = torch.tensor([r['mean_err'] for r in results])
    err_tl = torch.tensor([r['err_tl'] for r in results])
    err_tr = torch.tensor([r['err_tr'] for r in results])
    err_br = torch.tensor([r['err_br'] for r in results])
    err_bl = torch.tensor([r['err_bl'] for r in results])

    results_sorted = sorted(results, key=lambda x: x['mean_err'], reverse=True)

    # Aggregate patch recall
    avg_pr = {}
    for ps in (64, 80, 96):
        vals = patch_recall_accum[ps]
        avg_pr[ps] = sum(vals) / len(vals) if vals else 0.0

    # === Reporting ===
    logger.info("\n" + "="*50)
    logger.info("=== EVALUATION RESULTS ===")
    logger.info("="*50)
    logger.info(f"Model path: {args.weights}")
    logger.info(f"Total evaluated images: {total_num_images}")

    logger.info("\n--- Timing ---")
    logger.info(f"Total loop time: {total_eval_time:.2f} s")
    logger.info(f"Total model inference time: {total_infer_time:.2f} s")
    logger.info(f"Avg model inference time: {avg_infer_ms:.2f} ms")
    logger.info(f"Avg end-to-end time per image: {avg_e2e_ms:.2f} ms")

    logger.info("\n--- Pixel errors ---")
    logger.info(f"Mean pixel error: {all_mean_errs.mean().item():.3f} px")
    logger.info(f"Median pixel error: {all_mean_errs.median().item():.3f} px")

    logger.info("\n--- Per-Corner Mean errors (px) ---")
    logger.info(f"Top-Left: {err_tl.mean().item():.3f}")
    logger.info(f"Top-Right: {err_tr.mean().item():.3f}")
    logger.info(f"Bottom-Right: {err_br.mean().item():.3f}")
    logger.info(f"Bottom-Left: {err_bl.mean().item():.3f}")

    logger.info("\n--- Patch Recall (Stage 2 readiness) ---")
    logger.info(f"64x64 patch recall: {avg_pr[64]:.1f}%")
    logger.info(f"80x80 patch recall: {avg_pr[80]:.1f}%")
    logger.info(f"96x96 patch recall: {avg_pr[96]:.1f}%")

    logger.info("\n--- Spatial Accuracy ---")
    thresholds = [2, 3, 5, 10]
    for t in thresholds:
        logger.info(f"<{t} px accuracy: {(all_mean_errs < t).float().mean().item() * 100:.2f}%")

    logger.info("\n--- Failure Analysis (Worst-K) ---")
    for k in range(min(args.report_worst_k, len(results_sorted))):
        fault = results_sorted[k]
        logger.info(f"[{k+1}] Error: {fault['mean_err']:>6.2f} px | 64px recall: {fault['pr_64']:.0f}% | File: {os.path.basename(fault['img_path'])}")

    # CSV output
    if args.save_csv:
        csv_path = os.path.join(run_dir, 'evaluation_results.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['img_path', 'mean_err', 'err_tl', 'err_tr', 'err_br', 'err_bl', 'pr_64', 'pr_80', 'pr_96'])
                for res in results_sorted:
                    writer.writerow([res['img_path'], res['mean_err'], res['err_tl'], res['err_tr'],
                                     res['err_br'], res['err_bl'], res['pr_64'], res['pr_80'], res['pr_96']])
            logger.info(f"\nSaved CSV: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    # Standard visualizations
    if args.save_vis:
        vis_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        drawn = 0
        logger.info(f"\nRendering visualizations for up to {args.max_vis} images to: {vis_dir}")
        for res in results_sorted:
            if drawn >= args.max_vis:
                break
            pred_t = torch.tensor(res['p_norm'], dtype=torch.float32)
            tgt_t = torch.tensor(res['t_norm'], dtype=torch.float32)
            name = os.path.splitext(os.path.basename(res['img_path']))[0]
            save_path = os.path.join(vis_dir, f"vis_{name}.jpg")
            save_indexed_visualization(
                res['img_tensor'], pred_t, tgt_t,
                res['w'], res['h'], save_path,
                img_path=res['img_path']
            )
            drawn += 1

    # Patch diagnostic visualizations
    if args.save_patch_vis:
        patch_dir = os.path.join(run_dir, "patch_diagnostics")
        os.makedirs(patch_dir, exist_ok=True)
        drawn = 0
        logger.info(f"Rendering patch diagnostics for up to {args.max_vis} images to: {patch_dir}")
        for res in results_sorted:
            if drawn >= args.max_vis:
                break
            save_patch_diagnostic(
                res['img_tensor'], res['p_norm'], res['t_norm'],
                res['img_path'], patch_dir, res['w'], res['h']
            )
            drawn += 1


if __name__ == "__main__":
    main()
