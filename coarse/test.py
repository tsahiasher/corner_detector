import os
import sys
import os
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
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time

def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and a specific file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('eval_pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear redundant handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    
    # File handler
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
    parser.add_argument('--save_vis', action='store_true', default=True, help="If flag provided, saves image visualizations of predictions overlaid on targets.")
    parser.add_argument('--max_vis', type=int, default=20, help="Maximum number of visualizations to save (prioritizes the worst errors).")
    parser.add_argument('--save_csv', action='store_true', default=True, help="If flag provided, saves per-image metrics to a CSV file.")
    parser.add_argument('--report_worst_k', type=int, default=10, help="Number of worst-performing images to highlight in the final report.")
    return parser.parse_args()

def save_comparative_visualization(image_tensor: torch.Tensor, pred_norm: list, target_norm: list, img_path: str, out_dir: str) -> None:
    """Draws predicted (orange) and target (green) corners onto the image and saves it.
    
    Args:
        image_tensor (torch.Tensor): Unnormalized/normalized image tensor from dataloader.
        pred_norm (list): Array of [[x, y], ...] predicted normalized points.
        target_norm (list): Array of [[x, y], ...] ground truth normalized points.
        img_path (str): Original file path used for naming the output file.
        out_dir (str): Destination directory for the drawn preview.
    """
    try:
        import cv2
        import numpy as np
        from common.transforms import denormalize_image
    except ImportError as e:
        logging.getLogger('eval_pipeline').warning(f"Skipping visualization plotting, missing dependency: {e}")
        return
    
    # Render normalized tensor back into pixel values
    img = denormalize_image(image_tensor.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w, _ = img.shape
    
    pred_px = (np.array(pred_norm) * [w, h]).astype(np.int32)
    target_px = (np.array(target_norm) * [w, h]).astype(np.int32)
    
    def draw_poly(img_ref, pts, color, thickness):
        for i in range(4):
            cv2.circle(img_ref, tuple(pts[i]), 5, color, -1)
            next_i = (i + 1) % 4
            cv2.line(img_ref, tuple(pts[i]), tuple(pts[next_i]), color, thickness)
            
    # Draw Ground Truth Context (Green)
    draw_poly(img, target_px, (0, 255, 0), 2)
    # Draw Predicted Context (Orange/Red)
    draw_poly(img, pred_px, (0, 165, 255), 2)
    
    # Legend overlay
    cv2.putText(img, "Target (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Predict (Orange)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"vis_{name}")
    cv2.imwrite(out_path, img)

def main() -> None:
    args = parse_args()
    
    # Configure Directory & Logging
    if args.run_dir:
        run_dir = args.run_dir
    else:
        # Infer run namespace cleanly from structural convention "runs/{name}/checkpoints/best.pt"
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
    logger.info(f"Evaluation mapped to run directory: {run_dir}")
    
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    logger.info("Initializing context bindings mapped validations streams...")
    val_dataset = COCOValDataset(args.val_images, args.val_annotations, image_size=args.input_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load Model (Dynamically identifying pure PyTorch vs TorchScript footprints)
    is_torchscript = False
    try:
        model = torch.jit.load(args.weights, map_location=device)
        is_torchscript = True
        logger.info(f"Successfully loaded model layout mappings as native JIT TorchScript format.")
    except Exception:
        model = CoarseQuadNet().to(device)
        load_checkpoint(model, None, None, args.weights, device=device)
        logger.info(f"Loaded eager standard PyTorch object model checkpoint boundary structures.")
        
    model.eval()
    
    # Telemetry metrics arrays mapped lists limits mappings
    results = []
    total_infer_time = 0.0
    total_num_images = 0
    t_start_e2e = sync_time()
    
    logger.info("Commencing Evaluation Vector Space Map Loops...")
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating Images", leave=False)
        for batch_idx, batch in enumerate(val_pbar):
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            target_corners = batch['corners']
            paths = batch['img_path']
            
            # Handle old and new orig_size formats correctly
            orig_size = batch.get('orig_size')
            if isinstance(orig_size, list) and len(orig_size) == 2:
                w_list, h_list = orig_size
            elif isinstance(orig_size, tuple) and len(orig_size) == 2:
                w_list, h_list = orig_size
            else:
                w_list = [384] * images.size(0)
                h_list = [384] * images.size(0)

            # Execution block
            t_infer = sync_time()
            score, pred_corners = model(images)
            infer_time = sync_time() - t_infer
            
            total_infer_time += infer_time
            total_num_images += images.size(0)
            
            for b_idx in range(images.size(0)):
                try:
                    w = float(w_list[b_idx])
                    h = float(h_list[b_idx])
                except Exception:
                    w, h = float(w_list[0]), float(h_list[0]) # Fallback to a scalar width height block
                    
                p_norm = pred_corners[b_idx]
                t_norm = target_corners[b_idx]
                
                # Render to original pixel geometry values
                p_px = p_norm * torch.tensor([w, h], device=device)
                t_px = t_norm * torch.tensor([w, h], device=device)
                
                # Calculate absolute coordinate errors
                diff = p_px - t_px
                dist = torch.norm(diff, dim=-1)  # [4]
                
                results.append({
                    'img_path': paths[b_idx] if isinstance(paths, list) else paths[b_idx],
                    'w': w,
                    'h': h,
                    'err_tl': dist[0].item(),
                    'err_tr': dist[1].item(),
                    'err_br': dist[2].item(),
                    'err_bl': dist[3].item(),
                    'mean_err': dist.mean().item(),
                    'p_norm': p_norm.cpu().tolist(),
                    't_norm': t_norm.cpu().tolist(),
                    'img_tensor': images[b_idx].cpu()
                })
                
    total_eval_time = sync_time() - t_start_e2e
    avg_infer_ms = (total_infer_time / total_num_images) * 1000
    avg_e2e_ms = (total_eval_time / total_num_images) * 1000

    # Aggregate metric values into robust evaluation tensors
    all_mean_errs = torch.tensor([r['mean_err'] for r in results])
    err_tl = torch.tensor([r['err_tl'] for r in results])
    err_tr = torch.tensor([r['err_tr'] for r in results])
    err_br = torch.tensor([r['err_br'] for r in results])
    err_bl = torch.tensor([r['err_bl'] for r in results])
    
    # Sort results by the absolute worst errors initially so failure analysis limits bounds are properly pushed
    results_sorted = sorted(results, key=lambda x: x['mean_err'], reverse=True)

    logger.info("\n" + "="*50)
    logger.info("=== EVALUATION RESULTS ===")
    logger.info("="*50)
    logger.info(f"Model path: {args.weights}")
    logger.info(f"Total evaluated images: {total_num_images}")
    
    logger.info("\n--- Timing ---")
    logger.info(f"Total loop time: {total_eval_time:.2f} s")
    logger.info(f"Total model inference time: {total_infer_time:.2f} s")
    logger.info(f"Avg model inference time (device): {avg_infer_ms:.2f} ms")
    logger.info(f"Avg end-to-end loop time per image: {avg_e2e_ms:.2f} ms")
    
    logger.info("\n--- Pixel errors ---")
    logger.info(f"Mean pixel error: {all_mean_errs.mean().item():.3f} px")
    logger.info(f"Median pixel error: {all_mean_errs.median().item():.3f} px")
    
    logger.info("\n--- Per-Corner Mean errors (px) ---")
    logger.info(f"Top-Left: {err_tl.mean().item():.3f}")
    logger.info(f"Top-Right: {err_tr.mean().item():.3f}")
    logger.info(f"Bottom-Right: {err_br.mean().item():.3f}")
    logger.info(f"Bottom-Left: {err_bl.mean().item():.3f}")

    logger.info("\n--- Spatial Accuracy ---")
    thresholds = [2, 3, 5, 10]
    for t in thresholds:
        logger.info(f"<{t} px error accuracy: {(all_mean_errs < t).float().mean().item() * 100:.2f}%")
        
    logger.info("\n--- Failure Analysis (Worst-K results) ---")
    for k in range(min(args.report_worst_k, len(results_sorted))):
        fault = results_sorted[k]
        logger.info(f"[{k+1}] Error: {fault['mean_err']:>6.2f} px | File: {os.path.basename(fault['img_path'])}")

    # Dump CSV mapping arrays struct mapping formats parameters
    if args.save_csv:
        csv_path = os.path.join(run_dir, 'evaluation_results.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['img_path', 'mean_err', 'err_tl', 'err_tr', 'err_br', 'err_bl'])
                for res in results_sorted:
                    writer.writerow([res['img_path'], res['mean_err'], res['err_tl'], res['err_tr'], res['err_br'], res['err_bl']])
            logger.info(f"\nSaved CSV evaluation results: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV output: {e}")

    # Generate Image Visualizations
    if args.save_vis:
        vis_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        drawn = 0
        logger.info(f"\nRendering visualizations for up to {args.max_vis} images to: {vis_dir}")
        for res in results_sorted:
            if drawn >= args.max_vis:
                break
            save_comparative_visualization(
                res['img_tensor'], 
                res['p_norm'], 
                res['t_norm'], 
                res['img_path'], 
                vis_dir
            )
            drawn += 1

if __name__ == "__main__":
    main()
