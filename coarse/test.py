import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import csv
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from coarse.models.coarse_quad_net import CoarseQuadNet
from coarse.datasets.coco_val_dataset import COCOValDataset
from common.checkpoint import load_checkpoint
from common.seed import set_seed
from common.metrics import calculate_accuracy_metrics, compute_patch_recall
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.visualization import draw_quadrilateral, save_diagnostic_visualization


def setup_logging(log_file: str) -> logging.Logger:
    """Configures evaluation logging."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('coarse_eval')
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
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Coarse Quad network (Stage 1).")
    parser.add_argument('--weights', type=str, required=True, help="Path to checkpoint (.pt) or TorchScript.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Val images dir.")
    parser.add_argument('--val_json', type=str, default='../crop-dataset-eitan-yolo/annotations/val.json', help="Val COCO JSON.")
    parser.add_argument('--image_size', type=int, default=384, help="Input size.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for eval.")
    add_device_args(parser, default='auto')
    parser.add_argument('--save_csv', action='store_true', default=True, help="Save per-image results.")
    parser.add_argument('--save_vis', action='store_true', default=True, help="Save visualizations.")
    parser.add_argument('--max_vis', type=int, default=30, help="Max visualizations.")
    return parser.parse_args()


def save_diagnostic_visualization(image_t: torch.Tensor, pred_corners: torch.Tensor, gt_corners: torch.Tensor,
                                 mask_t: torch.Tensor, edge_t: torch.Tensor,
                                 img_path: str, out_dir: str) -> None:
    """Saves a multi-panel visualization of geometric anchoring."""
    try:
        import cv2
        from common.transforms import denormalize_image
    except ImportError:
        return

    # Original Image with Quads
    img = denormalize_image(image_t.cpu()).permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert corners to pixel space
    h, w = img.shape[:2]
    pred_px = (pred_corners.cpu().numpy() * [w, h]).astype(int)
    gt_px = (gt_corners.cpu().numpy() * [w, h]).astype(int)

    # Draw GT (blue) and Pred (orange)
    cv2.polylines(img, [gt_px], True, (255, 0, 0), 2)
    cv2.polylines(img, [pred_px], True, (0, 165, 255), 2)

    # Masks (48x48 upsampled)
    mask = (mask_t[0].cpu().numpy() * 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Edges (48x48 upsampled)
    edges = (edge_t[0].cpu().numpy() * 255).astype(np.uint8)
    edges = cv2.resize(edges, (w, h))
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack results
    combined = np.hstack([img, mask_bgr, edges_bgr])
    name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_dir, f"diag_{name}"), combined)


def main() -> None:
    args = parse_args()
    weights_dir = os.path.dirname(os.path.abspath(args.weights))
    parent_dir = os.path.dirname(weights_dir)
    if os.path.basename(weights_dir) == 'checkpoints':
        run_dir = parent_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join('.', 'runs', f"eval_{timestamp}")
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'visualizations'), exist_ok=True)

    logger = setup_logging(os.path.join(run_dir, 'logs', 'eval.log'))
    logger.info("=== Coarse Evaluation (Multi-task Geometric) ===")

    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    val_dataset = COCOValDataset(args.val_images, args.val_json, image_size=args.image_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Model
    try:
        model = torch.jit.load(args.weights, map_location=device)
        logger.info("Loaded TorchScript model.")
    except Exception:
        model = CoarseQuadNet().to(device)
        load_checkpoint(model, None, None, args.weights, device=device)
        logger.info("Loaded eager checkpoint.")

    model.eval()
    results = []
    all_errors_px = []
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            out = model(batch['image'])
            gt_corners = batch['corners']
            
            # Use structured corners for precision metrics
            pred = out['corners']

            # Compute pixel errors
            diff = (pred - gt_corners).abs()
            diff[:, :, 0] *= batch['orig_width'].view(-1, 1)
            diff[:, :, 1] *= batch['orig_height'].view(-1, 1)
            dist = torch.norm(diff, dim=-1) # [B, 4]
            all_errors_px.append(dist.cpu())
            for b in range(batch['image'].size(0)):
                errs = dist[b].cpu().numpy()
                results.append({
                    'img_path': batch['img_path'][b],
                    'mean_err': errs.mean(),
                    'err_tl': errs[0], 'err_tr': errs[1], 'err_br': errs[2], 'err_bl': errs[3]
                })

            if args.save_vis and len(results) <= args.max_vis:
                save_diagnostic_visualization(
                    batch['image'][0], pred[0], gt_corners[0],
                    out['mask'][0], out['edges'][0],
                    batch['img_path'][0], os.path.join(run_dir, 'visualizations')
                )

    all_errors = torch.cat(all_errors_px, dim=0)
    metrics = calculate_accuracy_metrics(all_errors)
    recall = compute_patch_recall(all_errors)

    logger.info("\n" + "="*50)
    logger.info("=== EVALUATION REPORT (Stage 1) ===")
    logger.info("="*50)
    logger.info(f"Mean Pixel Error: {metrics['mean']:.3f}")
    logger.info(f"Median Pixel Error: {metrics['median']:.3f}")
    logger.info(f"Per-Corner Mean: TL={metrics['tl']:.2f} | TR={metrics['tr']:.2f} | BR={metrics['br']:.2f} | BL={metrics['bl']:.2f}")
    logger.info("-" * 20)
    logger.info(f"Patch Recall (64px): {recall['recall_64']:.1f}%")
    logger.info(f"Patch Recall (80px): {recall['recall_80']:.1f}%")
    logger.info(f"Patch Recall (96px): {recall['recall_96']:.1f}%")
    logger.info("="*50)

    if args.save_csv:
        csv_path = os.path.join(run_dir, 'eval_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"CSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()
