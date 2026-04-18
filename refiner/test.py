import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import csv
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from refiner.models.patch_refiner import PatchRefinerNet
from refiner.datasets.refine_keypoint_dataset import FullCardRefinerDataset
from common.checkpoint import load_checkpoint
from common.seed import set_seed
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('refiner_eval')
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
    parser = argparse.ArgumentParser(description="Evaluate Full-Card Corner Refiner (Stage 2).")
    parser.add_argument('--weights', type=str, required=True, help="Path to refiner checkpoint (.pt) or TorchScript model.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Directory with validation images.")
    parser.add_argument('--input_size', type=int, default=640, help="Input size for the network.")
    parser.add_argument('--margin_ratio', type=float, default=0.15, help="Margin to expand Stage 1 BBOX.")
    parser.add_argument('--batch_size', type=int, default=1, help="Evaluation batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="Data loading workers.")
    add_device_args(parser, default='auto')

    parser.add_argument('--run_dir', type=str, default='', help="Explicit run directory.")
    parser.add_argument('--save_csv', action='store_true', default=True, help="Save per-image metrics to CSV.")
    parser.add_argument('--save_vis', action='store_true', default=True, help="Save visualizations for worst cases.")
    parser.add_argument('--max_vis', type=int, default=20, help="Max visualizations to save.")
    parser.add_argument('--report_worst_k', type=int, default=10, help="Number of worst images in report.")
    return parser.parse_args()


def save_refiner_visualization(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                               img_path: str, out_dir: str, input_size: int) -> None:
    """Draws predicted (orange) and target (green) points on the full-card crop.
    
    Args:
        image (torch.Tensor): [3, H, W] normalized image tensor.
        pred (torch.Tensor): [4, 2] normalized coordinates.
        target (torch.Tensor): [4, 2] normalized coordinates.
    """
    try:
        import cv2
        import numpy as np
        from common.transforms import denormalize_image
    except ImportError:
        return

    img = denormalize_image(image.cpu()).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    corner_labels = ["TL", "TR", "BR", "BL"]
    
    for i in range(4):
        px_pred = (pred[i].cpu().numpy() * input_size).astype(int)
        px_target = (target[i].cpu().numpy() * input_size).astype(int)

        # GT green
        cv2.circle(img, tuple(px_pred), 4, (0, 165, 255), -1)   # Pred orange
        cv2.circle(img, tuple(px_target), 4, (0, 255, 0), -1)   # GT green
        
        label = f"{i}:{corner_labels[i]}"
        cv2.putText(img, label, (px_pred[0] + 5, px_pred[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_dir, f"refine_vis_{name}"), img)


def main() -> None:
    args = parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        weights_dir = os.path.dirname(os.path.abspath(args.weights))
        parent_dir = os.path.dirname(weights_dir)
        if os.path.basename(weights_dir) == 'checkpoints':
            run_dir = parent_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join('.', 'runs', 'refiner', f"eval_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(os.path.join(log_dir, 'eval.log'))
    logger.info(f"Evaluation run directory: {run_dir}")

    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    logger.info("Loading validation dataset...")
    val_dataset = FullCardRefinerDataset(
        args.val_images, input_size=args.input_size, margin_ratio=args.margin_ratio
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    try:
        model = torch.jit.load(args.weights, map_location=device)
        logger.info("Loaded TorchScript model.")
    except Exception:
        model = PatchRefinerNet(input_size=args.input_size).to(device)
        load_checkpoint(model, None, None, args.weights, device=device)
        logger.info("Loaded eager PyTorch checkpoint.")

    model.eval()

    results = []
    total_infer_time = 0.0
    total_samples = 0
    t_start = sync_time()

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            targets = batch['targets']
            metadata = batch['metadata']
            crop_box = metadata['crop_box'] # [B, 4]

            B = images.size(0)
            t_infer = sync_time()
            pred, _ = model(images) # Extract final refined prediction from 2-output model
            infer_time = sync_time() - t_infer

            total_infer_time += infer_time
            total_samples += B

            for b in range(B):

                p_norm = pred[b]
                t_norm = targets[b]
                
                # Metadata for EXACT Inverse Mapping
                m_crop_box = metadata['crop_box'][b]
                m_scale = metadata['scale'][b]
                m_padding = metadata['padding'][b]
                m_in_size = metadata['input_size'][b]
                
                def inverse_map(pts_norm):
                    pts_px = pts_norm * m_in_size
                    pts_resized = pts_px - m_padding
                    pts_crop = pts_resized / m_scale
                    return pts_crop + m_crop_box[:2]

                p_orig = inverse_map(p_norm)
                t_orig = inverse_map(t_norm)
                
                dist = torch.norm(p_orig - t_orig, dim=-1) # [4]
                
                img_path = batch['img_path'][b]
                results.append({
                    'img_path': img_path,
                    'err_tl': dist[0].item(),
                    'err_tr': dist[1].item(),
                    'err_br': dist[2].item(),
                    'err_bl': dist[3].item(),
                    'mean_err': dist.mean().item(),
                    'pred_norm': p_norm.cpu(),
                    'target_norm': t_norm.cpu(),
                    'image_tensor': images[b].cpu(),
                })


    total_eval_time = sync_time() - t_start
    results_sorted = sorted(results, key=lambda x: x['mean_err'], reverse=True)

    all_errs = torch.tensor([r['mean_err'] for r in results])
    err_tl = torch.tensor([r['err_tl'] for r in results])
    err_tr = torch.tensor([r['err_tr'] for r in results])
    err_br = torch.tensor([r['err_br'] for r in results])
    err_bl = torch.tensor([r['err_bl'] for r in results])

    avg_infer_ms = (total_infer_time / total_samples) * 1000 if total_samples > 0 else 0

    # === Report ===
    logger.info("\n" + "=" * 50)
    logger.info("=== REFINER EVALUATION RESULTS ===")
    logger.info("=" * 50)
    logger.info(f"Model: {args.weights}")
    logger.info(f"Images evaluated: {len(results)}")

    logger.info("\n--- Timing ---")
    logger.info(f"Total eval time: {total_eval_time:.2f} s")
    logger.info(f"Avg inference per sample: {avg_infer_ms:.2f} ms")

    logger.info("\n--- Mean Euclidean Errors (Original Pixels) ---")
    logger.info(f"Global Mean: {all_errs.mean().item():.3f} px")
    logger.info(f"Global Median: {all_errs.median().item():.3f} px")

    logger.info("\n--- Per-Corner Mean errors (px) ---")
    logger.info(f"TL: {err_tl.mean().item():.3f} | TR: {err_tr.mean().item():.3f}")
    logger.info(f"BR: {err_br.mean().item():.3f} | BL: {err_bl.mean().item():.3f}")

    logger.info("\n--- Accuracy Thresholds ---")
    for t in [1, 2, 3, 5, 10]:
        acc = (all_errs < t).float().mean().item() * 100
        logger.info(f"<{t} px: {acc:.1f}%")

    logger.info("\n--- Worst-K ---")
    for k in range(min(args.report_worst_k, len(results_sorted))):
        r = results_sorted[k]
        logger.info(f"[{k+1}] {r['mean_err']:.2f} px | {os.path.basename(r['img_path'])}")

    # CSV
    if args.save_csv:
        csv_path = os.path.join(run_dir, 'refiner_eval_results.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['img_path', 'mean_err', 'err_tl', 'err_tr', 'err_br', 'err_bl'])
                for r in results_sorted:
                    writer.writerow([r['img_path'], r['mean_err'], r['err_tl'], r['err_tr'], r['err_br'], r['err_bl']])
            logger.info(f"\nCSV saved: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    # Visualizations
    if args.save_vis:
        vis_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        drawn = 0
        for r in results_sorted:
            if drawn >= args.max_vis:
                break
            save_refiner_visualization(r['image_tensor'], r['pred_norm'], r['target_norm'], r['img_path'], vis_dir, args.input_size)
            drawn += 1
        logger.info(f"Saved {drawn} visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
