import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
import math
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from refiner.models.patch_refiner import PatchRefinerNet
from refiner.datasets.refine_keypoint_dataset import FullCardRefinerDataset
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.metrics import calculate_accuracy_metrics, HeatmapLoss
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.logging_utils import TrainingTracker, TopLossTracker
from common.visualization import save_diagnostic_visualization, save_refiner_global_debug


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('refiner_train')
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
    parser = argparse.ArgumentParser(description="Train High-Precision Full-Card Corner Refiner (Stage 2).")
    parser.add_argument('--train_images', type=str, default='../crop-dataset-eitan-yolo/images/train', help="Path to training images.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Path to validation images.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument('--input_size', type=str, default='320,192', help="Input size for the network as 'W,H'.")
    parser.add_argument('--margin_ratio', type=float, default=0.15, help="Margin to expand Stage 1 BBOX.")
    parser.add_argument('--runs_dir', type=str, default='./refiner/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers.")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from.")
    parser.add_argument('--overfit', action='store_true', help="Filter and train on exactly 10 clean samples to verify convergence.")
    add_device_args(parser, default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Parse input_size "W,H"
    if ',' in args.input_size:
        args.input_size = tuple(map(int, args.input_size.split(',')))
    else:
        args.input_size = int(args.input_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.name}" if args.name else timestamp
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    vis_dir = os.path.join(run_dir, 'visualizations', 'top_losses')
    os.makedirs(vis_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger = setup_logging(os.path.join(run_dir, 'logs', 'train.log'))
    logger.info(f"Run directory: {run_dir}")

    tracker = TrainingTracker(logger)
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    if os.name == 'nt' and device.type == 'cpu' and args.num_workers > 0:
        logger.warning("Windows CPU detected: setting num_workers=0 to prevent shared file mapping error <1455>.")
        args.num_workers = 0

    logger.info("Initializing datasets...")
    train_dataset = FullCardRefinerDataset(args.train_images, input_size=args.input_size, margin_ratio=args.margin_ratio)
    val_dataset = FullCardRefinerDataset(args.val_images, input_size=args.input_size, margin_ratio=args.margin_ratio)

    if args.overfit:
        logger.info("=" * 80)
        logger.info(" OVERFIT MODE ".center(80, "="))
        indices = []
        for i, lbl in enumerate(train_dataset.labels):
            if lbl and 'keypoints' in lbl:
                kpts = lbl['keypoints']
                # Check if all 4 corners are well centered [0.1, 0.9]
                if all(0.1 < k[0] < 0.9 and 0.1 < k[1] < 0.9 for k in kpts):
                    indices.append(i)
            if len(indices) >= 10:
                break
        
        if not indices:
            logger.warning("OVERFIT MODE: No well-centered samples found. Falling back to first 10 valid samples.")
            indices = list(range(min(10, len(train_dataset))))
        elif len(indices) < 10:
            logger.warning(f"OVERFIT MODE: Only found {len(indices)} well-centered samples.")

        logger.info(f"OVERFIT MODE: Selected {len(indices)} samples.")
        logger.info(f"OVERFIT MODE: Selected indices: {indices}")
        logger.info("OVERFIT MODE: Validation set will use the same subset for verification.")
        
        train_dataset = Subset(train_dataset, indices)
        val_dataset = train_dataset
        args.batch_size = min(args.batch_size, len(indices))
        logger.info("=" * 80)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PatchRefinerNet(input_size=args.input_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Scheduler with warmup
    n_warmup = 3
    def lr_lambda(epoch):
        if args.overfit:
            return 1.0 # Standard learning rate for overfit mode (2e-4)
        if epoch < n_warmup:
            return (epoch + 1) / n_warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - n_warmup) / (args.epochs - n_warmup)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    focal_criterion = HeatmapLoss().to(device)
    
    start_epoch = 0
    best_mean_error = float('inf')
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_mean_error = checkpoint.get('best_metric', float('inf'))

    global_start_time = sync_time()
    for epoch in range(start_epoch, args.epochs):
        logger.info(f" EPOCH {epoch+1}/{args.epochs} ".center(80, "="))
        
        tracker.start_epoch()
        current_lr = optimizer.param_groups[0]['lr']

        # ROI Scheduling: 100% GT for epochs 0-15, linear decay to 0% by epoch 35
        # Overfit mode: Always 100% GT
        if args.overfit:
            gt_roi_prob = 1.0
        elif epoch < 15:
            gt_roi_prob = 1.0
        elif epoch > 35:
            gt_roi_prob = 0.0
        else:
            gt_roi_prob = 1.0 - (epoch - 15) / (35 - 15)

        # === Training ===
        model.train()
        tracker.start_train_phase()
        train_pbar = tqdm(train_loader, desc=f"Train", leave=False)
        for batch in train_pbar:
            batch_start = sync_time()
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            targets = batch['targets'] # [B, 4, 2] normalized in crop

            optimizer.zero_grad()
            # pred_coarse, pred_refined, pred_heatmaps, roi_boxes, pred_roi_boxes: [B, 4, 2]
            # Stochastic ROI choice for robustness
            use_gt = random.random() < gt_roi_prob
            pred_coarse, pred_refined, pred_heatmaps, roi_boxes, pred_roi_boxes = model(images, gt_pts=targets if use_gt else None)

            # 1. Coarse Loss
            loss_coarse = F.smooth_l1_loss(pred_coarse, targets)
            
            # 2. Refined Loss
            loss_refined = F.smooth_l1_loss(pred_refined, targets)
            
            # 3. Heatmap Loss (Uses RoI-local targets from DYNAMIC ROI)
            in_w = batch['metadata']['input_size'][:, 0].view(-1, 1, 1)
            in_h = batch['metadata']['input_size'][:, 1].view(-1, 1, 1)
            roi_x1 = roi_boxes[:, 0].view(-1, 1, 1)
            roi_y1 = roi_boxes[:, 1].view(-1, 1, 1)
            roi_w = (roi_boxes[:, 2] - roi_boxes[:, 0]).view(-1, 1, 1)
            roi_h = (roi_boxes[:, 3] - roi_boxes[:, 1]).view(-1, 1, 1)

            targets_roi_x = (targets[:, :, 0:1] * in_w - roi_x1) / torch.clamp(roi_w, min=1.0)
            targets_roi_y = (targets[:, :, 1:2] * in_h - roi_y1) / torch.clamp(roi_h, min=1.0)
            targets_roi = torch.cat([targets_roi_x, targets_roi_y], dim=-1)

            loss_heatmap = focal_criterion(pred_heatmaps, targets_roi)

            loss = 1.0 * loss_coarse + 1.0 * loss_refined + 1.0 * loss_heatmap
            
            loss.backward()
            optimizer.step()

            tracker.record_batch('train', loss.item(), sync_time() - batch_start, 
                                 components={'heatmap': loss_heatmap.item()})
            train_pbar.set_postfix({'loss': f"{loss.item():.5f}"})
        
        tracker.end_train_phase()
        avg_train_loss = sum(tracker.train_losses) / len(tracker.train_losses)

        # === Validation ===
        model.eval()
        errors_refined = []
        errors_coarse = []
        losses = []
        val_preds = []
        top_tracker = TopLossTracker(k=min(5, len(val_dataset)))
        
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                batch = move_batch_to_device(batch, device)
                images = batch['image']
                targets = batch['targets']
                meta = batch['metadata']
                
                # v_coarse, v_refined: [B, 4, 2] in full input space (Crop)
                # v_heatmaps: [B, 4, 112, 112] in RoI space
                v_coarse, v_refined, v_heatmaps, v_roi_boxes, v_pred_roi_boxes = model(images)
                
                # 1. Heatmap Loss (RoI Space)
                v_in_w = meta['input_size'][:, 0].view(-1, 1, 1)
                v_in_h = meta['input_size'][:, 1].view(-1, 1, 1)
                v_roi_x1 = v_roi_boxes[:, 0].view(-1, 1, 1)
                v_roi_y1 = v_roi_boxes[:, 1].view(-1, 1, 1)
                v_roi_w = (v_roi_boxes[:, 2] - v_roi_boxes[:, 0]).view(-1, 1, 1)
                v_roi_h = (v_roi_boxes[:, 3] - v_roi_boxes[:, 1]).view(-1, 1, 1)

                v_targets_roi_x = (targets[:, :, 0:1] * v_in_w - v_roi_x1) / torch.clamp(v_roi_w, min=1.0)
                v_targets_roi_y = (targets[:, :, 1:2] * v_in_h - v_roi_y1) / torch.clamp(v_roi_h, min=1.0)
                v_targets_roi = torch.cat([v_targets_roi_x, v_targets_roi_y], dim=-1)

                v_loss_heatmap = focal_criterion(v_heatmaps, v_targets_roi)
                v_loss_coarse = F.smooth_l1_loss(v_coarse, targets)
                v_loss_refined = F.smooth_l1_loss(v_refined, targets)

                v_loss = v_loss_coarse + v_loss_refined + v_loss_heatmap
                losses.append(v_loss.item())
                val_preds.append(v_refined.detach().cpu())
                
                # Compute raw peaks for visualization (Map from RoI space to Crop space)
                B_v, C_v, H_v, W_v = v_heatmaps.shape
                v_heatmaps_flat = v_heatmaps.view(B_v, C_v, -1)
                max_idx = torch.argmax(v_heatmaps_flat, dim=-1)
                peak_y = max_idx // W_v
                peak_x = max_idx % W_v
                roi_peaks = torch.stack([(peak_x.float() + 0.5) / W_v, (peak_y.float() + 0.5) / H_v], dim=-1)
                
                # Metadata mapping (using DYNAMIC ROI)
                v_roi_x1 = v_roi_boxes[:, 0].view(B_v, 1)
                v_roi_y1 = v_roi_boxes[:, 1].view(B_v, 1)
                v_roi_w = (v_roi_boxes[:, 2] - v_roi_boxes[:, 0]).view(B_v, 1)
                v_roi_h = (v_roi_boxes[:, 3] - v_roi_boxes[:, 1]).view(B_v, 1)
                
                in_w = meta['input_size'][:, 0].view(B_v, 1)
                in_h = meta['input_size'][:, 1].view(B_v, 1)
                
                final_peak_x = (roi_peaks[:, :, 0] * v_roi_w + v_roi_x1) / in_w
                final_peak_y = (roi_peaks[:, :, 1] * v_roi_h + v_roi_y1) / in_h
                raw_peaks = torch.stack([final_peak_x, final_peak_y], dim=-1)
                
                # EXACT Inverse Mapping to Original Pixel Coordinates
                B = v_refined.size(0)
                for b in range(B):
                    p_norm = v_refined[b] # [4, 2] in [0, 1] input space
                    p_coarse = v_coarse[b]
                    t_norm = targets[b]     # [4, 2]
                    
                    # Extract Metadata
                    crop_box = meta['crop_box'][b] # [x1, y1, x2, y2]
                    px1, py1, px2, py2 = crop_box
                    crop_w = px2 - px1
                    crop_h = py2 - py1
                    
                    def inverse_map(pts_norm):
                        kx = pts_norm[:, 0] * crop_w
                        ky = pts_norm[:, 1] * crop_h
                        return torch.stack([kx + px1, ky + py1], dim=-1)

                    p_orig = inverse_map(p_norm)
                    pc_orig = inverse_map(p_coarse)
                    t_orig = inverse_map(t_norm)
                    
                    dist_refined = torch.norm(p_orig - t_orig, dim=-1) # [4] pixels
                    dist_coarse = torch.norm(pc_orig - t_orig, dim=-1)
                    errors_refined.append(dist_refined.cpu())
                    errors_coarse.append(dist_coarse.cpu())
                    
                    top_tracker.update(dist_refined.mean().item(), {
                        'image': images[b],
                        'pred': p_norm,
                        'pred_coarse': p_coarse,
                        'gt': t_norm,
                        'raw_peaks': raw_peaks[b].cpu(),
                        'heatmaps': v_heatmaps[b].cpu(),
                        'roi_box': v_roi_boxes[b].cpu(),
                        'pred_roi_box': v_pred_roi_boxes[b].cpu(),
                        'path': batch['img_path'][b]
                    })

            
        all_errors_refined = torch.cat(errors_refined, dim=0)
        all_errors_coarse = torch.cat(errors_coarse, dim=0)
        metrics_refined = calculate_accuracy_metrics(all_errors_refined)
        metrics_coarse = calculate_accuracy_metrics(all_errors_coarse)
        avg_val_loss = sum(losses) / len(losses)
        
        # Logging Summary Table
        logger.info(f"LR: {current_lr:.6f} | GT ROI Prob: {gt_roi_prob:.2f} | Size: {args.input_size}")
        header = f"{'Phase':<15} | {'Loss':<10} | {'Coarse (px)':<12} | {'Refined (px)':<12} | {'<1px (%)':<10} | {'<2px (%)':<10} | {'<3px (%)':<10}"
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        logger.info(f"{'Train':<15} | {avg_train_loss:<10.5f} | {'-':<12} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10}")
        
        fmt = lambda m_ref, m_crs, l: f"{l:<10.5f} | {m_crs['mean']:<12.3f} | {m_ref['mean']:<12.3f} | {m_ref['acc_1px']:<10.1f} | {m_ref['acc_2px']:<10.1f} | {m_ref['acc_3px']:<10.1f}"
        logger.info(f"{'Validation':<15} | " + fmt(metrics_refined, metrics_coarse, avg_val_loss))
        logger.info("-" * len(header))
        
        scheduler.step()

        # Save Visualizations for Top Losses
        epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_vis_dir, exist_ok=True)
        for sample in top_tracker.get_samples():
                save_refiner_global_debug(
                    sample['image'], sample['pred'], sample['gt'],
                    sample['heatmaps'], sample['path'], epoch_vis_dir,
                    roi_box=sample['roi_box'],
                    coarse_corners=sample.get('pred_coarse'),
                    pred_roi_box=sample.get('pred_roi_box')
                )
        
        # Checkpoints based on Mean Error (Refined)
        latest_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics_refined['mean'], latest_path)
        
        if metrics_refined['mean'] < best_mean_error:
            best_mean_error = metrics_refined['mean']
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_mean_error, best_path)
            logger.info(f"New best model: {best_path} (mean_err={best_mean_error:.3f})")

    total_time = (sync_time() - global_start_time) / 60
    logger.info(f"=== Training finished in {total_time:.1f}m. Best Mean Error: {best_mean_error:.3f} px ===")


if __name__ == "__main__":
    main()
