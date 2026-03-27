import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coarse.models.coarse_quad_net import CoarseQuadNet
from coarse.datasets.yolo_keypoint_dataset import YOLOKeypointDataset
from coarse.datasets.coco_val_dataset import COCOValDataset
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.metrics import (DiceLoss, GeometryAlignmentLoss, HeatmapFocalLoss, 
                            calculate_accuracy_metrics, compute_patch_recall, 
                            CenterOffsetL1Loss)
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.logging_utils import TrainingTracker, TopLossTracker, HardExampleMiner
from common.visualization import save_diagnostic_visualization
from torch.utils.data import WeightedRandomSampler


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('coarse_train')
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
    parser = argparse.ArgumentParser(description="Train Structured Coarse Quad Net (Stage 1).")
    parser.add_argument('--train_images', type=str, default='../crop-dataset-eitan-yolo/images/train', help="Path to training images.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Path to validation images.")
    parser.add_argument('--val_json', type=str, default='../crop-dataset-eitan-yolo/annotations/val.json', help="Path to validation COCO JSON.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--image_size', type=int, default=384, help="Input image spatial size.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--runs_dir', type=str, default='./coarse/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from.")
    
    # Performance
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--pin_memory', action='store_true', default=True, help="Use pinned memory for faster GPU transfer.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    
    # Simplified Loss Weights (v8.0 - orient head removed)
    parser.add_argument('--w_offset', type=float, default=1.0, help="Dense Sub-grid Offset L1 Loss.")
    parser.add_argument('--w_heatmap', type=float, default=1.0, help="Dense Heatmap supervision (Focal Loss).")
    parser.add_argument('--w_coord_quad', type=float, default=0.5, help="Auxiliary global quad anchoring loss.")

    # Mining
    parser.add_argument('--mine_hard', action='store_true', default=True, help="Enable hard example mining.")
    parser.add_argument('--mine_exponent', type=float, default=1.5, help="Aggressiveness of hard mining (weight = error ^ exponent).")

    add_device_args(parser, default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    logger.info(f"Stage 1 Redesign Losses: Heatmap={args.w_heatmap}, DenseCoord={args.w_offset}, QuadCoord={args.w_coord_quad}")
    
    tracker = TrainingTracker(logger)
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    logger.info("Initializing datasets...")
    train_dataset = YOLOKeypointDataset(args.train_images, image_size=args.image_size)
    val_dataset = COCOValDataset(args.val_images, args.val_json, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = CoarseQuadNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: Dense-Primary CoarseQuadNet | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss Functions (v8.0)
    heatmap_criterion = HeatmapFocalLoss(alpha=2.0, beta=4.0)
    offset_criterion = CenterOffsetL1Loss()

    start_epoch = 0
    best_mean_error = float('inf')
    best_recall_96 = 0.0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_recall_96 = checkpoint.get('best_metric', 0.0)
            logger.info(f"Restored checkpoint at epoch {start_epoch} (best recall: {best_recall_96:.2f})")

    miner = HardExampleMiner(len(train_loader.dataset))
    
    global_start_time = sync_time()
    for epoch in range(start_epoch, args.epochs):
        tracker.start_epoch()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr

        # === Training ===
        model.train()
        tracker.start_train_phase()
        
        # Dynamic Aggressiveness
        current_exponent = 1.0 + (args.mine_exponent - 1.0) * (epoch / max(1, args.epochs - 1))
        
        if args.mine_hard and epoch >= 10:
            weights = miner.get_weights(exponent=current_exponent)
            sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)
            logger.info(f"Epoch {epoch+1}: Recreated train_loader (Mining Exp: {current_exponent:.2f}).")

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for batch in train_pbar:
            batch_start = sync_time()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            
            out = model(batch['image'])
            gt_corners = batch['corners']
            
            # --- Dynamic Corner Sorting (Physical to Visual) ---
            # CNN heatmaps/offsets are explicitly trained on 'visual' spatial geometry.
            centroid = gt_corners.mean(dim=1, keepdim=True) # [B, 1, 2]
            diffs = gt_corners - centroid
            angles = torch.atan2(diffs[:, :, 1], diffs[:, :, 0]) # [B, 4]
            sort_idx = torch.argsort(angles, dim=1) # [B, 4]
            
            B_cur = gt_corners.size(0)
            b_idx_t = torch.arange(B_cur, device=device).unsqueeze(1).expand(B_cur, 4)
            gt_corners_visual = gt_corners[b_idx_t, sort_idx] # [B, 4, 2]
            
            # The 'shift' mapping visual back to physical is simply the physical index of the visual TL
            gt_orient = sort_idx[:, 0] # [B]
            
            # Compute Exact Centroid [B, 1, 2]
            gt_centers = gt_corners_visual.mean(dim=1, keepdim=True)
            
            # 1. Primary Dense Spatial Supervision
            loss_offset = offset_criterion(out['dense_offsets'], gt_centers, gt_corners_visual)
            loss_heatmap = heatmap_criterion(out['dense_center'], gt_centers)
            
            total_loss = (loss_offset * args.w_offset + 
                          loss_heatmap * args.w_heatmap)
            
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Detailed component logging (v8.0)
            components = {
                'offset_raw': loss_offset.item(),
                'offset_w': (loss_offset * args.w_offset).item(),
                'heatmap_raw': loss_heatmap.item(),
                'heatmap_w': (loss_heatmap * args.w_heatmap).item(),
            }
            tracker.record_batch('train', total_loss.item(), sync_time() - batch_start, components=components)
            train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
            # Update Miner with per-sample errors (Pixel space)
            with torch.no_grad():
                w_orig = batch.get('orig_width', torch.tensor(args.image_size, device=device)).view(-1, 1)
                h_orig = batch.get('orig_height', torch.tensor(args.image_size, device=device)).view(-1, 1)
                
                # Use geometric visual corners for geometric mining
                diff = (out['corners_visual'] - gt_corners_visual).abs()
                diff[:, :, 0] *= w_orig
                diff[:, :, 1] *= h_orig
                
                # Mining: Worst-corner prioritization
                dist_max = torch.norm(diff, dim=-1).max(dim=-1)[0] # max error per image [B]
                
                weights = []
                for b_idx in range(gt_corners.size(0)):
                    d = dist_max[b_idx].item()
                    w = d
                    if d > 48.0:
                        w *= 5.0 # heavily penalize corners falling outside the 96px capture box
                    weights.append(w)
                miner.update(batch['index'].tolist(), weights)

            del out, batch, total_loss, components
            del loss_offset, loss_heatmap

        tracker.end_train_phase()
        torch.cuda.empty_cache()

        # === Validation ===
        model.eval()
        tracker.start_val_phase()
        all_errs = []
        top_tracker = TopLossTracker(k=5)
        
        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.inference_mode():
            for batch in val_pbar:
                batch_start = sync_time()
                batch = move_batch_to_device(batch, device)
                out = model(batch['image'])
                gt_corners = batch['corners']
                
                # --- Dynamic Corner Sorting (Physical to Visual) ---
                centroid = gt_corners.mean(dim=1, keepdim=True) # [B, 1, 2]
                diffs = gt_corners - centroid
                angles = torch.atan2(diffs[:, :, 1], diffs[:, :, 0]) # [B, 4]
                sort_idx = torch.argsort(angles, dim=1) # [B, 4]
                
                B_cur = gt_corners.size(0)
                b_idx_t = torch.arange(B_cur, device=device).unsqueeze(1).expand(B_cur, 4)
                gt_corners_visual = gt_corners[b_idx_t, sort_idx] # [B, 4, 2]
                gt_orient = sort_idx[:, 0] # [B]
                gt_centers = gt_corners_visual.mean(dim=1, keepdim=True)
                
                # Val Loss tracking
                loss_offset = offset_criterion(out['dense_offsets'], gt_centers, gt_corners_visual)
                loss_heatmap = heatmap_criterion(out['dense_center'], gt_centers)
                
                val_total_loss = (loss_offset * args.w_offset + 
                                  loss_heatmap * args.w_heatmap)
                
                components = {
                    'offset_raw': loss_offset.item(),
                    'offset_w': (loss_offset * args.w_offset).item(),
                    'heatmap_raw': loss_heatmap.item(),
                    'heatmap_w': (loss_heatmap * args.w_heatmap).item(),
                }
                tracker.record_batch('val', val_total_loss.item(), sync_time() - batch_start, components=components)

                # Accuracy (Pixel Space)
                w_orig = batch.get('orig_width', torch.tensor(args.image_size, device=device)).view(-1, 1)
                h_orig = batch.get('orig_height', torch.tensor(args.image_size, device=device)).view(-1, 1)

                diff = (out['corners'] - gt_corners_visual).abs()
                diff[:, :, 0] *= w_orig
                diff[:, :, 1] *= h_orig
                dist = torch.norm(diff, dim=-1) # [B, 4]
                all_errs.append(dist.cpu())
                
                # Update top-loss samples
                m_dist = dist.mean(dim=-1)
                for b_idx in range(batch['image'].size(0)):
                    top_tracker.update(m_dist[b_idx].item(), {
                        'image': batch['image'][b_idx],
                        'pred': out['corners'][b_idx],
                        'gt': gt_corners[b_idx],
                        'path': batch['img_path'][b_idx]
                    })
                
                # Explicitly free variables
                del out, batch, val_total_loss, components
                del loss_offset, loss_heatmap

        tracker.end_val_phase()
        torch.cuda.empty_cache()

        # Metrics
        errors = torch.cat(all_errs, dim=0)
        metrics = calculate_accuracy_metrics(errors)
        recall_val = compute_patch_recall(errors)
        
        # Summary Log
        tracker.log_epoch_summary(epoch + 1, args.epochs, current_lr, metrics, recall_val)
        scheduler.step()

        # Save Visualizations
        epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1:03d}")
        for sample in top_tracker.get_samples():
            save_diagnostic_visualization(
                sample['image'], sample['pred'], sample['gt'],
                None, None, # No mask/edges in redesigned version
                sample['path'], epoch_vis_dir
            )

        # Checkpoint (Targeting recall_96 explicitly)
        latest_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, recall_val['recall_96'], latest_path)
        
        current_recall = recall_val['recall_96']
        current_mean = metrics['mean']
        
        is_better = False
        if current_recall > best_recall_96:
            is_better = True
        elif current_recall == best_recall_96 and current_mean < best_mean_error:
            is_better = True
            
        if is_better:
            best_recall_96 = current_recall
            best_mean_error = current_mean
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_recall_96, best_path)
            logger.info(f"New best model: {best_path} (recall_96={best_recall_96:.1f}%, mean_err={best_mean_error:.3f})")

    total_time = (sync_time() - global_start_time) / 60
    logger.info(f"Training finished in {total_time:.1f}m. Best Recall@96: {best_recall_96:.1f}%")


if __name__ == "__main__":
    main()
