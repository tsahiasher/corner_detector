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
from common.metrics import (DiceLoss, WingLoss, GeometryAlignmentLoss, 
                            QuadShapeLoss, calculate_accuracy_metrics, 
                            compute_patch_recall)
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
    
    # Structured Loss Weights (v6)
    parser.add_argument('--w_coord', type=float, default=2.5, help="Primary WingLoss on 4 ordered corners.")
    parser.add_argument('--w_mask', type=float, default=0.5, help="Dense Mask supervision.")
    parser.add_argument('--w_edge', type=float, default=2.0, help="Dense Boundary/Contour supervision (Gaussian/v2).")
    parser.add_argument('--w_shape', type=float, default=0.5, help="Quad shape regularizer (Clockwise/Convex).")
    parser.add_argument('--w_reg', type=float, default=0.1, help="L2 Regularization on corner residuals.")
    parser.add_argument('--w_align', type=float, default=1.0, help="Geometry Alignment (Corners to Gaussian Edges).")
    parser.add_argument('--w_score', type=float, default=0.1, help="Confidence Score.")
    
    # Mining
    parser.add_argument('--mine_hard', action='store_true', default=True, help="Enable hard example mining via WeightedRandomSampler.")
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
    logger.info(f"Structured Loss Config: Coord={args.w_coord}, Mask={args.w_mask}, Edge={args.w_edge}, Align={args.w_align}, Shape={args.w_shape}")
    
    tracker = TrainingTracker(logger)
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    logger.info("Initializing datasets...")
    train_dataset = YOLOKeypointDataset(args.train_images, image_size=args.image_size)
    val_dataset = COCOValDataset(args.val_images, args.val_json, image_size=args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    miner = HardExampleMiner(len(train_dataset))
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = CoarseQuadNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: Structured CoarseQuadNet | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Criterions
    score_criterion = nn.BCELoss()
    coord_criterion = WingLoss(wing_w=5.0, epsilon=1.0) 
    dice_criterion = DiceLoss()
    bce_dense_criterion = nn.BCELoss()
    align_criterion = GeometryAlignmentLoss()
    shape_criterion = QuadShapeLoss(weight_diag=0.5, weight_convex=1.5)

    start_epoch = 0
    best_mean_error = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_mean_error = checkpoint.get('best_metric', float('inf'))
            logger.info(f"Restored checkpoint at epoch {start_epoch} (best: {best_mean_error:.3f})")

    miner = HardExampleMiner(len(train_loader.dataset))
    
    global_start_time = sync_time()
    for epoch in range(start_epoch, args.epochs):
        tracker.start_epoch()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr

        # === Training ===
        model.train()
        tracker.start_train_phase()
        
        # Performance Boost v2: Dynamic Aggressiveness
        # Gradually increase exponent from 1.0 to 2.5 over epochs to focus on remaining hard cases
        current_exponent = 1.0 + (args.mine_exponent - 1.0) * (epoch / max(1, args.epochs - 1))
        
        # Update Weighted Sampler every epoch if mining is enabled
        if args.mine_hard and epoch > 0:
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
            
            # 1. Coordinate Accuracy (Ordered 4 Corners)
            loss_coord = coord_criterion(out['corners'], gt_corners)
            
            # 2. Geometric Anchoring (Mask + Boundary)
            loss_mask = 0.5 * dice_criterion(out['mask'], batch['mask']) + 0.5 * bce_dense_criterion(out['mask'], batch['mask'])
            loss_edge = 0.5 * dice_criterion(out['edges'], batch['edges']) + 0.5 * bce_dense_criterion(out['edges'], batch['edges'])
            
            # 3. Shape & Parameter Regularization
            loss_shape = shape_criterion(out['corners'])
            loss_reg = torch.mean(out['residuals'] ** 2)
            
            # 4. Alignment
            loss_align = align_criterion(out['corners'], out['edges'].detach(), out['mask'].detach())
            
            # 5. Confidence
            loss_score = score_criterion(out['score'], torch.ones_like(out['score']))
            
            total_loss = (loss_coord * args.w_coord + 
                          loss_mask * args.w_mask + 
                          loss_edge * args.w_edge + 
                          loss_shape * args.w_shape + 
                          loss_reg * args.w_reg +
                          loss_align * args.w_align + 
                          loss_score * args.w_score)
            
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tracker.record_batch('train', total_loss.item(), sync_time() - batch_start)
            train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

            # Update Miner with per-sample errors (Pixel space)
            with torch.no_grad():
                w_orig = batch.get('orig_width', torch.tensor(args.image_size, device=device)).view(-1, 1)
                h_orig = batch.get('orig_height', torch.tensor(args.image_size, device=device)).view(-1, 1)
                diff = (out['corners'] - gt_corners).abs()
                diff[:, :, 0] *= w_orig
                diff[:, :, 1] *= h_orig
                dist = torch.norm(diff, dim=-1).mean(dim=-1) # Mean pixel error per image
                miner.update(batch['index'].tolist(), dist.cpu().tolist())

        tracker.end_train_phase()

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
                
                # Val Loss tracking
                l_coord = coord_criterion(out['corners'], gt_corners)
                val_total_loss = l_coord * args.w_coord 
                
                tracker.record_batch('val', val_total_loss.item(), sync_time() - batch_start)

                # Accuracy (Pixel Space)
                w_orig = batch.get('orig_width', torch.tensor(args.image_size, device=device)).view(-1, 1)
                h_orig = batch.get('orig_height', torch.tensor(args.image_size, device=device)).view(-1, 1)

                diff = (out['corners'] - gt_corners).abs()
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
                        'mask': out['mask'][b_idx],
                        'edges': out['edges'][b_idx],
                        'path': batch['img_path'][b_idx]
                    })

        tracker.end_val_phase()
        
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
                sample['mask'], sample['edges'],
                sample['path'], epoch_vis_dir
            )

        # Checkpoint
        latest_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics['mean'], latest_path)
        
        if metrics['mean'] < best_mean_error:
            best_mean_error = metrics['mean']
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_mean_error, best_path)
            logger.info(f"New best model: {best_path} (mean_err={best_mean_error:.3f})")

    total_time = (sync_time() - global_start_time) / 60
    logger.info(f"Training finished in {total_time:.1f}m. Best Mean Error: {best_mean_error:.3f}")


if __name__ == "__main__":
    main()
