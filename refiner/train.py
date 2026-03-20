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

from refiner.models.patch_refiner import PatchRefinerNet
from refiner.datasets.refine_keypoint_dataset import RefineKeypointDataset
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.metrics import WingLoss, calculate_accuracy_metrics
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.logging_utils import TrainingTracker, TopLossTracker
from common.visualization import save_diagnostic_visualization


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
    parser = argparse.ArgumentParser(description="Train Patch Refiner Net (Stage 2).")
    parser.add_argument('--train_images', type=str, default='../crop-dataset-eitan-yolo/images/train', help="Path to training images directory.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Path to validation images directory.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument('--jitter_px', type=float, default=25.0, help="Max jitter in pixels to simulate Stage 1 errors.")
    parser.add_argument('--patch_size', type=int, default=96, help="Side length of corner patches.")
    parser.add_argument('--runs_dir', type=str, default='./refiner/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loading workers.")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from.")
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

    tracker = TrainingTracker(logger)
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    logger.info("Initializing datasets...")
    train_dataset = RefineKeypointDataset(args.train_images, is_train=True, jitter_px=args.jitter_px, patch_size=args.patch_size)
    # Clean validation (centered)
    val_dataset_clean = RefineKeypointDataset(args.val_images, is_train=False, jitter_px=0.0, patch_size=args.patch_size)
    # Realistic jittered validation
    val_dataset_jittered = RefineKeypointDataset(args.val_images, is_train=False, jitter_px=args.jitter_px, patch_size=args.patch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader_clean = DataLoader(val_dataset_clean, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader_jittered = DataLoader(val_dataset_jittered, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PatchRefinerNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Scheduler with warmup
    n_warmup = 3
    def lr_lambda(epoch):
        if epoch < n_warmup:
            return (epoch + 1) / n_warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - n_warmup) / (args.epochs - n_warmup)))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    corner_criterion = WingLoss(wing_w=5.0, epsilon=1.0)

    start_epoch = 0
    best_jittered_error = float('inf')
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_jittered_error = checkpoint.get('best_metric', float('inf'))

    global_start_time = sync_time()
    for epoch in range(start_epoch, args.epochs):
        logger.info("\n" + "="*80)
        logger.info(f" EPOCH {epoch+1}/{args.epochs} ".center(80, "="))
        logger.info("="*80)
        
        tracker.start_epoch()
        current_lr = optimizer.param_groups[0]['lr']

        # === Training ===
        model.train()
        tracker.start_train_phase()
        train_pbar = tqdm(train_loader, desc=f"Train", leave=False)
        for batch in train_pbar:
            batch_start = sync_time()
            batch = move_batch_to_device(batch, device)
            patches = batch['patches']
            targets = batch['targets']
            B = patches.size(0)
            
            patches_flat = patches.view(B * 4, 3, args.patch_size, args.patch_size)
            targets_flat = targets.view(B * 4, 2)

            optimizer.zero_grad()
            # IterativeRefinerNet returns (final, coarse) in training mode
            final_pred, coarse_pred = model(patches_flat)
            
            # Multi-level loss
            loss_coarse = corner_criterion(coarse_pred, targets_flat, width=args.patch_size, height=args.patch_size)
            loss_final = corner_criterion(final_pred, targets_flat, width=args.patch_size, height=args.patch_size)
            loss = 0.5 * loss_coarse + 1.0 * loss_final
            
            loss.backward()
            optimizer.step()

            tracker.record_batch('train', loss.item(), sync_time() - batch_start)
            train_pbar.set_postfix({'loss': f"{loss.item():.5f}"})
        
        tracker.end_train_phase()
        avg_train_loss = sum(tracker.train_losses) / len(tracker.train_losses)

        # === Multi-Mode Validation ===
        def validate(loader, name, is_top_tracker=False):
            model.eval()
            errors = []
            losses = []
            top_tracker = TopLossTracker(k=5) if is_top_tracker else None
            
            val_pbar = tqdm(loader, desc=f"Val {name}", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    batch = move_batch_to_device(batch, device)
                    patches = batch['patches']
                    targets = batch['targets']
                    B = patches.size(0)
                    
                    patches_flat = patches.view(B * 4, 3, args.patch_size, args.patch_size)
                    targets_flat = targets.view(B * 4, 2)

                    # In eval mode, model now returns (final, coarse) for TS consistency
                    final_pred, coarse_pred = model(patches_flat)
                    v_loss = corner_criterion(final_pred, targets_flat, width=args.patch_size, height=args.patch_size)
                    losses.append(v_loss.item())
                    
                    # Pixel error in patch coordinates
                    diff = (final_pred - targets_flat) * args.patch_size
                    dist = torch.norm(diff, dim=-1) # [B*4]
                    errors.append(dist.cpu())
                    
                    if top_tracker:
                        for i in range(dist.size(0)):
                            top_tracker.update(dist[i].item(), {
                                'image': patches_flat[i],
                                'pred': final_pred[i:i+1],
                                'secondary': coarse_pred[i:i+1],
                                'gt': targets_flat[i:i+1],
                                'path': batch['img_path'][i // 4]
                            })
            
            all_errors = torch.cat(errors, dim=0)
            metrics = calculate_accuracy_metrics(all_errors)
            avg_loss = sum(losses) / len(losses)
            return metrics, avg_loss, top_tracker

        metrics_clean, loss_clean, _ = validate(val_loader_clean, "Clean")
        metrics_jitter, loss_jitter, jitter_top = validate(val_loader_jittered, "Jittered", is_top_tracker=True)
        
        # Logging Summary Table
        logger.info(f"LR: {current_lr:.6f} | Jitter: {train_loader.dataset.jitter_px:.1f}px")
        header = f"{'Phase':<15} | {'Loss':<10} | {'Mean (px)':<10} | {'<1px (%)':<10} | {'<2px (%)':<10} | {'<3px (%)':<10}"
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        logger.info(f"{'Train':<15} | {avg_train_loss:<10.5f} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10}")
        fmt = lambda m, l: f"{l:<10.5f} | {m['mean']:<10.3f} | {m['acc_1px']:<10.1f} | {m['acc_2px']:<10.1f} | {m['acc_3px']:<10.1f}"
        logger.info(f"{'Clean Val':<15} | " + fmt(metrics_clean, loss_clean))
        logger.info(f"{'Jittered Val':<15} | " + fmt(metrics_jitter, loss_jitter))
        logger.info("-" * len(header))
        
        # Jitter Curriculum: reduce jitter in last 10 epochs
        if (epoch + 1) > (args.epochs - 10):
            frac = (epoch + 1 - (args.epochs - 10)) / 10.0
            # Target 5px jitter in final epoch
            train_loader.dataset.jitter_px = max(5.0, args.jitter_px * (1.0 - frac * 0.66))

        scheduler.step()

        # Save Visualizations for Top Jittered Losses
        epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1:03d}")
        if jitter_top is not None:
            for i, sample in enumerate(jitter_top.get_samples()):
                save_diagnostic_visualization(
                    sample['image'], sample['pred'], sample['gt'],
                    None, None,
                    sample['path'], epoch_vis_dir,
                    secondary_corners=sample.get('secondary')
                )
        
        # Checkpoints based on Jittered Mean Error
        latest_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics_jitter['mean'], latest_path)
        
        if metrics_jitter['mean'] < best_jittered_error:
            best_jittered_error = metrics_jitter['mean']
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_jittered_error, best_path)
            logger.info(f"New best jittered model: {best_path} (mean_err={best_jittered_error:.3f})")

    total_time = (sync_time() - global_start_time) / 60
    logger.info(f"=== Training finished in {total_time:.1f}m. Best Jittered Mean Error: {best_jittered_error:.3f} ===")


if __name__ == "__main__":
    main()
