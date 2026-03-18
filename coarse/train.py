import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import logging
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
from common.metrics import compute_pixel_error, calculate_accuracy_metrics
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time

def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and a specified file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('train_pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear redundant handlers if already exist in environment
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
    
    logger.info(f"Logging initialized. Writing logs to {log_file}")
    return logger

def parse_args() -> argparse.Namespace:
    """Parses command line configuration arguments."""
    parser = argparse.ArgumentParser(description="Train Coarse Quad Net (Stage 1).")
    parser.add_argument('--train_images', type=str, default='../../crop-dataset-eitan-yolo/images/train', help="Path to training images directory.")
    parser.add_argument('--val_images', type=str, default='../../crop-dataset-eitan-yolo/images/val', help="Path to validation images directory.")
    parser.add_argument('--val_annotations', type=str, default='../../crop-dataset-eitan-yolo/annotations/val.json', help="Path to validation annotations JSON file.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--runs_dir', type=str, default='./runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--resume', type=str, default='', help="Path to a checkpoint to resume training from.")
    parser.add_argument('--log_freq', type=int, default=10, help="Logging frequency (in batches) during training.")
    add_device_args(parser, default='auto')
    return parser.parse_args()

def main() -> None:
    """Main training loop."""
    args = parse_args()
    
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.name}" if args.name else timestamp
    run_dir = os.path.join(args.runs_dir, run_name)
    
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Setup robust logging
    logger = setup_logging(os.path.join(log_dir, 'train.log'))
    logger.info(f"Run directory mapped to: {run_dir}")
    
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    logger.info("Initializing datasets...")
    train_dataset = YOLOKeypointDataset(args.train_images, image_size=384, is_train=True)
    val_dataset = COCOValDataset(args.val_images, args.val_annotations, image_size=384)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    logger.info(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    
    model = CoarseQuadNet().to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.SmoothL1Loss()
    
    start_epoch = 0
    best_mean_error = float('inf')
    best_epoch = 0
    
    if args.resume:
        logger.info(f"Attempting to resume from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint is not None:
            start_epoch = checkpoint.get('epoch', 0)
            best_mean_error = checkpoint.get('best_metric', checkpoint.get('val_loss', float('inf')))
            best_epoch = start_epoch
            
            logger.info(f"Successfully resumed at epoch {start_epoch}")
            logger.info(f"Restored best metric (mean px error): {best_mean_error:.2f}")
            logger.info(f"Restored optimizer state: {'optimizer_state_dict' in checkpoint}")
            logger.info(f"Restored scheduler state: {'scheduler_state_dict' in checkpoint}")
        else:
            logger.warning("Could not properly resume checkpoint! Starting from scratch.")

    global_start_time = sync_time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = sync_time()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        # We explicitly supervise the score head to 1.0 (since 1 card is guaranteed present)
        bce_criterion = nn.BCELoss()
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            target_corners = batch['corners']
            
            optimizer.zero_grad()
            pred_score, pred_corners = model(images)
            
            # Corner loss + Score loss
            l1_loss = criterion(pred_corners, target_corners)
            score_target = torch.ones_like(pred_score)
            score_loss = bce_criterion(pred_score, score_target)
            
            loss = l1_loss + score_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            running_loss = train_loss / ((batch_idx + 1) * images.size(0))
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.5f}", 
                'avg_loss': f"{running_loss:.5f}", 
                'lr': f"{current_lr:.6f}"
            })
            
        train_loss /= len(train_loader.dataset)
        train_time = sync_time() - epoch_start_time
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_errors = []
        
        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                batch = move_batch_to_device(batch, device)
                images = batch['image']
                target_corners = batch['corners']
                w_list, h_list = batch.get('orig_size', ([384]*images.size(0), [384]*images.size(0)))
                
                pred_score, pred_corners = model(images)
                
                l1_loss = criterion(pred_corners, target_corners)
                score_target = torch.ones_like(pred_score)
                loss = l1_loss + bce_criterion(pred_score, score_target)
                
                val_loss += loss.item() * images.size(0)
                
                # Compute per-image pixel errors
                for b_idx in range(images.size(0)):
                    w = w_list[b_idx].item()
                    h = h_list[b_idx].item()
                    err = compute_pixel_error(pred_corners[b_idx].unsqueeze(0), target_corners[b_idx].unsqueeze(0), w, h)
                    all_errors.append(err)
                    
        val_time = sync_time() - train_time - epoch_start_time
        val_loss /= len(val_loader.dataset)
        cat_errors = torch.cat(all_errors, dim=0) # [N, 4]
        val_metrics = calculate_accuracy_metrics(cat_errors)
        
        mean_px_err = val_metrics.get('mean_error', float('inf'))
        median_px_err = val_metrics.get('median_error', float('inf'))
        
        # Step LR Scheduler
        scheduler.step()
        
        # Timing
        epoch_duration = train_time + val_time
        avg_train_batch_time = train_time / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_batch_time = val_time / len(val_loader) if len(val_loader) > 0 else 0
        
        # Log Consolidated Results
        logger.info(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.1f}s (Train: {train_time:.1f}s | Val: {val_time:.1f}s)")
        logger.info(f"Train: loss={train_loss:.5f} | lr={current_lr:.6f} | avg_batch_time={avg_train_batch_time:.3f}s")
        logger.info(f"Val: loss={val_loss:.5f} | mean_px={mean_px_err:.2f} | median_px={median_px_err:.2f} | avg_batch_time={avg_val_batch_time:.3f}s")
        
        # Detailed metrics
        logger.info(f"Corners (mean px): TL={val_metrics.get('tl_mean',0):.2f} | TR={val_metrics.get('tr_mean',0):.2f} | BR={val_metrics.get('br_mean',0):.2f} | BL={val_metrics.get('bl_mean',0):.2f}")
        logger.info(f"Outliers (px): p90={val_metrics.get('p90_error',0):.2f} | p95={val_metrics.get('p95_error',0):.2f} | max={val_metrics.get('max_error',0):.2f}")
        
        threshold_metrics = []
        for key, val in val_metrics.items():
            if key.startswith('acc_under_'):
                threshold = key.split('_')[-1]
                threshold_metrics.append(f"<{threshold}={val:.1f}%")
        
        if threshold_metrics:
            logger.info(f"Thresholds: {' | '.join(threshold_metrics)}")
        
        # Save checkpoints
        latest_path = os.path.join(checkpoint_dir, 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch+1, mean_px_err, latest_path)
        logger.info(f"Last checkpoint saved: {latest_path}")
        
        if mean_px_err < best_mean_error:
            best_mean_error = mean_px_err
            best_epoch = epoch + 1
            best_path = os.path.join(checkpoint_dir, 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch+1, best_mean_error, best_path)
            logger.info(f"New best model saved: {best_path} | mean_px={best_mean_error:.2f} | epoch={best_epoch}")
            
        logger.info(f"Best so far: mean_px={best_mean_error:.2f} at epoch {best_epoch}\n" + "-"*50)

    total_time = sync_time() - global_start_time
    logger.info(f"=== Training finished! Total time: {total_time/60:.2f} minutes ===")
    logger.info(f"=== Best Mean px Error: {best_mean_error:.2f} at epoch {best_epoch} ===")

if __name__ == "__main__":
    main()
