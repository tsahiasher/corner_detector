import os
import sys
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
from common.metrics import compute_pixel_error, calculate_accuracy_metrics, compute_patch_recall, WingLoss, QuadShapeLoss
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.visualization import save_indexed_visualization


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and a specified file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('train_pipeline')
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

    logger.info(f"Logging initialized. Writing logs to {log_file}")
    return logger


def parse_args() -> argparse.Namespace:
    """Parses command line configuration arguments."""
    parser = argparse.ArgumentParser(description="Train Coarse Quad Net (Stage 1).")
    parser.add_argument('--train_images', type=str, default='../crop-dataset-eitan-yolo/images/train', help="Path to training images directory.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Path to validation images directory.")
    parser.add_argument('--val_annotations', type=str, default='../crop-dataset-eitan-yolo/annotations/val.json', help="Path to validation annotations JSON file.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--runs_dir', type=str, default='./coarse/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--resume', type=str, default='', help="Path to a checkpoint to resume training from.")
    parser.add_argument('--log_freq', type=int, default=10, help="Logging frequency (in batches) during training.")
    parser.add_argument('--shape_loss_weight', type=float, default=0.1, help="Weight for quadrilateral shape regularization loss.")
    add_device_args(parser, default='auto')
    return parser.parse_args()


def main() -> None:
    """Main training loop with Wing loss, shape regularization, and patch recall metrics."""
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

    logger = setup_logging(os.path.join(log_dir, 'train.log'))
    logger.info(f"Run directory: {run_dir}")

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
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Losses
    corner_criterion = WingLoss(wing_w=10.0, epsilon=2.0)
    shape_criterion = QuadShapeLoss(weight_diag=1.0, weight_convex=0.5)
    bce_criterion = nn.BCELoss()

    logger.info(f"Losses: WingLoss(w=10, eps=2) + QuadShapeLoss(weight={args.shape_loss_weight}) + BCELoss(score)")

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
            logger.info(f"Resumed at epoch {start_epoch}, best metric: {best_mean_error:.2f}")
        else:
            logger.warning("Could not resume checkpoint. Starting from scratch.")

    global_start_time = sync_time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = sync_time()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr

        # === Training Phase ===
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            batch = move_batch_to_device(batch, device)
            images = batch['image']
            target_corners = batch['corners']
            w_list = batch.get('orig_width', torch.full((images.size(0),), 384.0, device=device))
            h_list = batch.get('orig_height', torch.full((images.size(0),), 384.0, device=device))

            optimizer.zero_grad()
            pred_score, pred_corners = model(images)

            # Wing loss for corner coordinates with independent X/Y scaling
            wing_loss = corner_criterion(pred_corners, target_corners, width=w_list, height=h_list)

            # Geometry-aware shape regularization
            shape_loss = shape_criterion(pred_corners)

            # Score supervision (card always present)
            score_target = torch.ones_like(pred_score)
            score_loss = bce_criterion(pred_score, score_target)

            loss = wing_loss + args.shape_loss_weight * shape_loss + score_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            running_loss = train_loss / ((batch_idx + 1) * images.size(0))

            train_pbar.set_postfix({
                'loss': f"{loss.item():.5f}",
                'avg': f"{running_loss:.5f}",
                'lr': f"{current_lr:.6f}"
            })

        train_loss /= len(train_loader.dataset)
        train_time = sync_time() - epoch_start_time

        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        all_errors = []
        all_patch_recalls = {64: [], 80: [], 96: []}
        # Collect per-image data for top-k visualization
        per_image_records = []  # list of (mean_error, image, pred, target, w, h, path)

        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                batch = move_batch_to_device(batch, device)
                images = batch['image']
                target_corners = batch['corners']
                w_list = batch.get('orig_width', torch.full((images.size(0),), 384.0, device=device))
                h_list = batch.get('orig_height', torch.full((images.size(0),), 384.0, device=device))
                img_paths = batch.get('img_path', [''] * images.size(0))

                pred_score, pred_corners = model(images)

                wing_loss = corner_criterion(pred_corners, target_corners, width=w_list, height=h_list)
                shape_loss = shape_criterion(pred_corners)
                score_target = torch.ones_like(pred_score)
                loss = wing_loss + args.shape_loss_weight * shape_loss + bce_criterion(pred_score, score_target)
                val_loss += loss.item() * images.size(0)

                # Per-image metrics
                for b_idx in range(images.size(0)):
                    w = w_list[b_idx].item()
                    h = h_list[b_idx].item()

                    err = compute_pixel_error(
                        pred_corners[b_idx].unsqueeze(0),
                        target_corners[b_idx].unsqueeze(0), w, h
                    )
                    all_errors.append(err)
                    mean_err = err.mean().item()

                    # Store record for visualization (keep on CPU)
                    per_image_records.append((
                        mean_err,
                        images[b_idx].cpu(),
                        pred_corners[b_idx].cpu(),
                        target_corners[b_idx].cpu(),
                        w, h,
                        img_paths[b_idx] if isinstance(img_paths, list) else img_paths[b_idx]
                    ))

                    pr = compute_patch_recall(
                        pred_corners[b_idx].unsqueeze(0),
                        target_corners[b_idx].unsqueeze(0), w, h,
                        patch_sizes=(64, 80, 96)
                    )
                    for ps in (64, 80, 96):
                        all_patch_recalls[ps].append(pr[f'patch_recall_{ps}px'])

        # Save top-5 highest-loss visualizations
        per_image_records.sort(key=lambda r: r[0], reverse=True)
        vis_dir = os.path.join(run_dir, 'visualizations', f'epoch_{epoch+1}')
        for rank, (m_err, img_t, pred_c, tgt_c, ow, oh, ipath) in enumerate(per_image_records[:5]):
            base = os.path.splitext(os.path.basename(ipath))[0] if ipath else f'image_{rank}'
            save_path = os.path.join(vis_dir, f'rank{rank+1}_{base}.jpg')
            save_indexed_visualization(img_t, pred_c, tgt_c, ow, oh, save_path, img_path=ipath)
        logger.info(f"Saved top-5 loss visualizations to {vis_dir}")

        val_time = sync_time() - epoch_start_time - train_time
        val_loss /= len(val_loader.dataset)
        cat_errors = torch.cat(all_errors, dim=0)
        val_metrics = calculate_accuracy_metrics(cat_errors)

        mean_px_err = val_metrics.get('mean_error', float('inf'))
        median_px_err = val_metrics.get('median_error', float('inf'))

        # Aggregate patch recall
        avg_pr = {}
        for ps in (64, 80, 96):
            vals = all_patch_recalls[ps]
            avg_pr[ps] = sum(vals) / len(vals) if vals else 0.0

        scheduler.step()

        epoch_duration = train_time + val_time
        avg_train_batch_time = train_time / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_batch_time = val_time / len(val_loader) if len(val_loader) > 0 else 0

        # === Logging ===
        logger.info(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.1f}s (Train: {train_time:.1f}s | Val: {val_time:.1f}s)")
        logger.info(f"Train: loss={train_loss:.5f} | lr={current_lr:.6f} | avg_batch={avg_train_batch_time:.3f}s")
        logger.info(f"Val: loss={val_loss:.5f} | mean_px={mean_px_err:.2f} | median_px={median_px_err:.2f} | avg_batch={avg_val_batch_time:.3f}s")

        logger.info(f"Corners (mean px): TL={val_metrics.get('tl_mean',0):.2f} | TR={val_metrics.get('tr_mean',0):.2f} | BR={val_metrics.get('br_mean',0):.2f} | BL={val_metrics.get('bl_mean',0):.2f}")
        logger.info(f"Outliers (px): p90={val_metrics.get('p90_error',0):.2f} | p95={val_metrics.get('p95_error',0):.2f} | max={val_metrics.get('max_error',0):.2f}")

        # Patch recall — THE key metric for stage 1 → stage 2 handoff
        logger.info(f"Patch Recall: 64px={avg_pr[64]:.1f}% | 80px={avg_pr[80]:.1f}% | 96px={avg_pr[96]:.1f}%")

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
