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
from coarse.datasets.yolo_keypoint_dataset import YOLOKeypointDataset, collate_fn_pad
from coarse.datasets.coco_val_dataset import COCOValDataset
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.metrics import (calculate_accuracy_metrics, compute_patch_recall,
                            YOLOPoseLoss)
from common.geometry import sort_corners_clockwise
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
    parser.add_argument('--min_size', type=int, default=800, help="Minimum image dimension.")
    parser.add_argument('--max_size', type=int, default=1333, help="Maximum image dimension.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--runs_dir', type=str, default='./coarse/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='', help="Optional suffix for the run directory.")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from.")
    
    # Performance
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--pin_memory', action='store_true', default=True, help="Use pinned memory for faster GPU transfer.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    
    # YOLO-Pose Loss Weights
    parser.add_argument('--w_obj', type=float, default=1.0, help='Objectness (focal) loss weight.')
    parser.add_argument('--w_box', type=float, default=5.0, help='CIoU bounding-box loss weight.')
    parser.add_argument('--w_kpt', type=float, default=0.2, help='Keypoint regression loss weight.')

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
    logger.info(f"YOLO-Pose Losses: Obj={args.w_obj}, Box={args.w_box}, Kpt={args.w_kpt}")
    
    tracker = TrainingTracker(logger)
    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    if os.name == 'nt' and device.type == 'cpu' and args.num_workers > 0:
        logger.warning("Windows CPU detected: setting num_workers=0 to prevent shared file mapping error <1455>.")
        args.num_workers = 0

    logger.info("Initializing datasets...")
    train_dataset = YOLOKeypointDataset(args.train_images, image_size=None, min_size=args.min_size, max_size=args.max_size, is_train=True)
    # COCO might need updates if not dynamic, using YOLO for both typically, but the original code used COCOValDataset
    # We will pass min_size and max_size to COCOValDataset as well (we need to update COCOValDataset to accept these)
    val_dataset = COCOValDataset(args.val_images, args.val_json, image_size=None, min_size=args.min_size, max_size=args.max_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn_pad)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn_pad)

    model = CoarseQuadNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: YOLO-Pose CoarseQuadNet | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # YOLO-Pose unified loss (objectness + CIoU box + keypoint)
    criterion = YOLOPoseLoss(w_obj=args.w_obj, w_box=args.w_box, w_kpt=args.w_kpt)

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
                                      num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn_pad)
            logger.info(f"Epoch {epoch+1}: Recreated train_loader (Mining Exp: {current_exponent:.2f}).")

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for batch in train_pbar:
            batch_start = sync_time()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            
            out = model(batch['image'])
            gt_corners = batch['corners']
            
            # --- Dynamic Corner Sorting (Physical to Visual) ---
            gt_corners_visual = sort_corners_clockwise(gt_corners)
            
            # YOLO-Pose unified loss
            losses = criterion(out['raw_pred'], gt_corners_visual)
            total_loss = losses['total']
            
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Detailed component logging
            components = {
                'obj': losses['obj'].item(),
                'box': losses['box'].item(),
                'kpt': losses['kpt'].item(),
            }
            tracker.record_batch('train', total_loss.item(), sync_time() - batch_start, components=components)
            train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
            # Update Miner with per-sample errors (Pixel space)
            with torch.no_grad():
                # Get the actual padded width/height used for offsets
                B_idx_temp = batch['image'].size(0)
                pad_h = torch.tensor(batch['image'].shape[2], device=device).float().view(-1, 1).expand(B_idx_temp, 1)
                pad_w = torch.tensor(batch['image'].shape[3], device=device).float().view(-1, 1).expand(B_idx_temp, 1)
                
                scaled_w = batch.get('scaled_width', pad_w[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                orig_w = batch.get('orig_width', pad_w[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                ratio_w = orig_w / scaled_w

                scaled_h = batch.get('scaled_height', pad_h[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                orig_h = batch.get('orig_height', pad_h[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                ratio_h = orig_h / scaled_h
                
                # Use geometric visual corners for geometric mining
                diff = (out['corners_visual'] - gt_corners_visual).abs()
                diff[:, :, 0] *= (pad_w * ratio_w)
                diff[:, :, 1] *= (pad_h * ratio_h)
                
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

            del out, batch, total_loss, components, losses

        tracker.end_train_phase()
        torch.cuda.empty_cache()

        # === Validation ===
        model.eval()
        tracker.start_val_phase()
        all_errs = []
        all_vis_data = []        
        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.inference_mode():
            for batch in val_pbar:
                batch_start = sync_time()
                batch = move_batch_to_device(batch, device)
                out = model(batch['image'])
                gt_corners = batch['corners']
                
                # --- Dynamic Corner Sorting (Physical to Visual) ---
                gt_corners_visual = sort_corners_clockwise(gt_corners)
                
                # YOLO-Pose unified loss
                losses = criterion(out['raw_pred'], gt_corners_visual)
                val_total_loss = losses['total']
                
                components = {
                    'obj': losses['obj'].item(),
                    'box': losses['box'].item(),
                    'kpt': losses['kpt'].item(),
                }
                tracker.record_batch('val', val_total_loss.item(), sync_time() - batch_start, components=components)

                # Accuracy (Pixel Space)
                # Compute absolute pixel error on the padded grid shape to match predicted scale
                B_idx_temp = out['corners'].size(0)
                pad_h = torch.tensor(batch['image'].shape[2], device=device).float().view(-1, 1).expand(B_idx_temp, 1)
                pad_w = torch.tensor(batch['image'].shape[3], device=device).float().view(-1, 1).expand(B_idx_temp, 1)

                scaled_w = batch.get('scaled_width', pad_w[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                orig_w = batch.get('orig_width', pad_w[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                ratio_w = orig_w / scaled_w

                scaled_h = batch.get('scaled_height', pad_h[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                orig_h = batch.get('orig_height', pad_h[:,0]).to(device).view(-1, 1).expand(B_idx_temp, 1)
                ratio_h = orig_h / scaled_h

                diff = (out['corners'] - gt_corners_visual).abs()
                diff[:, :, 0] *= (pad_w * ratio_w)
                diff[:, :, 1] *= (pad_h * ratio_h)
                dist = torch.norm(diff, dim=-1) # [B, 4]
                all_errs.append(dist.cpu())
                
                # Store metadata for all samples (without the 20MB image tensor)
                m_dist = dist.mean(dim=-1)
                for b_idx in range(batch['image'].size(0)):
                    all_vis_data.append({
                        'err': m_dist[b_idx].item(),
                        'pred': out['corners'][b_idx].cpu(),
                        'gt': gt_corners_visual[b_idx].cpu(),
                        'path': batch['img_path'][b_idx],
                        'pad_w': pad_w[b_idx].item(),
                        'pad_h': pad_h[b_idx].item()
                    })
                
                # Explicitly free variables
                del out, batch, components, losses

        tracker.end_val_phase()
        torch.cuda.empty_cache()

        # Metrics
        errors = torch.cat(all_errs, dim=0)
        metrics = calculate_accuracy_metrics(errors)
        recall_val = compute_patch_recall(errors)
        
        # Summary Log
        tracker.log_epoch_summary(epoch + 1, args.epochs, current_lr, metrics, recall_val)
        scheduler.step()

        # Save Visualizations (Top 5 and Median 5)
        epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_vis_dir, exist_ok=True)
        
        all_vis_data.sort(key=lambda x: x['err'], reverse=True)
        top_5 = all_vis_data[:5]
        
        median_idx = len(all_vis_data) // 2
        median_5 = all_vis_data[max(0, median_idx-2) : median_idx+3]
        
        from PIL import Image
        import torchvision.transforms.functional as TF
        from common.transforms import get_train_transforms
        vis_transforms = get_train_transforms(image_size=None, min_size=args.min_size, max_size=args.max_size, is_train=False)

        def dump_vis_set(subset, prefix):
            for i, sample in enumerate(subset):
                # 1. Reload the image and simulate the padding dynamically to match the tensor constraint
                try:
                    raw_img = Image.open(sample['path']).convert("RGB")
                except:
                    continue
                img_t, _ = vis_transforms(raw_img, [])
                c, h, w = img_t.shape
                # Must cast to int as pad_w/pad_h were stored as floats
                pad_w_int = int(sample['pad_w'])
                pad_h_int = int(sample['pad_h'])
                pad_len = (0, max(0, pad_w_int - w), 0, max(0, pad_h_int - h))
                padded_img = torch.nn.functional.pad(img_t, pad_len, value=0.0)
                
                # 2. Draw
                save_diagnostic_visualization(
                    padded_img, sample['pred'], sample['gt'],
                    None, None,
                    f"{prefix}_{i}_err_{sample['err']:.1f}_{os.path.basename(sample['path'])}", epoch_vis_dir
                )

        dump_vis_set(top_5, "top")
        dump_vis_set(median_5, "median")

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
