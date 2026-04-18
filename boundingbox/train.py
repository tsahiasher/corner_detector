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

from boundingbox.models.boundingbox_quad_net import BoundingBoxQuadNet
from boundingbox.datasets.yolo_keypoint_dataset import YOLOKeypointDataset, collate_fn_pad
from boundingbox.datasets.coco_val_dataset import COCOValDataset
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time
from common.logging_utils import TrainingTracker


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('boundingbox_train')
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
    parser = argparse.ArgumentParser(description="Train Spatially Aware Regressor (Stage 1).")
    parser.add_argument('--train_images', type=str, default='../crop-dataset-eitan-yolo/images/train', help="Path to training images.")
    parser.add_argument('--val_images', type=str, default='../crop-dataset-eitan-yolo/images/val', help="Path to validation images.")
    parser.add_argument('--val_json', type=str, default='../crop-dataset-eitan-yolo/annotations/val.json', help="Path to validation COCO JSON.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--image_size', type=int, default=384, help="Fixed square input size.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--runs_dir', type=str, default='./boundingbox/runs', help="Base directory for training runs.")
    parser.add_argument('--name', type=str, default='bounding_box', help="Optional suffix for the run directory.")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from.")
    
    # Performance
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True, help="Use pinned memory for faster GPU transfer.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    
    # Loss Weights
    parser.add_argument('--w_box', type=float, default=1.0, help='SmoothL1 bounding-box loss weight.')
    parser.add_argument('--w_giou', type=float, default=1.0, help='Generalized IoU bounding-box loss weight.')
    parser.add_argument('--w_sub', type=float, default=0.3, help='Penalizes confident sub-region enclosures.')
    parser.add_argument('--w_size', type=float, default=0.05, help='Direct unidirectional dimensionality prior explicitly punishing smaller bounds.')
    parser.add_argument('--gt_pad', type=float, default=0.05, help='Padding extension margin implicitly expanded around native dataset coordinates.')
    parser.add_argument('--w_loc', type=float, default=1.0, help='Weight for spatial soft-attention localization mask loss.')

    add_device_args(parser, default='auto')
    return parser.parse_args()


def calculate_iou_components(pred_box, gt_box):
    """Calculates internal bounding configurations, IoU and GIoU bounds inline."""
    # box format: cx, cy, w, h
    px1 = pred_box[:, 0] - pred_box[:, 2] / 2
    py1 = pred_box[:, 1] - pred_box[:, 3] / 2
    px2 = pred_box[:, 0] + pred_box[:, 2] / 2
    py2 = pred_box[:, 1] + pred_box[:, 3] / 2
    
    gx1 = gt_box[:, 0] - gt_box[:, 2] / 2
    gy1 = gt_box[:, 1] - gt_box[:, 3] / 2
    gx2 = gt_box[:, 0] + gt_box[:, 2] / 2
    gy2 = gt_box[:, 1] + gt_box[:, 3] / 2

    # Intersect Area
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter = inter_w * inter_h

    # Union Area
    area_pred = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_gt = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_pred + area_gt - inter + 1e-6
    iou = inter / union

    # Convex Hull Area (Smallest Enclosing Box)
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6
    
    giou = iou - (c_area - union) / c_area
    giou_loss = 1.0 - giou

    return iou, giou_loss, inter, area_pred, area_gt

def draw_diagnostic_boxes(image_t, pred_box, gt_box, path, out_dir):
    try:
        import os
        import cv2
        import numpy as np
        from common.transforms import denormalize_image
    except ImportError:
        return

    # Original Image
    img = denormalize_image(image_t.cpu()).permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    def box_to_poly(b, ww, hh):
        cx, cy, bw, bh = b
        x1, y1 = int((cx - bw/2) * ww), int((cy - bh/2) * hh)
        x2, y2 = int((cx + bw/2) * ww), int((cy + bh/2) * hh)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    gt_px = box_to_poly(gt_box, w, h)
    cv2.polylines(img, [gt_px], True, (255, 0, 0), 2)  # Blue = GT

    if pred_box is not None:
        pred_px = box_to_poly(pred_box, w, h)
        cv2.polylines(img, [pred_px], True, (0, 165, 255), 2)  # Orange = Pred

    name = os.path.basename(path)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"diag_{name}"), img)


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
    
    if os.name == 'nt' and device.type == 'cpu' and args.num_workers > 0:
        logger.warning("Windows CPU detected: setting num_workers=0 to prevent shared file mapping error <1455>.")
        args.num_workers = 0

    logger.info("Initializing datasets...")
    # is_train=False suppresses arbitrary dynamic scaling and rotations. Pure mapping only.
    train_dataset = YOLOKeypointDataset(args.train_images, image_size=args.image_size, min_size=None, max_size=None, is_train=False)
    val_dataset = COCOValDataset(args.val_images, args.val_json, image_size=args.image_size, min_size=None, max_size=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn_pad)

    model = BoundingBoxQuadNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: Spatially-Aware Regressor | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion_box = nn.SmoothL1Loss(reduction='mean')
    criterion_loc = nn.BCEWithLogitsLoss(reduction='mean')

    start_epoch = 0
    best_mean_iou = 0.0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_mean_iou = checkpoint.get('best_metric', 0.0)
            logger.info(f"Restored checkpoint at epoch {start_epoch} (best IoU: {best_mean_iou:.4f})")
    
    global_start_time = sync_time()
    for epoch in range(start_epoch, args.epochs):
        tracker.start_epoch()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr

        # === Training ===
        model.train()
        tracker.start_train_phase()
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for batch in train_pbar:
            batch_start = sync_time()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            
            # Forward pass
            out_dict = model(batch['image'])
            pred_box = out_dict['box'] # [B, 4]
            loc_logits = out_dict['loc_logits'] # [B, 1, H, W]
            
            # Ground truth targets
            gt_corners = batch['corners'] # [B, 4, 2]
            gx_min = gt_corners[:, :, 0].min(1).values
            gy_min = gt_corners[:, :, 1].min(1).values
            gx_max = gt_corners[:, :, 0].max(1).values
            gy_max = gt_corners[:, :, 1].max(1).values
            
            gt_cx = (gx_min + gx_max) / 2
            gt_cy = (gy_min + gy_max) / 2
            gt_w = (gx_max - gx_min).clamp(min=1e-6)
            gt_h = (gy_max - gy_min).clamp(min=1e-6)
            
            # Sub-Region Fix: Expand box natively by padding prior to boundaries constraints
            raw_gt_w = gt_w * (1.0 + args.gt_pad)
            raw_gt_h = gt_h * (1.0 + args.gt_pad)
            nx1 = (gt_cx - raw_gt_w / 2).clamp(min=0.0, max=1.0)
            ny1 = (gt_cy - raw_gt_h / 2).clamp(min=0.0, max=1.0)
            nx2 = (gt_cx + raw_gt_w / 2).clamp(min=0.0, max=1.0)
            ny2 = (gt_cy + raw_gt_h / 2).clamp(min=0.0, max=1.0)
            
            p_cx = (nx1 + nx2) / 2
            p_cy = (ny1 + ny2) / 2
            p_w = (nx2 - nx1).clamp(min=1e-6)
            p_h = (ny2 - ny1).clamp(min=1e-6)
            gt_box = torch.stack([p_cx, p_cy, p_w, p_h], dim=1)
            
            # Localization Mask Bounds
            B_b, _, H, W = loc_logits.size()
            y_grid = torch.arange(H, device=device).view(-1, 1).float() + 0.5
            x_grid = torch.arange(W, device=device).view(1, -1).float() + 0.5
            y_norm = y_grid / H
            x_norm = x_grid / W

            loc_target = torch.zeros_like(loc_logits)
            for B_idx in range(B_b):
                in_x = (x_norm >= nx1[B_idx]) & (x_norm <= nx2[B_idx])
                in_y = (y_norm >= ny1[B_idx]) & (y_norm <= ny2[B_idx])
                loc_target[B_idx, 0] = (in_y * in_x).float()
            
            # Losses
            loss_l1 = criterion_box(pred_box, gt_box) * args.w_box
            _, giou_l, inter, area_pred, area_gt = calculate_iou_components(pred_box, gt_box)
            loss_giou = giou_l.mean() * args.w_giou
            loss_loc = criterion_loc(loc_logits, loc_target) * args.w_loc
            
            # Sub-Region Check: active strictly when network safely isolates a small dense subsection perfectly enclosed natively.
            inside_ratio = inter / (area_pred + 1e-6)
            size_ratio = area_pred / (area_gt + 1e-6)

            inside_mask = (inside_ratio > 0.9).float()
            small_mask = (size_ratio < 0.75).float()

            sub_penalty = inside_mask * small_mask * (1.0 - size_ratio)
            sub_loss_val = sub_penalty.mean() * args.w_sub

            # Unidirectional Target Dimensionality Bounds explicitly pushing geometry larger if deficient
            width_ratio = pred_box[:, 2] / (gt_box[:, 2] + 1e-6)
            height_ratio = pred_box[:, 3] / (gt_box[:, 3] + 1e-6)

            width_small = torch.nn.functional.relu(0.85 - width_ratio)
            height_small = torch.nn.functional.relu(0.85 - height_ratio)

            size_loss_val = (width_small + height_small).mean() * args.w_size
            total_loss = loss_l1 + loss_giou + sub_loss_val + size_loss_val + loss_loc
            
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Detailed component logging
            components = {
                'box_l1': loss_l1.item(),
                'box_giou': loss_giou.item(),
                'box_sub': sub_loss_val.item(),
                'box_size': size_loss_val.item(),
                'loc_bce': loss_loc.item()
            }
            tracker.record_batch('train', total_loss.item(), sync_time() - batch_start, components=components)
            train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        tracker.end_train_phase()
        torch.cuda.empty_cache()

        # === Validation ===
        model.eval()
        tracker.start_val_phase()
        
        all_iou = []
        all_vis_data = []        
        
        val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.inference_mode():
            for batch in val_pbar:
                batch_start = sync_time()
                batch = move_batch_to_device(batch, device)
                
                # Forward pass
                out_dict = model(batch['image'])
                pred_box = out_dict['box'] # [B, 4]
                loc_logits = out_dict['loc_logits']
                
                # Ground truth targets
                gt_corners = batch['corners']
                gx_min = gt_corners[:, :, 0].min(1).values
                gy_min = gt_corners[:, :, 1].min(1).values
                gx_max = gt_corners[:, :, 0].max(1).values
                gy_max = gt_corners[:, :, 1].max(1).values
                
                gt_cx = (gx_min + gx_max) / 2
                gt_cy = (gy_min + gy_max) / 2
                gt_w = (gx_max - gx_min).clamp(min=1e-6)
                gt_h = (gy_max - gy_min).clamp(min=1e-6)
                
                # Sub-Region Fix: Expand box natively by padding prior to boundaries constraints
                raw_gt_w = gt_w * (1.0 + args.gt_pad)
                raw_gt_h = gt_h * (1.0 + args.gt_pad)
                nx1 = (gt_cx - raw_gt_w / 2).clamp(min=0.0, max=1.0)
                ny1 = (gt_cy - raw_gt_h / 2).clamp(min=0.0, max=1.0)
                nx2 = (gt_cx + raw_gt_w / 2).clamp(min=0.0, max=1.0)
                ny2 = (gt_cy + raw_gt_h / 2).clamp(min=0.0, max=1.0)
                
                p_cx = (nx1 + nx2) / 2
                p_cy = (ny1 + ny2) / 2
                p_w = (nx2 - nx1).clamp(min=1e-6)
                p_h = (ny2 - ny1).clamp(min=1e-6)
                gt_box = torch.stack([p_cx, p_cy, p_w, p_h], dim=1)
                
                # Localization Mask Bounds
                B_b, _, H, W = loc_logits.size()
                y_grid = torch.arange(H, device=device).view(-1, 1).float() + 0.5
                x_grid = torch.arange(W, device=device).view(1, -1).float() + 0.5
                y_norm = y_grid / H
                x_norm = x_grid / W

                loc_target = torch.zeros_like(loc_logits)
                for B_idx in range(B_b):
                    in_x = (x_norm >= nx1[B_idx]) & (x_norm <= nx2[B_idx])
                    in_y = (y_norm >= ny1[B_idx]) & (y_norm <= ny2[B_idx])
                    loc_target[B_idx, 0] = (in_y * in_x).float()
                
                loss_l1 = criterion_box(pred_box, gt_box) * args.w_box
                iou, giou_l, inter, area_pred, area_gt = calculate_iou_components(pred_box, gt_box)
                loss_giou = giou_l.mean() * args.w_giou
                loss_loc = criterion_loc(loc_logits, loc_target) * args.w_loc
                
                inside_ratio = inter / (area_pred + 1e-6)
                size_ratio = area_pred / (area_gt + 1e-6)

                inside_mask = (inside_ratio > 0.9).float()
                small_mask = (size_ratio < 0.75).float()

                sub_penalty = inside_mask * small_mask * (1.0 - size_ratio)
                sub_loss_val = sub_penalty.mean() * args.w_sub
                width_ratio = pred_box[:, 2] / (gt_box[:, 2] + 1e-6)
                height_ratio = pred_box[:, 3] / (gt_box[:, 3] + 1e-6)

                width_small = torch.nn.functional.relu(0.85 - width_ratio)
                height_small = torch.nn.functional.relu(0.85 - height_ratio)

                size_loss_val = (width_small + height_small).mean() * args.w_size
                val_loss = loss_l1 + loss_giou + sub_loss_val + size_loss_val + loss_loc
                
                components = {
                    'box_l1': loss_l1.item(),
                    'box_giou': loss_giou.item(),
                    'box_sub': sub_loss_val.item(),
                    'box_size': size_loss_val.item(),
                    'loc_bce': loss_loc.item()
                }
                tracker.record_batch('val', val_loss.item(), sync_time() - batch_start, components=components)
                all_iou.append(iou.cpu())
                
                for b_idx in range(batch['image'].size(0)):
                    all_vis_data.append({
                        'iou': iou[b_idx].item(),
                        'pred_box': pred_box[b_idx].cpu().numpy(),
                        'gt_box': gt_box[b_idx].cpu().numpy(),
                        'path': batch['img_path'][b_idx],
                    })

        tracker.end_val_phase()
        torch.cuda.empty_cache()

        # Metrics
        ious = torch.cat(all_iou, dim=0)
        mean_iou = ious.mean().item()
        metrics = {
            'mean_iou': mean_iou,
            'median_iou': ious.median().item(),
            'iou_05': (ious > 0.5).float().mean().item(),
            'iou_75': (ious > 0.75).float().mean().item(),
        }
        
        # Logging metrics
        tracker.log_epoch_summary(epoch + 1, args.epochs, current_lr, metrics)

        # Save Visualizations (Top and worst IoU)
        epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1:03d}")
        
        all_vis_data.sort(key=lambda x: x['iou']) # worst to best
        worst_5 = all_vis_data[:5]
        best_5 = all_vis_data[-5:]
        
        from PIL import Image
        from common.transforms import get_train_transforms
        vis_transforms = get_train_transforms(image_size=args.image_size, min_size=None, max_size=None, is_train=False)

        def dump_vis_set(subset, prefix):
            for i, sample in enumerate(subset):
                try:
                    raw_img = Image.open(sample['path']).convert("RGB")
                except:
                    continue
                img_t, _ = vis_transforms(raw_img, [])
                draw_diagnostic_boxes(img_t, sample['pred_box'], sample['gt_box'], f"{prefix}_{i}_iou_{sample['iou']:.2f}_{os.path.basename(sample['path'])}", epoch_vis_dir)

        dump_vis_set(worst_5, "worst")
        dump_vis_set(best_5, "best")

        # Checkpoint
        latest_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, mean_iou, latest_path)
        
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_mean_iou, best_path)
            logger.info(f"New best model: {best_path} (IoU={best_mean_iou:.4f})")
            
        scheduler.step()

    total_time = (sync_time() - global_start_time) / 60
    logger.info(f"Training finished in {total_time:.1f}m. Best Mean IoU: {best_mean_iou:.4f}")

if __name__ == "__main__":
    main()
