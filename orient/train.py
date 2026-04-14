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

from orient.models.orient_net import OrientNet
from orient.datasets.orient_dataset import OrientDataset, collate_fn_pad
from common.checkpoint import save_checkpoint, load_checkpoint
from common.seed import set_seed
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device, sync_time


# Rotation labels for readable logging
ORIENT_LABELS = {0: '  0°', 1: ' 90°', 2: '180°', 3: '270°'}


def setup_logging(log_file: str) -> logging.Logger:
    """Configures production-quality logging to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('orient_train')
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
    parser = argparse.ArgumentParser(
        description="Train OrientNet: lightweight card orientation classifier (Stage 2.5)."
    )
    parser.add_argument('--train_images', type=str,
                        default='../crop-dataset-eitan-yolo/images/train',
                        help='Path to training image directory.')
    parser.add_argument('--val_images', type=str,
                        default='../crop-dataset-eitan-yolo/images/val',
                        help='Path to validation image directory.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--min_size', type=int, default=800,
                        help='Minimum image dimension.')
    parser.add_argument('--max_size', type=int, default=1333,
                        help='Maximum image dimension.')
    parser.add_argument('--runs_dir', type=str, default='./orient/runs',
                        help='Base directory for training runs.')
    parser.add_argument('--name', type=str, default='',
                        help='Optional suffix for the run directory.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Label smoothing for cross-entropy loss.')
    add_device_args(parser, default='auto')
    return parser.parse_args()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Returns top-1 accuracy as a fraction [0, 1]."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{timestamp}_{args.name}' if args.name else timestamp
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger = setup_logging(os.path.join(run_dir, 'logs', 'train.log'))
    logger.info(f'Run directory: {run_dir}')

    set_seed(42)
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    if os.name == 'nt' and device.type == 'cpu' and args.num_workers > 0:
        logger.warning("Windows CPU detected: setting num_workers=0 to prevent shared file mapping error <1455>.")
        args.num_workers = 0

    logger.info('Initializing datasets...')
    train_dataset = OrientDataset(args.train_images, min_size=args.min_size, max_size=args.max_size, is_train=True)
    val_dataset   = OrientDataset(args.val_images,   min_size=args.min_size, max_size=args.max_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_pad)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_pad)

    model = OrientNet(num_classes=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f'OrientNet | Parameters: {param_count:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        logger.info(f'Resuming from: {args.resume}')
        ckpt = load_checkpoint(model, optimizer, scheduler, args.resume, device=device)
        if ckpt:
            start_epoch  = ckpt.get('epoch', 0)
            best_val_acc = ckpt.get('best_metric', 0.0)
            logger.info(f'Restored epoch {start_epoch} (best acc={best_val_acc:.2%})')

    global_start = sync_time()
    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr

        # ── Training ─────────────────────────────────────────────────────
        model.train()
        train_loss, train_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f'Train {epoch+1}/{args.epochs}', leave=False)
        for batch in pbar:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc  += compute_accuracy(logits, labels)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n = max(1, len(train_loader))
        train_loss /= n
        train_acc  /= n

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        per_class_correct = [0] * 4
        per_class_total   = [0] * 4

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(imgs)
                vloss  = criterion(logits, labels)
                val_loss += vloss.item()
                val_acc  += compute_accuracy(logits, labels)

                preds = logits.argmax(dim=-1)
                for c in range(4):
                    mask = labels == c
                    per_class_total[c]   += mask.sum().item()
                    per_class_correct[c] += (preds[mask] == c).sum().item()

        n_val = max(1, len(val_loader))
        val_loss /= n_val
        val_acc  /= n_val

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        logger.info('-' * 70)
        logger.info(f'EPOCH {epoch+1:3d}/{args.epochs} | LR: {current_lr:.5f}')
        logger.info(f'  Train  Loss: {train_loss:.4f}  Acc: {train_acc:.2%}')
        logger.info(f'  Val    Loss: {val_loss:.4f}  Acc: {val_acc:.2%}')
        for c in range(4):
            tot = per_class_total[c]
            acc = per_class_correct[c] / tot if tot > 0 else 0.0
            logger.info(f'  Class {ORIENT_LABELS[c]}: {acc:.2%}  ({per_class_correct[c]}/{tot})')

        # ── Checkpoints ───────────────────────────────────────────────────
        last_path = os.path.join(run_dir, 'checkpoints', 'last.pt')
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(run_dir, 'checkpoints', 'best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_val_acc, best_path)
            logger.info(f'  ** New best: {best_path}  (val_acc={best_val_acc:.2%})')

    total_min = (sync_time() - global_start) / 60
    logger.info(f'Training complete in {total_min:.1f}m. Best val acc: {best_val_acc:.2%}')


if __name__ == '__main__':
    main()
