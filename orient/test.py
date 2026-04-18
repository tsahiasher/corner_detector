import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from orient.models.orient_net import OrientNet
from orient.datasets.orient_dataset import OrientDataset
from common.checkpoint import load_checkpoint
from common.device import add_device_args, resolve_device, log_device_info, move_batch_to_device

ORIENT_LABELS = {0: '  0 Deg', 1: ' 90 Deg', 2: '180 Deg', 3: '270 Deg'}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('orient_test')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate OrientNet on a validation set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--val_images', type=str,
                        default='../crop-dataset-eitan-yolo/images/val',
                        help='Path to validation image directory.')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to OrientNet checkpoint (.pt) or TorchScript (.pt).')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Canonical card crop size.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pytorch', action='store_true',
                        help='Load as PyTorch checkpoint (default: TorchScript).')
    add_device_args(parser, default='cpu')
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    logger = setup_logging()
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    # Load model
    if args.pytorch:
        model = OrientNet(num_classes=4).to(device)
        load_checkpoint(model, None, None, args.weights, device=device)
        model.eval()
    else:
        model = torch.jit.load(args.weights, map_location=device)
        model.eval()

    dataset    = OrientDataset(args.val_images, crop_size=args.crop_size, is_train=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    per_class_correct = [0] * 4
    per_class_total   = [0] * 4
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for c in range(4):
                mask = labels == c
                per_class_total[c]   += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()

    overall_acc = total_correct / max(1, total_samples)
    logger.info('=' * 50)
    logger.info(f'Overall Accuracy: {overall_acc:.2%}  ({total_correct}/{total_samples})')
    logger.info('')
    for c in range(4):
        tot = per_class_total[c]
        acc = per_class_correct[c] / tot if tot > 0 else 0.0
        logger.info(f'  Class {ORIENT_LABELS[c]}: {acc:.2%}  ({per_class_correct[c]}/{tot})')
    logger.info('=' * 50)


if __name__ == '__main__':
    main()
