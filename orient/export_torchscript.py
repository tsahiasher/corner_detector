import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch

from orient.models.orient_net import OrientNet
from common.checkpoint import load_checkpoint
from common.device import add_device_args, resolve_device, log_device_info


def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('orient_export')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export OrientNet to TorchScript.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--weights',  type=str, required=True,
                        help='Path to OrientNet checkpoint (.pt).')
    parser.add_argument('--output',   type=str, default='orient_net.pt',
                        help='Output TorchScript path.')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Input crop size used during tracing.')
    parser.add_argument('--export_method', type=str,
                        choices=['trace', 'script'], default='trace',
                        help='TorchScript export method.')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify exported model matches eager output.')
    parser.add_argument('--verify_runs', type=int, default=3,
                        help='Number of random verification runs.')
    parser.add_argument('--run_dir', type=str, default='',
                        help='Run directory for logs. Inferred from weights if empty.')
    add_device_args(parser, default='cpu')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Place output beside checkpoint if not an absolute path
    if not os.path.isabs(args.output) and os.path.dirname(args.output) == '':
        args.output = os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                                   args.output)

    from datetime import datetime
    if args.run_dir:
        run_dir = args.run_dir
    else:
        weights_dir = os.path.dirname(os.path.abspath(args.weights))
        parent_dir  = os.path.dirname(weights_dir)
        run_dir = parent_dir if os.path.basename(weights_dir) == 'checkpoints' else \
                  os.path.join('.', 'orient', 'runs', f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(os.path.join(log_dir, 'export.log'))
    logger.info('=== OrientNet TorchScript Export ===')
    logger.info(f'Run directory: {run_dir}')

    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    model = OrientNet(num_classes=4).to(device)
    logger.info(f'Loading weights: {args.weights}')
    load_checkpoint(model, None, None, args.weights, device=device)
    model.eval()

    input_shape = (1, 3, args.crop_size, args.crop_size)
    dummy = torch.randn(*input_shape).to(device)
    logger.info(f'Trace input shape: {input_shape}')

    logger.info(f'Exporting with method: {args.export_method.upper()}')
    if args.export_method == 'trace':
        try:
            with torch.inference_mode():
                exported = torch.jit.trace(model, dummy)
            logger.info('Trace successful.')
        except Exception as e:
            logger.error(f'Trace failed: {e}')
            return
    else:
        try:
            exported = torch.jit.script(model)
            logger.info('Script successful.')
        except Exception as e:
            logger.error(f'Script failed: {e}')
            return

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    exported.save(args.output)
    logger.info(f'Saved TorchScript model to: {args.output}')

    # ── Verification ─────────────────────────────────────────────────────
    if args.verify:
        logger.info(f'--- Verification ({args.verify_runs} runs) ---')
        max_err = 0.0
        with torch.inference_mode():
            for _ in range(args.verify_runs):
                v_in = torch.randn(*input_shape).to(device)
                err  = torch.abs(model(v_in) - exported(v_in)).max().item()
                max_err = max(max_err, err)
        logger.info(f'Max absolute logit error: {max_err:.6e}')
        if max_err < 1e-4:
            logger.info('Verification passed.')
        else:
            logger.warning('Precision exceeds threshold (1e-4)!')

    # ── Reload sanity check ───────────────────────────────────────────────
    logger.info('--- Reload Sanity Check ---')
    try:
        reloaded = torch.jit.load(args.output, map_location='cpu')
        reloaded.eval()
        with torch.inference_mode():
            out = reloaded(torch.randn(*input_shape))
        logger.info(f'Reload OK. Output shape: {out.shape}')
    except Exception as e:
        logger.error(f'Reload check failed: {e}')
        return

    logger.info('=== Export Completed Successfully ===')


if __name__ == '__main__':
    main()
