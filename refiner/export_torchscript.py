import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch

from refiner.models.patch_refiner import PatchRefinerNet
from common.checkpoint import load_checkpoint
from common.device import add_device_args, resolve_device, log_device_info


def setup_logging(log_file: str) -> logging.Logger:
    """Configures logging for the export script."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('refiner_export')
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
    """Parses TorchScript export arguments for the refiner model."""
    parser = argparse.ArgumentParser(description="Export PatchRefinerNet to TorchScript.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', type=str, required=True, help="Path to refiner checkpoint (.pt).")
    parser.add_argument('--output', type=str, default='patch_refiner.pt', help="Output TorchScript path.")
    add_device_args(parser, default='cpu')
    parser.add_argument('--patch_size', type=int, default=96, help="Patch input size for tracing.")
    parser.add_argument('--verify', action='store_true', default=True, help="Verify export matches eager model.")
    parser.add_argument('--verify_runs', type=int, default=3, help="Number of verification runs.")
    parser.add_argument('--export_method', type=str, choices=['trace', 'script'], default='trace', help="TorchScript export method.")
    parser.add_argument('--run_dir', type=str, default='', help="Run directory. If empty, inferred from weights path.")
    return parser.parse_args()


def main() -> None:
    """Main export function for PatchRefinerNet."""
    args = parse_args()
    from datetime import datetime

    if args.run_dir:
        run_dir = args.run_dir
    else:
        weights_dir = os.path.dirname(os.path.abspath(args.weights))
        parent_dir = os.path.dirname(weights_dir)
        if os.path.basename(weights_dir) == 'checkpoints':
            run_dir = parent_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join('.', 'runs', 'refiner', f"export_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(os.path.join(log_dir, 'export.log'))
    logger.info("=== PatchRefinerNet TorchScript Export ===")
    logger.info(f"Run directory: {run_dir}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)

    # Load model
    model = PatchRefinerNet().to(device)
    logger.info(f"Loading weights from: {args.weights}")
    load_checkpoint(model, None, None, args.weights, device=device)
    model.eval()

    input_shape = (1, 3, args.patch_size, args.patch_size)
    logger.info(f"Trace input shape: {input_shape}")
    dummy = torch.randn(*input_shape).to(device)

    # Export
    logger.info(f"Exporting with method: {args.export_method.upper()}")
    if args.export_method == 'trace':
        try:
            with torch.inference_mode():
                exported = torch.jit.trace(model, dummy)
            logger.info("Trace successful.")
        except Exception as e:
            logger.error(f"Trace failed: {e}")
            return
    else:
        try:
            exported = torch.jit.script(model)
            logger.info("Script successful.")
        except Exception as e:
            logger.error(f"Script failed: {e}")
            return

    try:
        exported.save(args.output)
        logger.info(f"Saved TorchScript model to: {args.output}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        return

    # Verify
    if args.verify:
        logger.info(f"--- Verification ({args.verify_runs} runs) ---")
        max_err = 0.0
        with torch.inference_mode():
            for i in range(args.verify_runs):
                v_input = torch.randn(*input_shape).to(device)
                eager_out = model(v_input)
                traced_out = exported(v_input)
                err = torch.abs(eager_out - traced_out).max().item()
                max_err = max(max_err, err)

        logger.info(f"Max absolute error: {max_err:.6e}")
        if max_err > 1e-4:
            logger.warning(f"Precision exceeds threshold (1e-4)!")
        else:
            logger.info("Verification passed: traced model matches eager model.")

    # Reload check
    logger.info("--- Reload Sanity Check ---")
    try:
        reloaded = torch.jit.load(args.output, map_location='cpu')
        reloaded.eval()
        cpu_input = torch.randn(*input_shape).to('cpu')
        with torch.inference_mode():
            out = reloaded(cpu_input)
        logger.info(f"Reload successful. Output shape: {out.shape}")
    except Exception as e:
        logger.error(f"Reload check failed: {e}")
        return

    logger.info("=== Export Completed Successfully ===")


if __name__ == "__main__":
    main()
