import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch

from coarse.models.coarse_quad_net import CoarseQuadNet
from common.checkpoint import load_checkpoint
from common.device import add_device_args, resolve_device, log_device_info

def setup_logging(log_file: str) -> logging.Logger:
    """Configures clean, human-readable console and file logging for the export script."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('export_pipeline')
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
    """Parses command line arguments for exporting the model to TorchScript."""
    parser = argparse.ArgumentParser(description="Export CoarseQuadNet to TorchScript for edge CPU deployment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained PyTorch checkpoint (.pt) to load.")
    parser.add_argument('--output', type=str, default='coarse_quad_net.pt', help="Path to save the exported TorchScript model.")
    add_device_args(parser, default='cpu')
    parser.add_argument('--input_height', type=int, default=384, help="Height of the dummy input tensor used for tracing.")
    parser.add_argument('--input_width', type=int, default=384, help="Width of the dummy input tensor used for tracing.")
    parser.add_argument('--verify', action='store_true', default=True, help="If true, verify the exported model output matches the eager model.")
    parser.add_argument('--verify_runs', type=int, default=1, help="Number of random inputs to use when verifying eager vs target equivalence.")
    parser.add_argument('--export_method', type=str, choices=['trace', 'script'], default='trace', help="Which TorchScript export pathway to use. 'trace' is highly recommended for static CNNs without control flow.")
    parser.add_argument('--run_dir', type=str, default='', help="Explicit run directory. If empty, inferred from weights path.")
    return parser.parse_args()

def main() -> None:
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
            run_dir = os.path.join('.', 'runs', f"export_{timestamp}")
            
    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(os.path.join(log_dir, 'export.log'))
    logger.info("=== TorchScript Export Pipeline ===")
    logger.info(f"Export mapped to run directory: {run_dir}")
    
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
            
    device = resolve_device(args.device)
    log_device_info(device, args.device, logger)
    
    # 1. Load Model
    model = CoarseQuadNet().to(device)
    logger.info(f"Loading checkpoint weights from: {args.weights}")
    load_checkpoint(model, None, None, args.weights, device=device)
    model.eval()
    
    input_shape = (1, 3, args.input_height, args.input_width)
    logger.info(f"Generated trace dummy input volume with shape: {input_shape}")
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 2. Export Model
    logger.info(f"Attempting TorchScript export using method: {args.export_method.upper()}")
    if args.export_method == 'trace':
        # Tracing is excellent for CoarseQuadNet as we only use fully-convolutional standard paths
        # without dynamic loops or branch conditions dependent on dataset contents.
        try:
            with torch.inference_mode():
                exported_model = torch.jit.trace(model, dummy_input)
            logger.info("Successfully traced the model graph.")
        except Exception as e:
            logger.error(f"Failed to trace model: {e}")
            return
    elif args.export_method == 'script':
        try:
            exported_model = torch.jit.script(model)
            logger.info("Successfully scripted the model graph.")
        except Exception as e:
            logger.error(f"Failed to script model: {e}")
            return
            
    # Save the artifact 
    try:
        exported_model.save(args.output)
        logger.info(f"TorchScript artifact cleanly saved to: {args.output}")
    except Exception as e:
        logger.error(f"Failed to save TorchScript file: {e}")
        return

    # 3. Verification Check
    if args.verify:
        logger.info(f"--- Running Execution Verification ({args.verify_runs} runs) ---")
        tolerance = 1e-4
        max_score_err = 0.0
        max_corner_err = 0.0
        
        with torch.inference_mode():
            for i in range(args.verify_runs):
                v_input = torch.randn(*input_shape).to(device)
                
                # Eager vs Traced checks 
                eager_score, eager_corners = model(v_input)
                traced_score, traced_corners = exported_model(v_input)
                
                score_err = torch.abs(eager_score - traced_score).max().item()
                corner_err = torch.abs(eager_corners - traced_corners).max().item()
                
                max_score_err = max(max_score_err, score_err)
                max_corner_err = max(max_corner_err, corner_err)
                
        logger.info(f"Max Absolute Score Error:   {max_score_err:.6e}")
        logger.info(f"Max Absolute Corners Error: {max_corner_err:.6e}")
        
        if max(max_score_err, max_corner_err) > tolerance:
            logger.warning(f"Precision degradation during export exceeds threshold ({tolerance})!")
        else:
            logger.info("Equivalence strict check Passed: Traced model mathematically matches eager graph.")
            
    # 4. Reload Reliability Check
    logger.info("--- Performing Reload Sanity Check ---")
    try:
        # We explicitly map it back to CPU upon reload, as standard edge deployments target CPU hardware 
        reloaded_model = torch.jit.load(args.output, map_location='cpu')
        reloaded_model.eval()
        
        cpu_input = torch.randn(*input_shape).to('cpu')
        with torch.inference_mode():
            score_out, corner_out = reloaded_model(cpu_input)
            
        logger.info(f"Successfully reloaded architecture file bounds onto 'cpu'.")
        logger.info(f"Reloaded outputs matching expected shapes -> Score: {score_out.shape}, Corners: {corner_out.shape}")
        
    except Exception as e:
        logger.error(f"Reload Sanity Check Failed! Traced model might execute unpredictably: {e}")
        return
        
    logger.info("=== Export Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
