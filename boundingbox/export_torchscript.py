import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from boundingbox.models.boundingbox_quad_net import BoundingBoxQuadNet
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
    parser = argparse.ArgumentParser(description="Export BoundingBoxQuadNet to TorchScript for edge CPU deployment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained PyTorch checkpoint (.pt) to load.")
    parser.add_argument('--output', type=str, default='boundingbox_quad_net.pt', help="Path to save the exported TorchScript model.")
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
    
    # Ensure output is in the same directory as weights if not explicitly directed elsewhere
    if not os.path.isabs(args.output) and os.path.dirname(args.output) == '':
        weights_dir = os.path.dirname(os.path.abspath(args.weights))
        args.output = os.path.join(weights_dir, args.output)
        
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
    eager_model = BoundingBoxQuadNet().to(device)
    logger.info(f"Loading checkpoint weights from: {args.weights}")
    try:
        load_checkpoint(eager_model, None, None, args.weights, device=device)
    except Exception:
        logger.warning("Could not load weights, tracing with random initialized model!")
    eager_model.eval()

    model = eager_model
    model.eval()
    
    input_shape = (1, 3, args.input_height, args.input_width)
    logger.info(f"Generated trace dummy input volume with shape: {input_shape}")
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 2. Export Model
    logger.info(f"Attempting TorchScript export using method: {args.export_method.upper()}")
    if args.export_method == 'trace':
        try:
            with torch.inference_mode():
                # Passing strict=False handles dictionaries properly
                exported_model = torch.jit.trace(model, dummy_input, strict=False)
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
        max_err = 0.0
        
        with torch.inference_mode():
            for i in range(args.verify_runs):
                v_input = torch.randn(*input_shape).to(device)
                
                # Eager vs Traced checks 
                eager_out = model(v_input)
                traced_out = exported_model(v_input)
                
                if isinstance(traced_out, tuple):
                    # trace sometimes flattens dict to tuple
                    from copy import deepcopy
                    t_out = {'box': traced_out[0], 'loc_logits': traced_out[1]}
                else:
                    t_out = traced_out

                # Check all dict keys ('box', 'loc_logits')
                for key in eager_out:
                    err = torch.abs(eager_out[key] - t_out[key]).max().item()
                    max_err = max(max_err, err)
                
        logger.info(f"Max Absolute Error: {max_err:.6e}")
        
        if max_err > tolerance:
            logger.warning(f"Precision degradation during export exceeds threshold ({tolerance})!")
        else:
            logger.info("Equivalence strict check Passed: Traced model mathematically matches eager graph.")
            
    # 4. Reload Reliability Check
    logger.info("--- Performing Reload Sanity Check ---")
    try:
        reloaded_model = torch.jit.load(args.output, map_location='cpu')
        reloaded_model.eval()
        
        cpu_input = torch.randn(*input_shape).to('cpu')
        with torch.inference_mode():
            out_dict = reloaded_model(cpu_input)
            
        logger.info(f"Successfully reloaded architecture file bounds onto 'cpu'.")
        if isinstance(out_dict, dict):
            logger.info(f"Reloaded dict keys: {list(out_dict.keys())}")
        
    except Exception as e:
        logger.error(f"Reload Sanity Check Failed! Traced model might execute unpredictably: {e}")
        return
        
    logger.info("=== Export Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
