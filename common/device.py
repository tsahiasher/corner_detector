import argparse
import logging
import torch
from typing import Tuple

logger = logging.getLogger(__name__)

def add_device_args(parser: argparse.ArgumentParser, default: str = 'auto') -> None:
    """Consistently appends the standard device selection parameter to any CLI parser.
    
    Args:
        parser (argparse.ArgumentParser): The executing script's argument configuration.
        default (str, optional): Default fallback value ('auto', 'cpu', 'cuda'). Defaults to 'auto'.
    """
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default=default, 
                        help="Hardware execution target mapping limits.")

def resolve_device(device_arg: str) -> torch.device:
    """Parses standard literal device args limits formats offsets struct boundary pointer into native structural objects.
    
    Args:
        device_arg (str): The requested limit mapping.
        
    Returns:
        torch.device: Extracted mapped bounds parameters.
        
    Raises:
        RuntimeError: Raised strictly if a hardcoded 'cuda' value strings mapping vectors lists is requested but hardware unavailable.
    """
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but is not available on this machine!")
        return torch.device('cuda')
    elif device_arg == 'cpu':
        return torch.device('cpu')
    else:  # auto
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_device_info(resolved_device: torch.device, requested_device: str, logger: logging.Logger) -> None:
    """Pretty prints device telemetry output to the console.
    
    Args:
        resolved_device (torch.device): Final evaluated device chosen.
        requested_device (str): Original requested device string.
        logger (logging.Logger): Main output logger instance.
    """
    logger.info(f"Requested device: {requested_device}")
    logger.info(f"Resolved device: {resolved_device}")
    
    is_cuda = resolved_device.type == 'cuda'
    logger.info(f"CUDA available: {is_cuda}")
    if is_cuda:
        try:
            name = torch.cuda.get_device_name(resolved_device)
            logger.info(f"GPU: {name}")
        except Exception:
            pass

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Safely and recursively moves batch tensors to the specified device.
    
    Args:
        batch (dict): Dataloader output dictionary arrays.
        device (torch.device): Target execution limits map.
        
    Returns:
        dict: Device-mapped payload variables format arrays.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = move_batch_to_device(v, device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out

def sync_time() -> float:
    """Returns timing limit synchronization offset layout value.
    
    Returns:
        float: Environment absolute float representation limits limits values mapping parameter.
    """
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
