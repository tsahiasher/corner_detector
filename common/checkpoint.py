import torch
import os
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], epoch: int, best_metric: float, path: str, **kwargs) -> None:
    """Saves a model checkpoint payload to disk, including optimizer and scheduler states.
    
    Args:
        model (torch.nn.Module): The actively trained model.
        optimizer (torch.optim.Optimizer): Model optimizer.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        epoch (int): Current epoch number.
        best_metric (float): Best recorded validation metric (e.g., mean pixel error).
        path (str): Output file path.
        **kwargs: Additional metadata to store in the checkpoint (e.g., best_score tuple).
    """
    state_dict: Dict[str, Any] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    # Merge additional kwargs
    state_dict.update(kwargs)
    
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
    try:
        torch.save(state_dict, path)
    except Exception as e:
        logger.error(f"Failed handling checkpoint dump: {e}")

def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], path: str, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
    """Loads a pre-computed model state from checkpoint.
    
    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Scheduler to load state into.
        path (str): Path to checkpoint file.
        device (Optional[torch.device]): Target device.
        
    Returns:
        Optional[Dict[str, Any]]: Raw checkpoint dictionary if successful, None otherwise.
    """
    map_loc = device if device is not None else 'cpu'
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=map_loc)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint
        except Exception as e:
            logger.error(f"Failed parsing binary tree payload at {path}: {e}")
            raise e
    logger.warning(f"Checkpoint trace root {path} missing, initialized default layer generation zero.")
    return None
