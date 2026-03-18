import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Sets standard deterministic pseudo-random stream seeds for computation reproducibility.
    
    Args:
        seed (int, optional): The canonical numeric seed value offset index. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Binds CuDNN graph compiler backends for constant time trace matching constraints
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global trace mapping pipeline seeds aligned to instance index: {seed}")
