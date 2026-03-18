"""
Stage 2: Refiner Objective Functions (Placeholder)

TODO: Implement losses for corner patch refinement.
This will likely include:
- Soft-argmax MSE / Heatmap focal loss
- Offset field regression loss
"""

import torch
import torch.nn as nn

class RefinerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        raise NotImplementedError("Stage 2 losses are pending implementation.")
