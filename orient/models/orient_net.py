import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import torchvision.models as models
class OrientNet(nn.Module):
    """Lightweight card orientation classifier.

    Input:  [B, 3, 128, 128]  — warped card crop (output of Coarse homography).
    Output: [B, 4]            — logits for rotation classes 0°, 90°, 180°, 270°.

    Architecture:
        ResNet18 backbone (layers 1-4) followed by AdaptiveAvgPool and 
        a Linear classifier.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(512, num_classes)

        # Initialise classifier to small weights for stable early training
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalised card crop (ImageNet mean/std).

        Returns:
            logits: [B, num_classes]  (raw, before softmax)
        """
        feats = self.backbone(x)          # [B, 128, 4, 4]
        feats = self.pool(feats)          # [B, 128, 1, 1]
        feats = feats.flatten(1)          # [B, 128]
        feats = self.dropout(feats)
        return self.classifier(feats)     # [B, 4]

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted rotation class (0–3) without gradient."""
        with torch.no_grad():
            logits = self.forward(x)
        return logits.argmax(dim=-1)   # [B]
