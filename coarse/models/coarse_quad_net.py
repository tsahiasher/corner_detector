"""YOLO-Pose inspired anchor-free keypoint detector for ID card corners.

Rewrites the CenterNet-style CoarseQuadNet with a proper YOLO-Pose formulation:
grid-based objectness, CIoU bounding-box regression, and absolute keypoint
prediction — all without any external YOLO library dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import torchvision.models as models
from common.geometry import sort_corners_clockwise


class CoarseQuadNet(nn.Module):
    """YOLO-Pose anchor-free single-object keypoint detector.

    Jointly predicts a bounding box and 4 corner keypoints for a single
    ID card per image, following the YOLO-Pose (Maji et al. 2022)
    formulation implemented from scratch.

    Architecture:
        Backbone : ResNet-18 (ImageNet pre-trained)
        Neck     : Top-down FPN merging stride-{32, 16, 8} → stride-8
        Head     : 2×(Conv3×3-BN-SiLU) + Conv1×1 → 13 channels per cell

    Per-cell output channels (13 total)::

        [0]       objectness logit
        [1:3]     tx, ty   – centre offsets  (sigmoid → grid-relative)
        [3:5]     tw, th   – bbox size       (sigmoid → normalised [0, 1])
        [5:13]    kx₁,ky₁ … kx₄,ky₄ – keypoints (sigmoid → absolute [0, 1])

    Returns:
        Dict with keys:
            score          [B, 1]            objectness confidence at best cell
            corners        [B, 4, 2]         clockwise-sorted corners (normalised)
            corners_visual [B, 4, 2]         raw predicted corners (channel order)
            raw_pred       [B, 13, Hg, Wg]   full grid tensor (for loss)
    """

    NUM_KPT: int = 4
    NUM_OUT: int = 1 + 2 + 2 + NUM_KPT * 2  # 13

    def __init__(self) -> None:
        """Initialises backbone, FPN neck and detection head."""
        super().__init__()

        # ---- Backbone: ResNet-18 ----
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1   # stride  4,  64 ch
        self.layer2 = resnet.layer2   # stride  8, 128 ch
        self.layer3 = resnet.layer3   # stride 16, 256 ch
        self.layer4 = resnet.layer4   # stride 32, 512 ch

        # ---- FPN Neck → stride 8 ----
        _nc = 128  # neck channel width
        self.lat4 = nn.Conv2d(512, _nc, 1, bias=False)
        self.lat3 = nn.Conv2d(256, _nc, 1, bias=False)
        self.lat2 = nn.Conv2d(128, _nc, 1, bias=False)

        self.smooth = nn.Sequential(
            nn.Conv2d(_nc, _nc, 3, padding=1, bias=False),
            nn.BatchNorm2d(_nc),
            nn.SiLU(inplace=True),
        )

        # ---- Detection Head ----
        self.detect = nn.Sequential(
            nn.Conv2d(_nc, _nc, 3, padding=1, bias=False),
            nn.BatchNorm2d(_nc),
            nn.SiLU(inplace=True),
            nn.Conv2d(_nc, _nc, 3, padding=1, bias=False),
            nn.BatchNorm2d(_nc),
            nn.SiLU(inplace=True),
            nn.Conv2d(_nc, self.NUM_OUT, 1),  # final predictor
        )

        # Bias init: prior objectness ≈ 1 %  →  sigmoid(−4.6) ≈ 0.01
        nn.init.constant_(self.detect[-1].bias[:1], -4.6)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run backbone → FPN → head and decode the best-cell prediction.

        Args:
            x: Input image tensor ``[B, 3, H, W]``.

        Returns:
            Dict with ``score``, ``corners``, ``corners_visual``, ``raw_pred``.
        """
        B = x.size(0)

        # 1. Backbone
        c1 = self.stem(x)
        c2 = self.layer1(c1)    # stride 4
        c3 = self.layer2(c2)    # stride 8
        c4 = self.layer3(c3)    # stride 16
        c5 = self.layer4(c4)    # stride 32

        # 2. FPN (top-down → stride 8)
        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode='bilinear', align_corners=False,
        )
        p3 = self.lat2(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode='bilinear', align_corners=False,
        )
        feat = self.smooth(p3)

        # 3. Head
        raw = self.detect(feat)  # [B, 13, Hg, Wg]
        Hg, Wg = raw.shape[2:]

        # 4. Select cell with highest objectness
        obj = torch.sigmoid(raw[:, 0])           # [B, Hg, Wg]
        flat = obj.reshape(B, -1)
        best_val, best_idx = flat.max(dim=-1)    # [B]

        gi = best_idx // Wg   # grid row  (y)
        gj = best_idx % Wg    # grid col  (x)
        bi = torch.arange(B, device=x.device)

        # 5. Decode at best cell
        r = raw[bi, :, gi, gj]                   # [B, 13]

        # Keypoints (absolute normalised via sigmoid)
        kpt = torch.sigmoid(r[:, 5:13]).view(B, 4, 2)
        kpt = torch.clamp(kpt, 0.0, 1.0)

        corners_sorted = sort_corners_clockwise(kpt)

        return {
            'score':          best_val.unsqueeze(1),  # [B, 1]
            'corners':        corners_sorted,          # [B, 4, 2]
            'corners_visual': kpt,                     # [B, 4, 2]
            'raw_pred':       raw,                     # [B, 13, Hg, Wg]
        }
