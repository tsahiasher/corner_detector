# ID Card Corner Detection Pipeline

A production-grade, three-stage geometric machine learning pipeline for accurate high-speed CPU-first document extraction.

## Project Directory Structure

```text
corner_detection/
├── coarse/                     (Stage 1: Coarse Quad Detection)
│   ├── datasets/               (YOLO/COCO parsing with Mask/Edge generation)
│   ├── models/                 (CenterNet backbone + FPN + dilated refinement)
│   ├── runs/                   (Training logs, checkpoints, visualizations)
│   ├── train.py                (Training entry point)
│   ├── test.py                 (Evaluation with diagnostics)
│   └── export_torchscript.py   (Export Stage 1 to TorchScript)
├── orient/                     (Stage 2.5: Orientation Classification)
│   ├── datasets/               (Warp-then-classify dataset)
│   ├── models/                 (OrientNet: ~50k param classifier)
│   ├── runs/                   (Training logs, checkpoints)
│   ├── train.py                (Orientation classifier training)
│   ├── test.py                 (Evaluation: overall + per-class accuracy)
│   ├── export_torchscript.py   (Export Stage 2.5 to TorchScript)
│   └── run_torchscript_image.py (Standalone coarse+orient inference)
├── refiner/                    (Stage 2: Sub-Pixel Patch Refinement)
│   ├── datasets/               (Homography-rectified patch extraction)
│   ├── models/                 (Iterative two-level refinement CNN)
│   ├── runs/                   (Training logs, checkpoints)
│   ├── train.py                (Refinement training pipeline)
│   └── test.py                 (Evaluation of sub-pixel precision)
├── common/                     (Shared core utility library)
│   ├── checkpoint.py           (Weights management)
│   ├── device.py               (Device-aware orchestration)
│   ├── geometry.py             (Homography and projection logic)
│   ├── metrics.py              (Robust metric calculation)
│   ├── transforms.py           (Geometric data augmentation)
│   └── visualization.py        (Diagnostic drawing and saving)
├── run_torchscript_image.py    (Unified Coarse → Orient → Refiner inference)
└── debugging_scripts/          (Standalone sanity-check scripts)
```

---

## Stage 1: Coarse Quad Detection


### Loss Formulation
Stage 1 uses a **Composite Geometric Loss** to enforce global consistency:
1.  **Homography Reprojection Loss**: $\mathcal{L}_{reproj} = \text{SmoothL1}(\text{project}(H_{pred}, G), \text{project}(H_{gt}, G))$ where $G$ is a canonical grid. This supervises the **entire surface** of the card.
2.  **Corner Huber Loss**: $\mathcal{L}_{corner} = \text{Huber}(P_{pred}, P_{gt})$ for robust point regression.
3.  **Dense Loss**: $\mathcal{L}_{dense} = 0.5 \cdot \text{Dice}(M, \hat{M}) + 0.5 \cdot \text{BCE}(M, \hat{M})$ for Mask and Edges.
4.  **Consistency Loss**: $\text{SmoothL1}(P_{spatial}, P_{gh})$ to synchronize the heatmap and homography heads.

---

## Stage 2: Sub-Pixel Patch Refinement

Stage 2 achieves sub-pixel precision by focusing on local context.

### Architecture
- **Input**: Four 96x96 image patches centered on Stage 1 predictions, rectified to a canonical view using the Stage 1 homography.
- **Shared CNN**: A lightweight backbone (3x3 Convs + BN + ReLU) that processes all 4 patches.
- **Output**: Relative $(dx, dy)$ regression to refine the local corner position.

### Loss Formulation
Stage 2 uses a direct **Supervised Regression**:
- **Loss**: $\mathcal{L}_{refine} = \text{MSE}(\Delta P_{pred}, \Delta P_{gt})$ where $\Delta P$ is the relative displacement in patch-space.

---

## Usage
- **Train Coarse**: `python coarse/train.py`
- **Train Refine**: `python refiner/train.py --batch_size 64`
