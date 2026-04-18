# ID Card Corner Detection Pipeline

A production-grade, three-stage geometric machine learning pipeline for accurate high-speed CPU-first document extraction.

## Project Directory Structure

```text
corner_detection/
├── boundingbox/                     (Stage 1: BoundingBox Quad Detection)
│   ├── datasets/               (YOLO keypoint dataset parsing)
│   ├── models/                 (ResNet-18 + FPN + Dense Head)
│   ├── runs/                   (Training logs, checkpoints, visualizations)
│   ├── train.py                (Training entry point with Hard Example Mining)
│   ├── test.py                 (Evaluation with pixel-space diagnostics)
│   ├── export_torchscript.py   (Export Stage 1 to TorchScript)
│   └── run_torchscript_image.py (Standalone boundingbox inference)
├── orient/                     (Stage 1.5: Orientation Classification)
│   ├── datasets/               (Warped card dataset)
│   ├── models/                 (OrientNet: ~50k param classifier)
│   ├── train.py                (Orientation training loop)
│   └── test.py                 (Evaluation of 0/90/180/270 accuracy)
├── refiner/                    (Stage 2: Iterative Sub-Pixel Refinement)
│   ├── datasets/               (Patch extraction from warped card)
│   ├── models/                 (IterativeRefinerNet: Two-level differentiable zoom)
│   ├── train.py                (Refinement training pipeline)
│   └── test.py                 (Evaluation of sub-pixel precision)
├── common/                     (Shared core utility library)
│   ├── geometry.py             (Homography and corner sorting)
│   ├── metrics.py              (KeyPoint loss and accuracy metrics)
│   ├── transforms.py           (Geometric data augmentation)
│   └── visualization.py        (Diagnostic drawing)
└── run_torchscript_image.py    (Unified BoundingBox → Refiner inference)
```

---

## Stage 1: BoundingBox Quad Detection

Stage 1 uses a **KeyPoint style** architecture for robust, multi-scale corner localization.

### Architecture
- **Backbone**: ResNet-18 (ImageNet pre-trained).
- **Neck**: Top-down FPN producing features at strides 8, 16, and 32.
- **Head**: Anchor-free dense regression of bounding boxes, 4 keypoints, and confidence scores.

### Loss Formulation
Uses a composite KeyPointLoss:
1.  **Objectness Loss**: BCE loss for card presence at every grid cell.
2.  **Box Loss**: CIoU loss for the axis-aligned bounding box (positives only).
3.  **Keypoint Loss**: L1 loss for the 4 corner coordinates (positives only).
4.  **Keypoint Confidence**: BCE loss for corner visibility/accuracy (all cells).

**Positive Assignment**: Uses a `center_radius` filter combined with `top-k` nearest cells to the ground-truth center across all scales.

---

## Stage 1.5: Orientation Classification (OrientNet)

Ensures consistent physical corner identity (Top-Left, etc.) by classifying the card's rotation (0°, 90°, 180°, 270°).

- **Input**: 128x128 rectified card crop.
- **Architecture**: Lightweight MobileNet-style classifier (~50k parameters).
- **Benefit**: Resolves ambiguity in card orientation, allowing downstream tasks (like OCR) to work on upright images.

---

## Stage 2: Sub-Pixel Patch Refinement

Stage 2 achieves sub-pixel precision using an **Iterative Refinement** network with a differentiable zoom.

### Architecture
- **Input**: 96x96 image patches centered on Stage 1 predictions.
- **Global Stage**: Predicts a boundingbox corner location within the 96x96 patch.
- **Local Stage (Zoom)**: Uses `F.grid_sample` to differentiably crop a 32x32 sub-patch centered on the boundingbox prediction.
- **Fine Stage**: Predicts a high-precision offset within the 32x32 patch area using `SoftArgmax2D`.

### Loss Formulation
- **Wing Loss**: Used for robust keypoint regression on pixel-space coordinates.
- **Heatmap Focal Loss**: Used when training with Gaussian heatmap targets to prevent background dominance.

---

## Usage
- **Train BoundingBox**: `python boundingbox/train.py --batch_size 16 --epochs 100`
- **Train Orient**: `python orient/train.py --batch_size 64 --epochs 30`
- **Train Refine**: `python refiner/train.py --batch_size 64`
- **Run Full Pipeline**: `python run_torchscript_image.py --boundingbox_model path/to/boundingbox.pt --refiner_model path/to/refiner.pt --input image.jpg`
