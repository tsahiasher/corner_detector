# ID Card Corner Detection Pipeline

This framework provides a production-grade, two-stage geometric machine learning pipeline for accurate high-speed CPU-first document extraction. 

## Project Directory Structure

```text
corner_detection/
├── coarse/                     (Stage 1: Coarse Quad Corners Detection)
│   ├── datasets/               (YOLO/COCO Dataset parsing logic)
│   ├── models/                 (MobileNetV3 backbone and quad extraction head)
│   ├── runs/                 (Automated stage 1 logs, configs, evaluations)
│   ├── train.py                (Coarse training entry point)
│   ├── test.py                 (Coarse evaluation script)
│   ├── export_torchscript.py   (Deploy execution tracing graphs)
│   └── run_torchscript_image.py(Inference visual bounds execution)
├── refiner/                    (Stage 2: High-Precision Sub-Pixel Local Refinement)
│   ├── dataset.py              (Homography patch extraction mock)
│   ├── losses.py               (Heatmap mapping format constraints mock)
│   ├── model.py                (Patch-shared refinement logic mock)
│   ├── test.py                 (Local evaluation tracking mock)
│   └── train.py                (Refinement training pipeline mock)
├── common/                     (Shared core utility library)
│   ├── runs/                   (Automated stage 2 outputs boundaries)
│   ├── checkpoint.py           (Weights management)
│   ├── device.py               (CUDA/CPU mapping execution tracking)
│   ├── geometry.py             (Homography calculations and logic)
│   ├── metrics.py              (Centroid sorting formatting validation)
│   └── ...                     
├── pipeline/                   (Full Orchestration Stage 1 -> Stage 2 logic)
├── configs/                    (Hyperparameter definitions mapping bounds)
└── debugging_scripts/          (Standalone debugging sanity-check format logic)
```

## Running the Pipeline (Stage 1)
- **Train**: `python coarse/train.py`
- **Evaluate**: `python coarse/test.py --weights <path>`
- **Export Model**: `python coarse/export_torchscript.py --weights <path>`
- **Run Inference**: `python coarse/run_torchscript_image.py --model <pt> --image <path>`
