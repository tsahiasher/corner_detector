---
name: id-card-corner-pipeline
description: build and iterate a production-quality, edge-optimized machine learning pipeline for id-card corner extraction with four ordered keypoints and sub-pixel precision. use when designing, implementing, training, evaluating, exporting, or running inference for a cpu-first planar-card corner pipeline under a strict latency budget, with cuda-capable development scripts and torchscript deployment.
---

# Build the production pipeline

Implement a production-quality, CPU-first, boundingbox-to-refiner keypoint pipeline for a single rectangular ID card with exactly 4 ordered corners.


Always optimize for this constraint set:
- single class
- single instance per image
- cpu-only edge deployment target
- total inference time per image must be `<= 0.5 seconds`
- final output is 4 precise corner points in original image coordinates
- prioritize corner accuracy and deployment reliability over generic detector flexibility

Do not default to heavyweight generic detectors unless the user explicitly asks for alternatives or comparisons.

## Core architecture

Use this 4-stage design as the default:

1. **BoundingBox detector (Spatially-Aware Bounding Box Regressor)**
   - Run a lightweight CPU-friendly backbone on a fixed padded input.
   - **Crucial Pattern**: Do NOT use 4 independent dense heatmaps for the corners.
   - **Crucial Pattern**: Do NOT use massive `AdaptiveAvgPool` global pooling dropping straight into independent coordinate MLPs since it wipes away structural relationships.
   - **Architecture**: Use a progressive spatial reduction regressor explicitly outputting a unified bounding box:
     - Output exactly ONE object prediction configuration natively.
     - Use a ResNet-18 Backbone scaled down to its final layer space.
     - Attach a specialized Conv Neck preserving a moderate spatial grid (e.g., `4x4`) routing into a dense layout to conserve bounding geometric boundaries naturally before squashing coordinates out.
     - Eliminate score objectness networks entirely for environments assuming purely 1 card per image stream.
   - Downstream processes manually expand this generalized bounding block configuration.

2. **Stage 2.5 – Orientation classification (OrientNet, optional)**
   - Run ONLY after the boundingbox model has localized the card.
   - Warp the card to a 128×128 canonical square using the boundingbox homography.
   - Feed the crop into a tiny (~50k param) MobileNet-style classifier predicting rotation: 0°, 90°, 180°, or 270°.
   - Cyclically shift the boundingbox corners so `corners[0]` is the physical TL.
   - **When to add this stage**: Only when physical corner identity (not just geometric consistency) is required downstream (e.g., crop face vs. barcode side). If only precise localization is needed, skip it.
   - Training: Use GT homography to warp training images to canonical form. Apply colour jitter. Train with `CrossEntropyLoss(label_smoothing=0.05)` for 25–30 epochs with `AdamW + CosineAnnealingLR`.
   - Export: TorchScript trace at 128×128 input.

3. **Rectification**
   - Compute a homography from the boundingbox quadrilateral to a canonical upright card plane.
   - Warp the image into a fixed canonical resolution.

3. **Corner refinement**
   - Extract 4 fixed-location corner patches from the rectified card.
   - Use one small shared refinement network on all 4 patches.
   - Predict a heatmap and offset field for each patch.
   - Decode sub-pixel corner coordinates from heatmap plus offset.

4. **Back-projection**
   - Map refined canonical-space corners back to the original image using the inverse homography.

## Dataset assumptions

Assume this dataset layout unless the user explicitly says otherwise:
- Training images: `../crop-dataset-eitan-yolo/images/train`
- Validation images: `../crop-dataset-eitan-yolo/images/val`
- Validation annotations: `../crop-dataset-eitan-yolo/annotations/val.json`

Dataset rules:
- Training annotations come from YOLO txt label files alongside the training images. Do not assume `train.json` exists.
- Validation may use the COCO-format `val.json`.
- Each image contains exactly one card.
- All 4 keypoints are always present.
- Keypoint order is fixed in physical space and not image space and must remain consistent everywhere:
  - `top-left`
  - `top-right`
  - `bottom-right`
  - `bottom-left`
- **Crucial Pattern**: NEVER trust native dataset keypoint annotation order. Calculate the planar geometric centroid and sort all 4 parsed bounding targets uniformly by `math.atan2` explicitly before sending them to optimization layers or metrics.
- **Fail Loudly**: NEVER silently yield dummy default coordinates (e.g., `[0,0], [1,0]`) if label files are entirely missing/malformed. Throw loud mapping errors immediately to prevent `0.00` train loss network collapse.
- Images may be portrait, landscape, tightly cropped, padded, rotated, or skewed.
- Model input size is fixed.

## Output contract

Keep module interfaces simple and stable.

Preferred modules:
- `BoundingBoxQuadNet`
- `OrientNet`
- `PatchRefiner`
- `geometry`
- `transforms`
- `metrics`
- `checkpoint`
- `device`
- `visualization`

Default model output:
- `score`
- `corners_norm`

Where:
- `corners_norm` has shape `[B, 4, 2]`
- corner order is always `top-left, top-right, bottom-right, bottom-left`
- normalized coordinates are in `[0, 1]` relative to the model input

## Device handling

Require explicit, consistent device handling across all scripts.

All core scripts must support:
- `--device auto`
- `--device cpu`
- `--device cuda`

Default behavior:
- `train.py`: default to `auto`
- `test.py`: default to `auto`
- `export_torchscript.py`: default to `cpu`
- `run_torchscript_image.py`: default to `cpu`

Rules:
- `auto` means CUDA if available, otherwise CPU
- if `cuda` is requested and unavailable, fail clearly
- always move model, inputs, targets, and checkpoints to the resolved device explicitly
- for TorchScript loading, use device-aware loading such as `map_location=device`
- move tensors back to CPU before converting to NumPy, PIL, or saving visualizations

## Script quality requirements

All core scripts must be production-ready, not prototype-level.

### General requirements for all scripts
- use clear `argparse` interfaces with readable help text
- use structured logging with both:
  - console output
  - file logging
- **Unified Run Tracking**: Route all execution artifacts (checkpoints, metrics CSVs, scripts logs, configs, and visual inferences) into isolated grouped directories (e.g., `runs/20261102_120000_name/`) instead of flooding standalone loose mapping folders safely.
- log selected device and relevant runtime information
- keep code readable, explicit, and easy to debug
- avoid hidden assumptions about tensor shapes, coordinate transforms, or device placement
- use ml conda environment
- put scripts that are not part of the core pipeline in the `debugging_scripts` directory

### Training script (`train.py`)
Require:
- training and validation support
- checkpoint saving for:
  - best model
  - last model
- best model selection based on **mean pixel error**, not only loss
- learning-rate logging each epoch
- epoch timing and total training time
- clean resume support with explicit logging of:
  - checkpoint path
  - resumed epoch
  - previous best metric

Training logs should include at least:
- epoch
- learning rate
- train loss
- validation loss
- mean pixel error
- median pixel error
- threshold metrics such as `<2 px`, `<3 px`, `<5 px`, `<10 px`
- epoch duration

### Evaluation script (`test.py`)
Require:
- aggregate evaluation metrics
- per-corner error reporting:
  - top-left
  - top-right
  - bottom-right
  - bottom-left
- threshold metrics:
  - `<2 px`
  - `<3 px`
  - `<5 px`
  - `<10 px`
- optional per-image result export
- optional visualization saving
- failure analysis support, including reporting the worst samples by error
- inference timing and average time per image

### TorchScript export (`export_torchscript.py`)
Require:
- explicit device selection
- clean export path handling
- support for `trace` by default unless there is a concrete reason to prefer `script`
- verification after export by comparing eager and TorchScript outputs
- reporting of max absolute error for exported outputs
- reloading the saved `.pt` file and running a sanity-check inference
- explicit confirmation that the exported model can be loaded and run on CPU

### Single-image TorchScript inference (`run_torchscript_image.py`)
Require:
- deterministic inference preprocessing
- correct decoding of normalized corners back to original image coordinates
- visualization drawn in original-image coordinates
- optional JSON result output
- timing breakdown for:
  - image load
  - preprocessing
  - inference
  - postprocessing
  - total time
- clear printing of:
  - score
  - normalized corners
  - decoded pixel coordinates

## Metrics priority

The most important metric is **pixel-level corner accuracy**.

Always prefer reporting:
- mean pixel error
- median pixel error
- per-corner error
- threshold metrics such as `<2 px`, `<3 px`, `<5 px`, `<10 px`
- latency where relevant

Do not treat loss as the primary success metric once evaluation metrics are available.

## Model guidance for stage 1

For the boundingbox detector:
- keep the backbone lightweight and CPU-friendly
- preserve spatial information in the head (do NOT drop straight to 1x1 Global Average Pooling dynamically mapping outputs without intermediary preservation bounds)
- output exactly 1 strict bounding context natively per image assuming ID cards stream inherently.
- explicitly optimize around a generalized target bounding box structure relying on `SmoothL1` alongside `GIoU` boundary constraints natively.

## Training guidance

For stage 1:
- optimize for robust boundingbox localization, not final sub-pixel output
- use augmentations that match the real task:
  - rotation
  - scaling
  - translation
  - perspective or skew
  - crop and padding simulation
- only use flips if corner ordering is updated correctly afterward


For stage 2:
- use local heatmaps plus offsets for precise refinement on rectified corner patches

## Code generation rules

When writing code:
- use PyTorch
- provide complete runnable module definitions when requested
- keep tensor flow explicit and TorchScript-friendly
- use type hints where practical
- use clear docstrings for public classes and functions
- do not silently swallow exceptions
- log important error conditions clearly
- keep implementation compact, readable, and deployment-oriented
- put scripts that are not part of the core pipeline in the `debugging_scripts` directory

## Evaluation and deployment expectations

Evaluation and deployment code should be practical, not smoke-test quality.

Always prefer:
- correct coordinate handling
- clear logging
- measurable latency
- verifiable export behavior
- debuggable outputs
- reproducible script behavior

## Maintenance

As work on the project progresses, keep this `SKILL.md` up to date with anything learned that would improve future implementation, debugging, training, evaluation, export, or deployment for this project.

If there is a tradeoff between prettier abstractions and clearer deployment-safe code, prefer the clearer deployment-safe code.

> Note: All code development for this repository assumes a Windows primary environment explicitly. Training loop components and dataset parsing remains universally compliant, while worker thread spawning and path mappings expect Windows OS structural fallbacks implicitly.