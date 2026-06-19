# VSL-400 Ablation Study and Research Report

This report aggregates and analyzes the results of the 19 runs conducted for the Vietnamese Sign Language isolated word recognition ablation study.

## Phase 1: TBL Preprocessing Sweep (Góc Ngưỡng & Trễ Đệm)
Optimizing the temporal boundary localization parameters for segmenting gesture boundaries.

| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 01 | TBL Preprocessing (θ = 140°, τb = 400 ms) | *Err* | *Err* | *Err* | *Err* | Failed |
| Run 02 | TBL Preprocessing (θ = 150°, τb = 400 ms) | *N/A* | *N/A* | *N/A* | *N/A* | Missing |
| Run 03 | TBL Preprocessing (θ = 160°, τb = 400 ms - Default Base) | 89.61% | 90.01% | 81.08% | 80.86% | Completed |
| Run 04 | TBL Preprocessing (θ = 170°, τb = 400 ms) | *N/A* | *N/A* | *N/A* | *N/A* | Missing |
| Run 05 | TBL Preprocessing (θ = 160°, τb = 200 ms) | *N/A* | *N/A* | *N/A* | *N/A* | Missing |
| Run 06 | TBL Preprocessing (θ = 160°, τb = 600 ms) | *N/A* | *N/A* | *N/A* | *N/A* | Missing |

## Phase 2: Keypoint Interpolation & Anchor Normalization
Comparing linear joint interpolation and centering strategies (Neck vs. Nose vs. Box).

| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 03 | TBL Preprocessing (θ = 160°, τb = 400 ms - Default Base) | 89.61% | 90.01% | 81.08% | 80.86% | Completed |
| Run 07 | Keypoint Interpolation (using best TBL θ = 160°, τb = 400 ms) | 90.21% | 90.46% | 80.80% | 80.72% | Completed |
| Run 08 | Neck Anchor Normalization | 90.37% | 90.61% | 84.08% | 84.02% | Completed |
| Run 09 | Nose Anchor Normalization | 89.89% | 90.23% | 80.71% | 80.57% | Completed |

## Phase 3: Augmentation Ablation Study
Evaluating rotation, squeezing, perspective transforms, joint kinematics, and noise additions.

| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 07 | Keypoint Interpolation (using best TBL θ = 160°, τb = 400 ms) | 90.21% | 90.46% | 80.80% | 80.72% | Completed |
| Run 10 | Spatial Augmentations only (Rotate / Squeeze) | 89.73% | 89.85% | 81.61% | 81.58% | Completed |
| Run 11 | Perspective Skew Augmentation only | 89.89% | 90.12% | 81.27% | 81.10% | Completed |
| Run 12 | Kinematic Augmentation only (ArmJointRotate) | 88.80% | 89.13% | 80.55% | 80.44% | Completed |
| Run 13 | Gaussian Noise Augmentation only | 88.32% | 88.69% | 79.71% | 79.39% | Completed |
| Run 14 | Combined Augmentations (Spatial + Perspective + Kinematic + Noise) | 90.25% | 90.36% | 82.30% | 82.30% | Completed |
| Run 15 | Facial Landmarks Integration (Eyebrows, Eyes, Mouth + Combined Augs) | 89.49% | 89.80% | 80.96% | 80.94% | Completed |

## Phase 4: Cross-Model Validation (SL-GCN)
Transferring the best preprocessing, interpolation, and face selections to the local SL-GCN model.

| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 16 | SL-GCN Baseline | *N/A* | *N/A* | *N/A* | *N/A* | Missing |
| Run 17 | SL-GCN Optimized (Interpolation + Best TBL) | *N/A* | *N/A* | *N/A* | *N/A* | Missing |
| Run 18 | SL-GCN Optimized + Face Landmarks | *N/A* | *N/A* | *N/A* | *N/A* | Missing |

## Reference: Raw Baseline

| Run ID | Configuration Description | Val Acc | Val F1 | Test Acc | Test F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 00 | Raw Baseline (No TBL, no interpolation, no augmentations) | 88.76% | 89.21% | 78.59% | 78.47% | Completed |