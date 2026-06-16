# Vietnamese Sign Language Recognition - VSL400 (Pose-based)

This repository is a modified version of the source code released alongside the
**VSL400 dataset** ([DOI: 10.5281/zenodo.17943574](https://doi.org/10.5281/zenodo.17943574)),
focused strictly on **Pose-based (skeleton keypoints)** Isolated Sign Language Recognition.
All direct video/RGB modeling modules have been stripped for a lightweight, keypoint-only pipeline.

---

## Project Layout

```text
LICENSE
Makefile
README.md
requirements.txt
src/                       # Main project code and entrypoints
  ├── train.py             # Model training engine
  ├── evaluate_model.py    # Model evaluation metrics engine
  ├── inference.py         # Batch inference engine
  ├── demo_web.py          # Webcam live demo interface (Gradio)
  ├── extract_keypoints.py # MediaPipe keypoint extractor from raw videos
  ├── configs/             # YAML configurations for models (SPOTER, SL-GCN)
  ├── data/                # Data pre-processing utilities (TBL, pruning)
  ├── features/            # Dataset definitions and preprocessing pipelines (Interpolate, Normalization)
  ├── models/              # Local model architectures (SPOTER, SL-GCN, DSTA-SLR)
  ├── tools/               # Training helper functions
  └── utils/               # Constants and configuration parameters
```

---

## Installation

1. Create a Python 3.9+ virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

*Note: For keypoint extraction, the environment installs `video_to_pose` which depends on MediaPipe.*

---

## Data Setup & Preprocessing

The model operates purely on `.pose` keypoint files. Below is how to set up the dataset.

### 1. Dataset Directory Structure
Place your VSL-400 dataset under `data/processed/vsl_400/` following this layout:
```text
data/
└── processed/
    └── vsl_400/
        ├── cam_1.json
        ├── gloss.csv
        └── cam_1/
            ├── video_0001.mp4   # (Optional) raw videos
            └── video_0001.pose  # Tensors containing keypoints (Required for training)
```

### 2. Extracting Pose Keypoints from Raw Videos
If you only have raw videos, you must first extract the MediaPipe keypoint coordinates to generate `.pose` files:
```bash
python3 src/extract_keypoints.py --video_dir data/processed/vsl_400/cam_1 --num_workers 4
```
This script runs the `video_to_pose` tool in parallel to output `.pose` files next to each video.

### 3. Exporting the Preprocessed Keypoint Dataset (Offline Export)
To generate and publish a **preprocessed keypoint dataset** (applying keypoint interpolation, facial landmarks subset filtering, and body/hand coordinate normalizations offline):
```bash
python3 src/preprocess_dataset.py --config_path src/configs/training/spoter.yaml --output_dir data/preprocessed_vsl_400
```
This script will:
1. Load the raw `.pose` files.
2. Run the specified normalization, interpolation, and facial subset transforms.
3. Save the clean, normalized keypoints as ready-to-use NumPy array files (`.npy`) inside `data/preprocessed_vsl_400/cam_1/` and output a corresponding `cam_1.json` metadata index file, which can be published directly to the community.

---

## Model Configuration

All training options are configured using YAML files located in `src/configs/`.
Key parameters in config files:
- `data.data_dir`: Path to the root of the processed dataset.
- `data.modality`: Must always be set to `"pose"`.
- `model.arch`: The model architecture (`"spoter"` or `"sl_gcn"`).
- `data.transform.interpolate`: Whether to perform Bilinear Interpolation on missing coordinates.
- `data.transform.anchor`: Coordinate shifting anchor (`"box"`, `"neck"`, or `"nose"`).

---

## Running Training

### Option A: Training with On-the-fly Preprocessing
To train a model from scratch using raw `.pose` files (applying the preprocessing pipeline dynamically on RAM during training):
1. In your configuration YAML (e.g., `src/configs/training/spoter.yaml`), point `data.data_dir` to the directory containing raw `.pose` files.
2. Run the training script:
   ```bash
   python3 src/train.py --config_path src/configs/training/spoter.yaml
   ```

### Option B: Training directly with Preprocessed Dataset (Fast Mode)
If you have already generated the preprocessed keypoint dataset (`.npy` files) using `preprocess_dataset.py` (or downloaded a preprocessed release):
1. In your configuration YAML, update `data.data_dir` to point to the preprocessed directory (e.g., `data/preprocessed_vsl_400`).
2. Run the training script exactly as before:
   ```bash
   python3 src/train.py --config_path src/configs/training/spoter.yaml
   ```
   *Note: The dataset loader automatically detects `.npy` files in the metadata index and loads them directly, bypassing the on-the-fly preprocessing transforms to speed up training dramatically.*

Checkpoint weights and training logs will be saved to the `experiments/` directory.

---

## Running Ablation Study

To systematically evaluate the impact of different preprocessing, augmentation, and model settings (the 19 greedy ablation trials):

### 1. Generate Configurations
First, generate the 19 configuration files for the ablation runs:
```bash
python3 generate_ablation_configs.py
```
This will output `run_00.yaml` to `run_18.yaml` inside `src/configs/ablation/`.

### 2. Run a Trial (Local)
To execute a specific trial locally (e.g., Run 07):
```bash
python3 run_ablation.py --run_id 7
```

**Dry-run check:** To quickly verify that the dataset and model load correctly without running the full training process:
```bash
python3 run_ablation.py --run_id 7 --dry_run
```

**Override epoch count:** For quick debugging, you can override the training epoch count (e.g., set to 1 epoch):
```bash
python3 run_ablation.py --run_id 7 --epochs 1
```

### 3. Generate Ablation Report
Once trials are completed (results saved as `.json` under `experiments/ablation_results/`), run the summary script to compile a structured markdown report comparing metrics across all runs:
```bash
python3 generate_ablation_report.py
```

---

## Running Evaluation & Inference

1. **Evaluate Checkpoint**:
   Run evaluation metrics (Accuracy, F1-score) on a test/validation split:
   ```bash
   python3 src/evaluate_model.py --config_path src/configs/evaluation/spoter.yaml
   ```

2. **Webcam Gradio Demo**:
   To launch a web browser interface that translates sign language gestures live from your webcam using a trained SPOTER checkpoint:
   ```bash
   python3 src/demo_web.py --config_path src/configs/inference/spoter_m4_cam1.yaml
   ```
   Open `http://127.0.0.1:7860` in your browser.

---

## Citation

If you use this code or the VSL400 dataset in your research, please cite the original work:

```bibtex
@dataset{nguyenquoc2026vsl400,
  author    = {Nguyen Quoc, Trung and
               Pham Dang, Khoi and
               Truong Duy, Viet and
               Truong Hoang, Vinh and
               Bilik, Simon and
               Sindelar, Matej and
               Stefansky, Jakub and
               {\L}ysiak, Adam and
               Martinek, Radek and
               Bilik, Petr},
  title     = {{A Multi-view Dataset for Vietnamese Word-Level
               Sign Language Recognition}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17943574},
  url       = {https://doi.org/10.5281/zenodo.17943574}
}
```

---

## License

This project is licensed under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license. You are free to share and adapt this work provided you give appropriate credit to the original authors.
