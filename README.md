# Vietnamese Sign Language Recognition - VSL400

This repository is a modified version of the source code released alongside the
**VSL400 dataset** ([DOI: 10.5281/zenodo.17943574](https://doi.org/10.5281/zenodo.17943574)),
which is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [Citation](#citation) and [License](#license) sections below.

## Project layout

Top-level layout (trimmed to most relevant folders):

```text
LICENSE
Makefile
README.md
requirements.txt
src/                       # main project code and entrypoints
  ├── train.py             # training entrypoint
  ├── inference.py         # inference entrypoint
  ├── configs/             # YAML config templates for training / inference
  ├── data/                # datasets and processing code
  ├── models/              # trained model binaries and checkpoints
  ├── utils/               # utility modules used across the project
  ├── visualization/       # plotting and visualization helpers
  ├── convert_model_to_onnx.py
  ├── convert_model_to_torchscript.py
  ├── evaluate_model.py
  ├── extract_keypoints.py
```

Note: See `src/` for the actual script names and `src/configs/` for example YAMLs.

## Datasets

Typical layout for VSL-400 in this repo's data directory:

```text
vsl_400/
  cam_1/    # videos from camera 1
  cam_2/
  cam_3/
  cam_1.json
  cam_2.json
  cam_3.json
  gloss.csv
```

## Installation

1. Create a Python 3.9 environment (this project was developed against Python 3.9.x).

2. (Optional) Install PyTorchVideo if your chosen configs depend on it:

```powershell
cd src/libs
git clone https://github.com/facebookresearch/pytorchvideo.git
pip install -e pytorchvideo
cd ../../
```

3. Install Python requirements:

```powershell
pip install -r requirements.txt
```

If you use `wandb` or private Hugging Face models/datasets, log into those services before running training/inference.

## Data Processing

Step 1: Detect gesture boundaries (TBL)

```powershell
python src/data/temporal_boundary_localization.py --input_video video.mp4 --get_cut_time
```

Step 2: Segment and crop videos (BGSP)

```powershell
python src/data/boundary_segmentation_pruning.py --input_video video.mp4 --cut_crop_video
```

Output: Individual segmented videos saved in `./video/` directory.

## Configuration

Configs live in `src/configs/` separated by training/inference subfolders. Typical fields to update:

- `data`: dataset, modality, subset, data_dir
- `training`: run_name, hub_model_id

## Training

From the project root you can start training with a config file:

```powershell
python src/train.py --config_path src/configs/training/config.yaml
```

This will read the YAML under `src/configs/` and run the training pipeline. Common issues to check:

- Ensure `data.data_dir` points to your local copy of the dataset.
- Make sure required pretrained weights are accessible (local path or HF hub).
- If using `wandb`, set `report_to` in the config and log in with `wandb login`.

## Inference

Run inference (evaluation or producing predictions) with:

```powershell
python src/inference.py --config_path src/configs/inference/config.yaml
```

For the local SPOTER webcam demo trained in `experiments/spoter_m4_cam1`, run:

```bash
.venv/bin/python src/demo_web.py --config_path src/configs/inference/spoter_m4_cam1.yaml
```

Then open `http://127.0.0.1:7860`, allow camera access, and press **Start camera**. Results are saved to `demo/spoter_m4_cam1_webcam/demo_web_results.csv` when you press **Save CSV** or stop the demo.

There are also helpers for model conversion and evaluation:

- `src/evaluate_model.py` — run evaluation metrics on predictions.
- `src/extract_keypoints.py` — utilities to extract pose/keypoint features from videos.

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

> Nguyen Quoc, T., Pham Dang, K., Truong Duy, V., Truong Hoang, V., Bilik, S., Sindelar, M.,
> Stefansky, J., Łysiak, A., Martinek, R., & Bilik, P. (2026).
> *A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition* [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.17943574

---

## License

This repository is based on material originally published under the
**[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license.

You are free to share and adapt this work for any purpose, including commercially,
provided you give appropriate credit to the original authors (see [Citation](#citation) above)
and indicate what changes were made.

See the [LICENSE](LICENSE) file for full details.

---

## Acknowledgements

This project builds upon the following open-source works:

- **VSL400 Dataset & Original Source Code**
  Nguyen Quoc, T. et al. (2026). VSB – Technical University of Ostrava.
  https://doi.org/10.5281/zenodo.17943574 | License: CC BY 4.0

- **SPOTER** — Sign Pose-based Transformer for Word-level Sign Language Recognition
  Boháček, M. & Hrúz, M. (2022). CVPR Workshops.
  https://github.com/matyasbohacek/spoter | License: Apache 2.0

- **HuggingFace Transformers**
  https://github.com/huggingface/transformers | License: Apache 2.0

- **VideoMAE** — Masked Autoencoders are Data-Efficient Learners for Video Understanding
  Tong, Z. et al. (2022). NeurIPS.
  HuggingFace: `MCG-NJU/videomae-small-finetuned-kinetics`
