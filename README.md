# VSL Keypoint Pipeline

> **Pipeline trích xuất và chuẩn hóa keypoint Ngôn ngữ Ký hiệu Tiếng Việt (VSL)**
> Phục vụ nghiên cứu nhận dạng ký hiệu tiếng Việt (Isolated Word-Level VSL Recognition)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Bắt đầu từ đây

Nếu bạn mới tiếp cận dự án, đây là thứ tự đọc được đề xuất:

| Bước | Tài liệu | Nội dung |
| :---: | :--- | :--- |
| 1 | [📖 Đặc trưng Ngôn ngữ Ký hiệu Tiếng Việt](docs/vsl_linguistics.md) | Kiến thức nền về VSL trước khi đọc kỹ thuật |
| 2 | [🔬 Khảo sát Kỹ thuật](docs/vsl_technical_survey.md) | Tổng quan các phương pháp, dataset và mô hình liên quan |
| 3 | [🏗️ Kiến trúc Pipeline](docs/architecture_analysis.md) | Pipeline xử lý end-to-end và cách các module kết nối |
| 4 | [📊 Kết quả Ablation Study](docs/ablation_study_report.md) | Thực nghiệm xác định cấu hình tiền xử lý tối ưu |
| 5 | [⚙️ Cài đặt & Chạy thử](#cài-đặt) | Hướng dẫn cài đặt và chạy pipeline |
| 6 | [🗃️ Dataset Keypoint](#dataset) | Tải về bộ dữ liệu keypoint đã xử lý sẵn |

---

## Giới thiệu

Repo này cung cấp:

1. **Pipeline keypoint end-to-end** từ video thô → huấn luyện mô hình nhận dạng VSL, hỗ trợ hai kiến trúc:
   - **SPOTER** (Pose-based Transformer)
   - **SL-GCN** (Spatial-Temporal Graph Convolutional Network)

2. **Bộ keypoint đã tiền xử lý** (định dạng `.npy`) sẵn sàng dùng để huấn luyện, không cần chạy lại pipeline.

3. **Kết quả ablation** 14 cấu hình tiền xử lý, xác định pipeline tối ưu trên VSL400.

Pipeline được phát triển dựa trên mã nguồn gốc từ **VSL400 Dataset** (Nguyen Quoc et al., 2026), mở rộng thêm:
- Keypoint interpolation cho điểm thiếu (theo Roh et al., 2024)
- Neck anchor normalization thay thế Box normalization
- Ablation study đầy đủ để chứng minh hiệu quả preprocessing

---

## Kết quả Chính

### Mô hình SPOTER (Transformer-based)

| Cấu hình | Test Acc | Test F1 | Ghi chú |
| :--- | ---: | ---: | :--- |
| Raw Baseline | 78.59% | 78.47% | Không tiền xử lý |
| + TBL Segmentation | 81.08% | 80.86% | Cắt video theo biên ký hiệu |
| **+ Neck Anchor Norm** | **84.08%** | **84.02%** | **Cấu hình tối ưu nhất (SPOTER)** |
| + Combined Augmentation | 82.30% | 82.30% | Tăng cường dữ liệu cộng dồn |

### Mô hình SL-GCN (Graph-based)

| Cấu hình | Test Acc | Test F1 | Ghi chú |
| :--- | ---: | ---: | :--- |
| SL-GCN Baseline | 67.44% | 66.88% | Không tiền xử lý |
| + Interpolation + Best TBL | 73.26% | 73.29% | Thêm nội suy khớp và TBL |
| **+ Preprocessing + Face** | **75.41%** | **75.23%** | **Cấu hình tối ưu nhất (SL-GCN)** |

*Chi tiết đầy đủ 14/19 runs: xem [docs/ablation_study_report.md](docs/ablation_study_report.md)*

---

## Dataset

Bộ dữ liệu keypoint được lưu trữ riêng (không nằm trong repo này do kích thước):

| Tài nguyên | Trạng thái | Link |
| :--- | :--- | :--- |
| Keypoint processed (`.npy`) | 🟡 Preview trên Google Drive | [🔗 Drive Folder](https://drive.google.com/drive/folders/1M_H0s_C6WhI4xfWCaXYv8REpTVmMJ0QG?usp=drive_link) |
| Keypoint raw (`.pose`) | 🟡 Preview trên Google Drive | [🔗 Drive Folder](https://drive.google.com/drive/folders/1M_H0s_C6WhI4xfWCaXYv8REpTVmMJ0QG?usp=drive_link) |
| Metadata (cam_front.json, gloss.csv) | ✅ Có sẵn | [🔗 Drive Folder](https://drive.google.com/drive/folders/1M_H0s_C6WhI4xfWCaXYv8REpTVmMJ0QG?usp=drive_link) |
| Bản chính thức (Zenodo/HF) | 🔜 Sắp có | — |
| Video gốc | ❌ Không phân phối | Xem DATASET_CARD.md |

**[→ Xem DATASET_CARD.md đầy đủ](https://drive.google.com/file/d/1HS4Js4pvt8A2zL6eozcuf9zTfAIUIfxK/view?usp=drive_link)**

---

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/thanhbinh55/VietnameseSignLanguageRecognition.git
cd VietnameseSignLanguageRecognition

# 2. Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

---

## Chạy nhanh với keypoint đã xử lý

```bash
# Tải keypoint processed từ Drive về data/preprocessed_vsl/
# Chỉnh data_dir trong config:
# data.data_dir: "data/preprocessed_vsl"

python src/train.py --config_path src/configs/training/spoter.yaml
```

## Chạy full pipeline từ video thô

Quy trình đầy đủ từ bước xử lý video thô đến huấn luyện và chạy thử nghiệm mô hình:

### Bước 1: Phát hiện biên và cắt video (TBL & BGSP)
Nếu bạn bắt đầu từ video ghi hình liên tục (chứa nhiều cử chỉ ký hiệu), cần định vị các khoảng thời gian cử chỉ diễn ra và cắt/crop video:

```bash
# 1.1 Xác định biên thời gian (tạo file CSV chứa mốc thời gian bắt đầu/kết thúc)
python src/data/temporal_boundary_localization.py \
    --input_video data/raw/your_video.mp4 \
    --get_cut_time \
    --overwrite

# 1.2 Cắt và crop video theo mốc thời gian đã tìm được (kết quả lưu trong thư mục con cùng tên)
python src/data/boundary_segmentation_pruning.py \
    --input_video data/raw/your_video.mp4 \
    --cut_crop_video \
    --overwrite
```

### Bước 2: Trích xuất keypoint từ video đã cắt

```bash
# Trích xuất tọa độ keypoint (định dạng MediaPipe .pose)
python src/extract_keypoints.py --video_dir data/raw/your_video --num_workers 4
```

### Bước 3: Xuất keypoint tiền xử lý (Interpolation & Normalization)

```bash
# Thực hiện các bước chuẩn hóa (ví dụ: Neck Anchor) và lưu thành file .npy
python src/preprocess_dataset.py \
    --config_path src/configs/training/spoter.yaml \
    --output_dir data/preprocessed_vsl
```

### Bước 4: Huấn luyện mô hình

```bash
python src/train.py --config_path src/configs/training/spoter.yaml
```

### Bước 5: Đánh giá mô hình

```bash
python src/evaluate_model.py --config_path src/configs/evaluation/spoter.yaml
```

### Demo với Webcam / Interface Web

```bash
python src/demo_web.py --config_path src/configs/inference/spoter.yaml
```

> [!TIP]
> **Lưu ý đối với hệ điều hành macOS (Apple Silicon):**
> - Dự án đã được cấu hình tương thích với thư viện `mediapipe-silicon` (import thông qua `mediapipe.python.solutions` thay vì `mediapipe.solutions` như thông thường để tránh lỗi `ModuleNotFoundError`).
> - Khi chạy các script trên macOS, bạn nên thêm cờ `-B` vào lệnh gọi python (ví dụ: `python -B src/train.py ...`) để ngăn Python cố gắng tạo/ghi đè file cache bytecode `.pyc` trong môi trường sandbox hệ thống nếu gặp vấn đề về quyền (permission).


---

## Cấu trúc Repo

```
vsl-keypoint-pipeline/
├── README.md                     ← bạn đang ở đây
├── CITATION.cff                  ← metadata citation chuẩn
├── LICENSE                       ← CC BY 4.0
├── requirements.txt
├── Makefile
│
├── docs/
│   ├── INDEX.md                  ← thứ tự đọc đề xuất
│   ├── vsl_linguistics.md        ← đặc trưng ngôn ngữ VSL
│   ├── vsl_technical_survey.md  ← khảo sát kỹ thuật & tài liệu liên quan
│   ├── architecture_analysis.md  ← phân tích kiến trúc pipeline
│   └── ablation_study_report.md  ← kết quả thực nghiệm
│
└── src/
    ├── train.py
    ├── evaluate_model.py
    ├── inference.py
    ├── preprocess_dataset.py
    ├── extract_keypoints.py
    ├── demo_web.py
    ├── configs/                  ← YAML cấu hình
    ├── data/                     ← TBL & BGSP preprocessing
    ├── features/                 ← DataLoader, transforms, augmentations
    ├── models/                   ← SPOTER & SL-GCN architectures
    ├── pipelines/                ← inference pipelines
    ├── tools/                    ← model & dataset loaders
    └── utils/                    ← constants, metrics, loggers
```

---

## Citation

Nếu bạn sử dụng code hoặc kết quả nghiên cứu này, vui lòng trích dẫn:

```bibtex
@software{vsl_keypoint_pipeline_2025,
  title   = {VSL Keypoint Pipeline: Preprocessing and Training for
             Vietnamese Sign Language Recognition},
  author  = {LVCF_VSL_KD},
  year    = {2025},
  url     = {https://github.com/thanhbinh55/VietnameseSignLanguageRecognition},
  license = {CC-BY-4.0}
}
```

Pipeline này được xây dựng dựa trên:

```bibtex
@dataset{nguyenquoc2026vsl400,
  author    = {Nguyen Quoc, Trung and others},
  title     = {{A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition}},
  year      = {2026},
  doi       = {10.5281/zenodo.17943574}
}

@inproceedings{bohacek2022spoter,
  author    = {Boháček, Matyáš and Hrúz, Marek},
  title     = {Sign Pose-Based Transformer for Word-Level Sign Language Recognition},
  booktitle = {WACV Workshops},
  year      = {2022}
}

@inproceedings{yan2018spatial,
  author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  booktitle = {AAAI},
  year      = {2018}
}
```

Ngoài ra, dự án còn tích hợp và sử dụng các thư viện nguồn mở:
- **pose-format** & **video-to-pose** ([Sign Language Processing](https://github.com/sign-language-processing)): Các công cụ tiêu chuẩn để đọc/ghi, chuyển đổi và chuẩn hóa định dạng dữ liệu pose của ngôn ngữ ký hiệu.
- **Google MediaPipe (Pose & Holistic)**: Trích xuất tọa độ khớp xương cơ thể và bàn tay thời gian thực.

---

## Giấy phép

Dự án này được cấp phép theo **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
Xem file [LICENSE](LICENSE) để biết chi tiết và danh sách các component bên thứ ba.

---

## Liên hệ

- GitHub Issues: [VietnameseSignLanguageRecognition/issues](https://github.com/thanhbinh55/VietnameseSignLanguageRecognition/issues)
- Email: *(để trống — điền sau)*
