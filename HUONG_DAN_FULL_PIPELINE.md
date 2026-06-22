# Hướng dẫn chạy thủ công FULL PIPELINE

Từ video gốc QIPEDC → bộ keypoint dataset (`.pose` + `.npy`) sẵn sàng train.

Pipeline gồm **3 giai đoạn**, nằm ở **2 repo**:

| GĐ | Việc | Repo / Package | Lệnh chính |
| :-- | :-- | :-- | :-- |
| **A** | Tách video nhiều cách + nhiều view | repo này — `qipedc_video_preprocess` | `python -m qipedc_video_preprocess.preprocess` |
| **A.2** | Chia signer + convert metadata | repo này — `qipedc2vsl400` | `python -m qipedc2vsl400.convert` |
| **B** | Trích keypoint + tiền xử lý `.npy` | repo này — thư mục `keypoint/` | `extract_keypoints.py` → `preprocess_dataset.py` |

> **Lưu ý nguồn:** Video gốc QIPEDC KHÔNG đi kèm repo. Bạn phải tự có video và đặt
> vào `Dataset/processed_videos/resize_720p/*.mp4` (và/hoặc `Dataset/raw_videos/`).

---

## 0. Chuẩn bị môi trường (1 lần)

Windows / PowerShell, mọi thứ cài vào `.venv` trên ổ `D:` (không đụng `C:`):

```powershell
# Tạo .venv + cài deps cho GIAI ĐOẠN A & A.2
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

# Mỗi phiên làm việc, trỏ PYTHONPATH vào src (package nằm trong src/, không cài editable)
$env:PYTHONPATH = "src"
```

Đặt dữ liệu đúng chỗ:

```
Dataset/labels/*.xlsx                          # nhãn QIPEDC (header: STT,ID,VIDEO,LABEL,REGION,TOPIC,signer)
Dataset/processed_videos/resize_720p/*.mp4     # video nguồn (ưu tiên 720p)
Dataset/raw_videos/**/*.mp4                     # nguồn dự phòng (tùy chọn)
```

---

## GIAI ĐOẠN A — Tách video (`qipedc_video_preprocess`)

Tách video nhiều **cách** (CÁCH 1/2/N — OCR số góc trên-trái) và nhiều **góc quay**
(front/side — ensemble hard-cut + pose).

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc_video_preprocess.preprocess
```

Lần đầu EasyOCR tự tải trọng số (~64 MB) về `Dataset/models/easyocr/`.

**Output:**
```
Dataset/processed_videos/split_variants/*.mp4            # clip tin cậy (confident); _side = góc phụ
Dataset/processed_videos/split_variant_predicted/*.mp4   # clip suy luận — nên kiểm tra ranh giới
Dataset/processed_videos/manual/                         # video cần duyệt tay + review_clips/
Dataset/processed_videos/labels_split.xlsx               # bảng nhãn mới
Dataset/logs/                                            # log + RunReport
```

> **Mẹo dùng cấu hình `result3` (đã chốt, GPU + 16 core):**
> `.\.venv\Scripts\python.exe Dataset\processed_videos\result3\run_result3.py`

### (Tùy chọn) Gộp clip predicted đã kiểm tra vào confident

Sau khi xem `split_variant_predicted/` và thấy đúng, copy chúng vào `split_variants/`
rồi dựng lại bảng nhãn cho đủ:

```powershell
# Copy clip predicted -> confident (đổi đường dẫn nếu dùng result3)
Copy-Item Dataset\processed_videos\split_variant_predicted\*.mp4 Dataset\processed_videos\split_variants\
# Dựng lại labels_split.xlsx cho đủ mọi clip (runner mẫu cho result3):
.\.venv\Scripts\python.exe Dataset\processed_videos\result3\rebuild_labels_result3.py
```

---

## GIAI ĐOẠN A.2 — Chia signer + convert metadata (`qipedc2vsl400`)

Gán signer (suy ra từ khuôn mặt: YuNet + SFace, gom cụm cosine) và xuất metadata.

```powershell
$env:PYTHONPATH = "src"
# Lần đầu (tự tải VSL400 labels từ Zenodo + model khuôn mặt ONNX về Dataset/models/)
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert
# Chạy lại nhanh (đã có model, không thêm clip mới):
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --skip-signer
```

**Output:**
```
Dataset/processed_videos/metadata/front_view.json   # metadata clip góc chính (signer_id, fps, num_frames...)
Dataset/processed_videos/metadata/side_view.json    # clip _side (chỉ tạo khi có clip nhiều góc)
Dataset/processed_videos/metadata/signers.csv       # video -> signer_id
Dataset/by_signer/signer_XXX/                        # clip gom theo signer để kiểm tra
```

> Muốn convert trên một cây output riêng (vd `result3`), ghi đè đường dẫn của
> `qipedc2vsl400.config.Config` bằng `dataclasses.replace` trong một runner script
> (trỏ `qipedc_labels_glob`, `video_search_dirs`, `output_dir` vào cây đó).

---

## Đồng bộ sang layout bộ dữ liệu công bố

Chuyển output GĐ A.2 sang layout `VSL Keypoint Dataset (QIPEDC-derived) - Public/`:

```powershell
.\.venv\Scripts\python.exe scripts\sync_public_dataset.py `
    --project-root . `
    --public-root  "<đường dẫn folder Public>" `
    --metadata-dir Dataset\processed_videos\metadata `
    --dry-run     # bỏ --dry-run khi chạy thật
```

Tạo: `cam_front.json`/`cam_side.json` (giữ nguyên `signer_id`), `labels.csv`,
`gloss.csv`, `split_info.csv` (chia train/val/test xác định theo video gốc).

---

## GIAI ĐOẠN B — Keypoint (`.pose` → `.npy`) — thư mục `keypoint/`

Code giai đoạn này nằm trong `keypoint/src/` (đã kèm sẵn trong repo). Đây là mã
**CC BY 4.0** từ dự án gốc VietnameseSignLanguageRecognition — xem `keypoint/LICENSE`
và `keypoint/CITATION.cff` (giữ nguyên ghi nhận nguồn).

### B0. Cài deps keypoint

Deps giai đoạn B đã nằm trong `requirements.txt` (phần "GIAI ĐOẠN B") nên
`setup_env.ps1` cài luôn. `pose-format` cung cấp CLI `video_to_pose` (MediaPipe
Holistic). Kiểm tra: `Test-Path .\.venv\Scripts\video_to_pose.exe`.

> Nếu chỉ chạy giai đoạn A/A.2, các gói nặng (torch...) vẫn bị cài. Muốn nhẹ thì
> tạm bỏ phần "GIAI ĐOẠN B" trong `requirements.txt` khi setup.

### B1. Trích keypoint (video `.mp4` → `.pose`)

Đặt clip vào `cam_front/` và `cam_side/` (theo hậu tố `_side`) rồi trích từng folder.
`.pose` được ghi cạnh mỗi `.mp4`.

```powershell
$env:PYTHONPATH = "keypoint\src"
.\.venv\Scripts\python.exe keypoint\src\extract_keypoints.py `
    --video_dir "<...>\cam_front" --num_workers 16
.\.venv\Scripts\python.exe keypoint\src\extract_keypoints.py `
    --video_dir "<...>\cam_side" --num_workers 16
```

> Nếu `extract_keypoints.py` lỗi import (kéo theo `transformers/wandb`), có thể viết
> driver gọn chỉ gọi `video_to_pose --format mediapipe -i <mp4> -o <pose>` song song
> (cùng phương pháp, không cần import `utils` của repo).

Sau đó copy `.pose` vào `Public/Keypoint/raw/{cam_front,cam_side}/`.

### B2. Tiền xử lý (`.pose` → `.npy`)

Cần `data_dir` chứa `gloss.csv`, `split_info.csv`, `cam_front.json`, `cam_side.json`
(đã có sau bước "Đồng bộ" ở trên) và `.pose` trong `keypoint_dir`.

```powershell
$env:PYTHONPATH = "keypoint\src"
.\.venv\Scripts\python.exe keypoint\src\preprocess_dataset.py `
    --data.dataset visl_400 --data.subset cam_front_side `
    --data.data_dir     "<Public>\Processed\metadata" `
    --data.keypoint_dir "<Public>\Keypoint\raw" `
    --model.arch spoter --model.num_frames 16 `
    --output_dir "<Public>\Keypoint\processed"
```

**Output:** `Keypoint/processed/{cam_front,cam_side}/{video_id}_preprocessed.npy`,
shape **`(16, 54, 2)`** float32 (16 frame × 54 joint × (x,y); Neck-anchor norm, bỏ mặt).

### B3. Huấn luyện (tùy chọn)

```powershell
$env:PYTHONPATH = "keypoint\src"
.\.venv\Scripts\python.exe keypoint\src\train.py `
    --config_path keypoint\src\configs\training\spoter.yaml
```

---

## Kiểm tra nhanh kết quả

```powershell
# Đếm clip / pose / npy
(Get-ChildItem "<Public>\Keypoint\raw\cam_front\*.pose").Count
(Get-ChildItem "<Public>\Keypoint\processed\cam_front\*.npy").Count

# Xem shape một .npy
.\.venv\Scripts\python.exe -c "import numpy as np; print(np.load(r'<...>_preprocessed.npy').shape)"
# -> (16, 54, 2)
```

---

## Tóm tắt phụ thuộc

| Giai đoạn | Cài bằng | Tự tải về |
| :-- | :-- | :-- |
| A (split) | `setup_env.ps1` (`requirements.txt`) | trọng số EasyOCR |
| A.2 (signer/convert) | `setup_env.ps1` | model ONNX YuNet/SFace + VSL400 labels (Zenodo) |
| B (keypoint) | `setup_env.ps1` (đã gộp vào `requirements.txt`) | (không) |

> `requirements.txt` phủ cả **3 giai đoạn**. Deps giai đoạn B nặng (torch ~2GB);
> nếu chỉ chạy A + A.2 có thể tạm bỏ phần đó khi cài.

## Ghi nhận nguồn

Code giai đoạn keypoint (`keypoint/`) là **CC BY 4.0**, dựa trên dự án
VietnameseSignLanguageRecognition và bộ dữ liệu gốc *"A Multi-view Dataset for
Vietnamese Word-Level Sign Language Recognition"* (DOI 10.5281/zenodo.17943574).
Xem `keypoint/LICENSE` và `keypoint/CITATION.cff`.
