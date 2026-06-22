# Pipeline xử lý dữ liệu Ngôn ngữ Ký hiệu QIPEDC

Toàn bộ pipeline để biến **video QIPEDC** thành **bộ keypoint dataset** sẵn sàng huấn luyện, gồm ba
giai đoạn trong cùng một repo:

| Giai đoạn | Package / Thư mục | Vai trò |
| :-- | :-- | :-- |
| **A. Tách video** | `src/qipedc_video_preprocess` | Tách video nhiều **cách** (CÁCH 1/2/N) và nhiều **góc quay** (front/side) → clip một-cách, một-góc |
| **A.2. Chia signer + metadata** | `src/qipedc2vsl400` | Suy ra `signer_id` từ khuôn mặt, xuất `front_view.json`/`side_view.json` |
| **B. Keypoint** | `keypoint/src` (CC BY 4.0) | Clip `.mp4` → `.pose` (MediaPipe Holistic) → `.npy` chuẩn hóa → SPOTER / SL-GCN |

Bộ chuyển đổi (giai đoạn A.2) tạo metadata là **superset của VSL400**: giữ các trường gốc của QIPEDC
(`region`, `topic`, `stt`, `id`), đổi `LABEL` → `gloss`, giữ `video_id` từ `VIDEO`, và thêm các trường
suy ra (`signer_id`, `fps`, `resolution`, `num_frames`, `length_seconds`). QIPEDC không có sẵn thông
tin signer nên `signer_id` được **suy ra từ nội dung video** (OpenCV YuNet phát hiện mặt + SFace
embedding, gom cụm theo ngưỡng).

> **Hướng dẫn chạy đầy đủ từng bước:** xem [`HUONG_DAN_FULL_PIPELINE.md`](HUONG_DAN_FULL_PIPELINE.md).

## Cấu trúc thư mục

### Mã nguồn (code)

```
src/qipedc_video_preprocess/         # GIAI ĐOẠN A — tách video nhiều cách / nhiều góc
├─ preprocess.py                     #   CLI tách tự động (OCR + pose ensemble)
├─ manual_cut.py                     #   CLI cắt THỦ CÔNG theo mốc giây tự nhập
├─ manual_cuts.example.csv           #   bảng mẫu cho manual_cut
├─ segmenter.py                      #   phân đoạn theo số CÁCH (OCR)
├─ multiview_detector.py             #   tìm hard-cut tách góc front/side
├─ pose_boundary.py                  #   dò ranh giới bằng pose (xác nhận hard-cut)
└─ config.py                         #   cấu hình + ngưỡng (PreprocessConfig)

src/qipedc2vsl400/                   # GIAI ĐOẠN A.2 — chia signer + convert metadata
├─ convert.py                        #   CLI điều phối (fetch → signer → map → write → verify)
├─ signer_extractor.py               #   YuNet + SFace, gom cụm signer
├─ mapper.py / writer.py             #   dựng + ghi front_view.json / side_view.json
└─ config.py                         #   cấu hình (Config)

keypoint/src/                        # GIAI ĐOẠN B — keypoint (mã CC BY 4.0)
├─ extract_keypoints.py              #   video .mp4 → .pose (MediaPipe Holistic)
├─ preprocess_dataset.py             #   .pose → .npy chuẩn hóa (16, 54, 2)
├─ train.py                          #   huấn luyện SPOTER / SL-GCN
└─ configs/, models/, features/      #   cấu hình + kiến trúc model

scripts/sync_public_dataset.py       # đồng bộ output → layout bộ dữ liệu công bố
tests/                               # unit test + property-based (Hypothesis)
setup_env.ps1                        # tạo .venv trên D: + cài requirements.txt
```

### Dữ liệu & output (git-ignore, sinh khi chạy)

```
Dataset/labels/*.xlsx                              # nhãn QIPEDC nguồn (được track)
Dataset/processed_videos/resize_720p/*.mp4         # video nguồn (tự đặt vào)
Dataset/processed_videos/split_variants/           # clip đã tách (kết quả tin cậy)
Dataset/processed_videos/split_variant_predicted/  # clip cắt bằng suy luận — cần kiểm ranh giới
Dataset/processed_videos/manual/                   # video cần cắt thủ công
Dataset/processed_videos/labels_split.xlsx         # bảng nhãn mới sau khi tách
Dataset/processed_videos/metadata/                 # front_view.json, side_view.json, signers.csv
Dataset/by_signer/                                 # clip gom theo signer (để kiểm tra)
Dataset/models/                                    # model tải về (EasyOCR, YuNet, SFace)
```

Dữ liệu lớn (`Dataset/raw_videos/`, `Dataset/processed_videos/`, `Dataset/by_signer/`,
`Dataset/models/`), `.venv/` và cache đều được git-ignore. **Không cài gì lên `C:` — mọi thứ nằm
dưới project trên `D:`.**

## Cài đặt môi trường (Windows / PowerShell)

`setup_env.ps1` tạo `.venv` trên `D:`, đổi cache pip sang `D:`, và cài `requirements.txt` (đã gồm cả
deps giai đoạn keypoint):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

Mỗi phiên làm việc, đặt biến môi trường một lần (package nằm trong `src\`, không cài editable):

```powershell
$env:PYTHONPATH = "src"
```

> Có thể dùng trực tiếp `.\.venv\Scripts\python.exe` mà không cần "activate" venv.

---

## GIAI ĐOẠN A — Tách video (`qipedc_video_preprocess`)

Một số video QIPEDC gộp **2–3 cách biểu diễn** của **cùng một từ** vào một file. Con số ở **góc trên
bên trái** ("CÁCH 1", "CÁCH 2", …) đánh dấu từng cách. Công đoạn này tự phát hiện con số bằng OCR, tách
video nhiều cách thành nhiều video con (mỗi cách một file), đồng thời tách **nhiều góc quay**
(front/side), và xuất bảng nhãn mới.

Giai đoạn A có **hai công cụ tách**:

- **Tách tự động** — `qipedc_video_preprocess.preprocess` ([`src/qipedc_video_preprocess/preprocess.py`](src/qipedc_video_preprocess/preprocess.py)): dùng OCR + pose ensemble tự tìm điểm cắt.
- **Cắt thủ công** — `qipedc_video_preprocess.manual_cut` ([`src/qipedc_video_preprocess/manual_cut.py`](src/qipedc_video_preprocess/manual_cut.py)): bạn tự nhập mốc giây cắt (nhiều cách / nhiều view) trong một bảng CSV/XLSX, không dùng OCR — xem mục [Cắt thủ công](#cắt-thủ-công-theo-mốc-tự-nhập-manual_cut) bên dưới.

### Cơ chế phát hiện

Pipeline lấy mẫu ~1 frame/giây và chạy 2 lượt OCR trên vùng ROI rộng (40%×25% góc trên trái):

1. **Lượt 1 — token đầy đủ**: tìm token khớp với "CÁCH" (bao gồm các biến thể OCR do logo che: `"Cach"`, `"Bach"`, `"'SacH 2'"`, `"P█GH 1"`…). Nếu thấy → đọc số kèm theo hoặc token số gần nhất bên phải.
2. **Lượt 2 — mảnh + số lân cận**: khi chỉ thấy một phần nhỏ (`"ca"`, `"ch"`, `"á"`…), tìm token số gần đó trong ROI.
3. **Fallback glyph-width**: khi logo che cả con số, phóng to vùng bên phải "CÁCH" và đo tỉ lệ chiều rộng pixel của glyph — số "1" hẹp (~0.33), số "2" rộng hơn (~0.37) — kết hợp với OCR để quyết định.

Ranh giới giữa các cách cần `boundary_confirm_frames` (mặc định 2) frame mẫu liên tiếp cùng giá trị số
mới mới được chấp nhận. Tách góc front/side dùng **ensemble**: OCR/hard-cut đề xuất điểm cắt, tín hiệu
**pose** (tay hạ xuống giữa hai góc) xác nhận.

### Chạy

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc_video_preprocess.preprocess
```

Mặc định: duyệt `Dataset\processed_videos\resize_720p\` (ưu tiên) rồi `Dataset\raw_videos\`, đọc nhãn
từ `Dataset\labels\*.xlsx`, ghi clip ra `Dataset\processed_videos\split_variants\` và bảng nhãn ra
`Dataset\processed_videos\labels_split.xlsx`. Lần đầu EasyOCR tự tải trọng số (~64 MB) về
`Dataset\models\easyocr\` (trên `D:`, không lên `C:`).

| Flag | Ý nghĩa |
| --- | --- |
| `--project-root <path>` | Đổi thư mục gốc (mặc định: thư mục hiện tại). |
| `--sample-interval <giây>` | Khoảng lấy mẫu frame, `0.1`–`5.0` (mặc định `1.0`). |
| `--safety-margin <frame>` | Số frame bỏ mỗi phía ranh giới, `0`–`60` (mặc định `3`). |
| `--ocr-threshold <float>` | Ngưỡng tin cậy OCR tối thiểu, `0.0`–`1.0` (mặc định `0.5`). |
| `--dry-run` | Chỉ duyệt + phân loại, không ghi clip/nhãn. |

### Kết quả

```
Dataset\processed_videos\split_variants\<id>.mp4       # video một cách (giữ nguyên tên); _side = góc phụ
Dataset\processed_videos\split_variants\<id>_c1.mp4 …  # mỗi cách một video con (multi xác nhận)
Dataset\processed_videos\labels_split.xlsx             # bảng nhãn mới (STT đánh lại 1..M)
Dataset\processed_videos\split_variant_predicted\<id>_cN.mp4   # clip đã cắt bằng suy luận (cần kiểm ranh giới)
Dataset\processed_videos\split_variant_predicted\<id>.mp4      # bản gốc đặt cạnh để so sánh
Dataset\processed_videos\manual\<id>.mp4               # bản gốc video cần cắt thủ công
Dataset\logs\preprocess_<timestamp>.log                # log + RunReport
```

Ba nhóm kết quả:

| Nhóm | Ý nghĩa | Việc cần làm |
| --- | --- | --- |
| `split_variants/` | OCR xác nhận ranh giới (≥2 frame ổn định mỗi cách) | Không cần — dùng ngay |
| `split_variant_predicted/` | Ranh giới suy luận từ khối ổn định; OCR không đọc trọn dãy số | Kiểm tra điểm cắt |
| `manual/` | Có nhãn "CÁCH" nhưng không xác định được ranh giới | Cắt tay |

### Hiệu chỉnh ROI con số (quan trọng)

Con số "CÁCH" nằm **ngay bên phải logo QIPEDC** ở góc trên trái. ROI hẹp để OCR đọc số (`roi_top_left`,
tọa độ tỉ lệ `0.0`–`1.0`) mặc định `(0.19, 0.05, 0.25, 0.22)` — hiệu chỉnh thực nghiệm trên video
1280×720: cô lập đúng con số, bỏ logo và chữ "H" của "CÁCH". ROI quá rộng (ôm cả logo) khiến OCR đọc
nhiễu (chuỗi 2 chữ số) → bị loại → **mọi video bị phân loại nhầm thành một cách**. Nếu bố cục video
khác, dump vài frame, xem góc trên trái, rồi chỉnh `roi_top_left` trong
`src\qipedc_video_preprocess\config.py`.

### Cắt thủ công theo mốc tự nhập (`manual_cut`)

Với video OCR không xử lý được (nằm trong `manual/`), hoặc khi đã biết chính xác mốc cắt, dùng
`manual_cut` để **tự khai báo điểm cắt** trong một bảng CSV/XLSX — không dùng OCR. Tool cắt và đặt tên
đúng quy ước (`_c1/_c2/…`, `_side`).

- **Code:** [`src/qipedc_video_preprocess/manual_cut.py`](src/qipedc_video_preprocess/manual_cut.py)
- **Bảng mẫu:** [`src/qipedc_video_preprocess/manual_cuts.example.csv`](src/qipedc_video_preprocess/manual_cuts.example.csv)

**Bước 1 — tạo bảng cắt** (header không phân biệt hoa/thường):

| Cột | Ý nghĩa |
| :-- | :-- |
| `video_id` | stem video nguồn, vd `W00738` |
| `mode` | `multiway` (nhiều cách) \| `multiview` (front/side) \| `both` |
| `cut_seconds` | mốc giây bắt đầu của cách 2,3… — vd `2.5` hoặc `2.5,5.0` |
| `view_cut_seconds` | mốc giây cắt front→side. `multiview`: một mốc; `both`: một mốc mỗi cách |
| `notes` | ghi chú (không bắt buộc) |

Ví dụ nội dung (`manual_cuts.csv`):

```csv
video_id,mode,cut_seconds,view_cut_seconds,notes
W00738,multiway,2.5,,Cach 2 bat dau o giay 2.5  -> W00738_c1.mp4 + W00738_c2.mp4
D0530,multiview,,3.0,Cat front->side o giay 3.0 -> D0530.mp4 + D0530_side.mp4
W01234,both,5.0,"2.0,7.0",2 cach, moi cach 1 moc front->side -> _c1/_c1_side + _c2/_c2_side
W00999,multiway,"2.5,5.0",,3 cach (moc bat dau cach 2 va cach 3) -> _c1/_c2/_c3
```

**Bước 2 — chạy** (thử `--dry-run` để xem kế hoạch trước, bỏ đi để ghi clip thật):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc_video_preprocess.manual_cut `
    --input Dataset\processed_videos\manual_cuts.csv --dry-run
# bỏ --dry-run để ghi clip thật vào split_variants/
```

Cắt theo **đúng** mốc giây nhập vào (không trừ safety-margin, không suy đoán), ghi thẳng vào
`split_variants/` như kết quả đã duyệt. Cờ khác: `--project-root <path>`.

---

## GIAI ĐOẠN A.2 — Chia signer + convert metadata (`qipedc2vsl400`)

### Cơ chế phân cụm signer

Mỗi clip được rút thành một **vector embedding khuôn mặt 128 chiều** (OpenCV YuNet phát hiện mặt →
SFace sinh embedding → trung bình, chuẩn hóa L2). Hai clip được coi là **cùng một người** khi hai
vector gần nhau theo **cosine distance** (ngưỡng `--signer-threshold`, mặc định `0.363`). Clip không
thấy mặt rơi vào nhóm `unknown`.

Lưu ý đánh số: trong **một lần chạy**, mọi clip của một người luôn cùng một `signer_id`. Nhưng **con
số** `signer_001/002/...` có thể đổi nhãn giữa các lần chạy (trừ khi dùng `--stable-signers`).

### Chạy (các tình huống)

`...` = `.\.venv\Scripts\python.exe -m qipedc2vsl400.convert`, sau khi đã `$env:PYTHONPATH = "src"`.

**Trường hợp 1 — Chạy lần đầu từ đầu (máy mới)**

1. Đặt dữ liệu: label QIPEDC ở `Dataset\labels\*.xlsx` (header `STT, ID, VIDEO, LABEL, REGION, TOPIC, signer`); video ở `Dataset\processed_videos\resize_720p\*.mp4` (và/hoặc `Dataset\raw_videos\`).
2. Tạo venv: `powershell -ExecutionPolicy Bypass -File .\setup_env.ps1`
3. Chạy (lần đầu tự tải VSL400 labels + model khuôn mặt): `... ` (không cờ).
4. Kết quả: `Dataset\processed_videos\metadata\front_view.json`, `signers.csv`, `Dataset\by_signer\...`, log trong `Dataset\logs\`. Mã thoát `0` = verify đạt (`echo $LASTEXITCODE`).

**Trường hợp 2 — Đã chạy 1 lần, thêm video mới (KHÔNG xóa video cũ).** Nhớ thêm dòng label cho video mới.

```powershell
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --embeddings-cache            # lần đầu: dựng cache
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --embeddings-cache  # sau: chỉ embed clip mới
```
- `--no-fetch`: bỏ tải mạng. **Không** dùng `--skip-signer` ở đây (clip mới sẽ rơi vào `unknown`).
- Phân cụm lại trên toàn bộ (cũ + mới); số `signer_id` có thể đổi nhãn so với lần trước.

**Trường hợp 3 — Video quá nặng, chạy theo từng đợt (xóa cũ, thêm mới).** Dùng `--batch`: clip store tích lũy nên metadata cuối luôn đủ cả các đợt, kể cả video đã xóa.

```powershell
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch   # lặp lại cho từng đợt
```
- `--batch` giả định model đã tải sẵn (chạy TH1 một lần trước). Không tự fetch.
- `Dataset\by_signer\` chỉ chứa được video còn trên đĩa; clip đợt đã xóa không có thư mục xem lại (nhưng metadata vẫn đủ).

**Trường hợp 4 — Muốn `signer_id` CỐ ĐỊNH giữa các lần chạy.** Thêm `--stable-signers` (kết hợp `--batch`): người khớp signer cũ giữ nguyên số; người mới được cấp số mới; số cũ không đổi.

```powershell
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch --stable-signers
# gom lại tối ưu toàn cục + đánh số lại:
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch --stable-signers --recluster
```
- Chế độ ổn định gán "trực tuyến" (phụ thuộc thứ tự) và **không tự gộp** hai `signer_id` đã đăng ký — dùng `--recluster` khi cần gộp/đặt lại số.

**Trường hợp 5 — Chạy lại nhanh trên dữ liệu không đổi:**

```powershell
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --skip-signer
```
`--skip-signer` dùng lại `signers.csv` đã có (không nhúng lại). Chỉ dùng khi **không** thêm clip mới.

**Trường hợp 6 — Đổi đường dẫn video / output / thư mục signer.** Các đường dẫn nằm trong `Config` (`src\qipedc2vsl400\config.py`).

- **Cách A** — sửa mặc định trong `config.py`: `video_search_dirs`, `foldering_source`, `by_signer_dir`, `output_dir`, `output_view_name`, `qipedc_labels_glob`, `models_dir`.
- **Cách B** — ghi đè bằng `dataclasses.replace` trong một runner script (không sửa code gốc):
  ```python
  from pathlib import Path
  import dataclasses
  from qipedc2vsl400.config import Config
  from qipedc2vsl400.convert import run_pipeline

  cfg = Config(project_root=Path.cwd())
  cfg = dataclasses.replace(
      cfg,
      video_search_dirs=("E:/my_videos",),
      foldering_source="E:/my_videos",
      output_dir="E:/output/metadata",
  )
  raise SystemExit(run_pipeline(cfg, fetch=False).exit_code)
  ```

**Trường hợp 7 — Xóa bớt clip khỏi dataset.** Xóa **cả dòng label** của clip (và file video). Chế độ batch: dùng `--prune-missing` (với `--batch`) để bỏ khỏi kho các clip không có trong dòng label hiện tại — chỉ dùng khi file label đang liệt kê **toàn bộ** dataset mong muốn.

### Bảng chọn nhanh

| Tình huống | Lệnh |
| --- | --- |
| Lần đầu từ đầu | `... convert` |
| Thêm video mới (giữ video cũ) | `... convert --no-fetch --embeddings-cache` |
| Chạy lại nhanh, dữ liệu không đổi | `... convert --no-fetch --skip-signer` |
| Chạy theo đợt (xóa cũ thêm mới) | `... convert --batch` |
| Chạy theo đợt + giữ cố định số signer | `... convert --batch --stable-signers` |
| Gom lại + đánh số lại (chế độ ổn định) | `... convert --batch --stable-signers --recluster` |

(`...` = `.\.venv\Scripts\python.exe -m qipedc2vsl400`, sau khi đã `$env:PYTHONPATH = "src"`.)

### Kiểm tra kết quả nhanh

```powershell
echo $LASTEXITCODE                                    # 0 = verify đạt
Get-Content Dataset\by_signer\_summary.txt
Get-ChildItem Dataset\processed_videos\metadata\
```

---

## GIAI ĐOẠN B — Keypoint (`keypoint/`)

Mã trích keypoint + tiền xử lý + huấn luyện nằm trong `keypoint/src/` (mã **CC BY 4.0**, dựa trên dự án
[VietnameseSignLanguageRecognition](https://github.com/thanhbinh55/VietnameseSignLanguageRecognition);
ghi nhận nguồn trong `keypoint/LICENSE` + `keypoint/CITATION.cff`). Deps đã nằm trong `requirements.txt`
(phần "GIAI ĐOẠN B": `pose-format`, `torch`, `transformers`, `mediapipe`…).

```powershell
# B1 — trích keypoint (video .mp4 -> .pose)
$env:PYTHONPATH = "keypoint\src"
.\.venv\Scripts\python.exe keypoint\src\extract_keypoints.py --video_dir "<...>\cam_front" --num_workers 16

# B2 — tiền xử lý (.pose -> .npy), shape (16, 54, 2)
.\.venv\Scripts\python.exe keypoint\src\preprocess_dataset.py `
    --data.dataset visl_400 --data.subset cam_front_side `
    --data.data_dir "<Public>\Processed\metadata" --data.keypoint_dir "<Public>\Keypoint\raw" `
    --model.arch spoter --model.num_frames 16 --output_dir "<Public>\Keypoint\processed"

# B3 — huấn luyện
.\.venv\Scripts\python.exe keypoint\src\train.py --config_path keypoint\src\configs\training\spoter.yaml
```

Mô hình: **SPOTER** (Pose Transformer) / **SL-GCN**. Chuẩn hóa **Neck Anchor**, nội suy keypoint thiếu,
bỏ landmark mặt. Chi tiết từng bước: xem [`HUONG_DAN_FULL_PIPELINE.md`](HUONG_DAN_FULL_PIPELINE.md).

---

## Đồng bộ sang bộ dữ liệu công bố

`scripts/sync_public_dataset.py` chuyển output (`metadata/front_view.json` + `side_view.json`) sang
layout bộ dữ liệu công bố (`cam_front.json`/`cam_side.json`, `labels.csv`, `gloss.csv`,
`split_info.csv`), định tuyến clip `_side` sang góc phụ. `gloss_id` xác định (vị trí gloss trong danh
sách gloss duy nhất đã sắp xếp); `split_info.csv` chia train/val/test xác định theo video gốc nên các
cách/góc của cùng một clip không bị tách rời giữa các split.

```powershell
.\.venv\Scripts\python.exe scripts\sync_public_dataset.py --project-root . --public-root "<thư-mục-công-bố>" --dry-run
```

---

## Kiểm thử & Yêu cầu

```powershell
python -m pytest
```

Xem `requirements.txt` cho danh sách phụ thuộc đã ghim (giai đoạn A/A.2: `openpyxl`,
`opencv-python-headless`, `numpy`, `scikit-learn`, `easyocr`, `mediapipe`…; giai đoạn B: `pose-format`,
`torch`, `transformers`…).
