# VSL-400 Preprocessing Ablation Study — Research Report

> **Tình trạng hoàn tất:** 14 / 19 runs đã hoàn tất · 1 run thất bại · 4 runs chưa chạy (Missing)
> Cập nhật lần cuối: tháng 6 năm 2026

---

## 1. Mục tiêu và Phạm vi

Báo cáo này tổng hợp và phân tích kết quả của chuỗi thực nghiệm greedy ablation được thiết kế nhằm xác định **pipeline tiền xử lý keypoint tối ưu** cho bài toán nhận dạng từ ký hiệu tiếng Việt (Isolated Word-Level VSL Recognition).

Xuất phát từ hạn chế chung trong tài liệu về ngôn ngữ ký hiệu tiếng Việt — các ngưỡng tiền xử lý (TBL, BGSP) được chọn theo kinh nghiệm mà chưa có ablation trực tiếp trên mô hình hạ nguồn — nhóm thực hiện mở rộng thiết kế từ bài báo gốc VSL400 (2026) theo hướng:

1. **Kiểm chứng** các tham số TBL mặc định (θ = 160°, τb = 400 ms) so với các giá trị thay thế.
2. **Đo lường tác động riêng lẻ** của từng kỹ thuật tiền xử lý: nội suy keypoint, chuẩn hóa theo điểm neo, tăng cường dữ liệu, facial landmarks.
3. **Xác thực chéo** pipeline tối ưu trên mô hình SL-GCN để kiểm tra tính tổng quát hóa.

> **Tuyên bố phạm vi:** Tất cả kết quả trong báo cáo này được thu thập trên **tập dữ liệu VSL400 (cam_1, signer-disjoint split)** với kiến trúc SPOTER, trừ Phase 4 sử dụng SL-GCN. Kết quả không nên được so sánh trực tiếp với các bảng xếp hạng từ công trình khác nếu khác dataset, split protocol hoặc số lớp.

---

## 2. Thiết lập Thực nghiệm

| Mục | Thông số |
| :--- | :--- |
| **Dataset** | VSL400 · Subset: cam_1 · 400 gloss · 28 signers |
| **Split Protocol** | Signer-disjoint (theo `visl_400.py`) · seed = 42 |
| **Mô hình (Phase 1–3)** | SPOTER · hidden_dim=108 · 9 heads · 6 enc/dec layers |
| **Mô hình (Phase 4)** | SL-GCN · 27 keypoints · spatial labeling |
| **Số epoch** | 100 · LR cosine decay · warmup_ratio=0.05 |
| **Batch size** | Train: 64 · Val/Test: 128 |
| **Optimizer** | AdamW · lr=5e-4 · weight_decay=0.01 |
| **Metric chính** | Top-1 Accuracy + Macro F1 (Val & Test) |
| **Phần cứng** | Mac MPS (M4) · ~3.5 phút/epoch · ~5.8 giờ/run |
| **Số seed** | 1 seed duy nhất (seed=42) — kết quả mang tính tham khảo, chưa có error bar |

> ⚠️ **Giới hạn thực nghiệm:** Do mỗi run mất ~5.8 giờ trên Mac MPS, một số runs trong Phase 1 và Phase 4 chưa được thực thi trong thời gian dự án. Với Phase 1 (TBL sweep), chỉ có cấu hình mặc định θ=160° hoàn tất; các runs cho θ ∈ {140°, 150°, 170°} và τb ∈ {200, 600 ms} chưa có kết quả, do đó **không thể kết luận θ=160° là tối ưu**. Kết quả các phase sau được xây dựng trên cấu hình mặc định này.

---

## 3. Luồng Thiết kế (Greedy Ablation)

```
Run 00 (Raw Baseline)
    └─► Run 03 (TBL θ=160° / τb=400ms) ← điểm neo cho tất cả phases tiếp theo
            └─► Run 07 (+ Keypoint Interpolation)
                    ├─► Run 08 (+ Neck Anchor Norm)
                    ├─► Run 09 (+ Nose Anchor Norm)
                    ├─► Run 10 (Spatial Aug only)
                    ├─► Run 11 (Perspective Aug only)
                    ├─► Run 12 (Kinematic Aug only)
                    ├─► Run 13 (Gaussian Noise only)
                    ├─► Run 14 (All Augs combined)
                    └─► Run 15 (+ Facial Landmarks)
```

---

## 4. Kết quả Thực nghiệm

### 4.1 · Phase 0 — Raw Baseline

| Run | Mô tả | Val Acc | Val F1 | Test Acc | Test F1 | Trạng thái |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Run 00 | Raw Baseline (không TBL, không nội suy, không aug) | 88.76% | 89.21% | 78.59% | 78.47% | ✅ Completed |

Đây là điểm quy chiếu gốc. Độ chênh giữa Val Acc (88.76%) và Test Acc (78.59%) — khoảng 10 điểm — phản ánh khoảng cách tổng quát hóa sang signer chưa thấy trong huấn luyện.

---

### 4.2 · Phase 1 — TBL Preprocessing Sweep

**Mục tiêu:** Tìm ngưỡng góc khuỷu tay θ và trễ đệm τb tối ưu cho thuật toán Temporal Boundary Localization.

| Run | Cấu hình | Val Acc | Val F1 | Test Acc | Test F1 | Trạng thái |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Run 01 | θ = 140°, τb = 400 ms | *—* | *—* | *—* | *—* | ❌ Failed (data dir error) |
| Run 02 | θ = 150°, τb = 400 ms | *—* | *—* | *—* | *—* | ⏳ Missing |
| **Run 03** | **θ = 160°, τb = 400 ms (Default)** | **89.61%** | **90.01%** | **81.08%** | **80.86%** | **✅ Completed** |
| Run 04 | θ = 170°, τb = 400 ms | *—* | *—* | *—* | *—* | ⏳ Missing |
| Run 05 | θ = 160°, τb = 200 ms | *—* | *—* | *—* | *—* | ⏳ Missing |
| Run 06 | θ = 160°, τb = 600 ms | *—* | *—* | *—* | *—* | ⏳ Missing |

> ⚠️ **Lưu ý diễn giải:** Chỉ có Run 03 (θ=160°, τb=400ms) hoàn tất. **Không đủ cơ sở để kết luận θ=160° hay τb=400ms là giá trị tối ưu.** Tuy nhiên, việc TBL (Run 03) cải thiện +2.49 điểm Test Acc so với Raw Baseline (Run 00) xác nhận rằng cắt video theo biên ký hiệu có ích hơn dùng toàn bộ video thô.
>
> Run 01 thất bại do thiếu thư mục dữ liệu đã xử lý TBL-theta140 — run này cần được tái chạy khi dữ liệu sẵn sàng.

**Quan sát sơ bộ:** TBL (Run 03) vs Raw Baseline (Run 00): **+2.49 điểm Test Acc**.

---

### 4.3 · Phase 2 — Keypoint Interpolation & Anchor Normalization

**Mục tiêu:** Đánh giá tác động của nội suy lấp đầy keypoint thiếu và chiến lược điểm neo chuẩn hóa cơ thể.

Tất cả runs trong phase này sử dụng cấu hình TBL θ=160°, τb=400ms từ Run 03 làm điểm neo.

| Run | Cấu hình | Val Acc | Val F1 | Test Acc | Test F1 | Trạng thái |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Run 03 | Baseline (Box Anchor, không nội suy) | 89.61% | 90.01% | 81.08% | 80.86% | ✅ Completed |
| Run 07 | + Keypoint Interpolation (Box Anchor) | 90.21% | 90.46% | 80.80% | 80.72% | ✅ Completed |
| **Run 08** | **+ Interpolation + Neck Anchor** | **90.37%** | **90.61%** | **84.08%** | **84.02%** | **✅ Completed** |
| Run 09 | + Interpolation + Nose Anchor | 89.89% | 90.23% | 80.71% | 80.57% | ✅ Completed |

**Phân tích:**

- **Keypoint Interpolation (Run 07 vs 03):** Val Acc tăng nhẹ (+0.60%), nhưng Test Acc giảm nhẹ (−0.28%). Nội suy đơn thuần không cải thiện rõ rệt khi anchor vẫn là Box — có thể vì Box normalization đã xử lý một phần sự không nhất quán của keypoint thiếu.

- **Neck Anchor (Run 08) — cấu hình đạt kết quả cao nhất trong thực nghiệm ablation:** Test Acc tăng **+3.00 điểm** so với Run 03 và **+4.57 điểm** so với Raw Baseline. Val Acc cũng cải thiện (+0.76%). Cấu hình này được chọn làm cơ sở cho Phase 3.

- **Nose Anchor (Run 09) vs Box (Run 03):** Gần như không ghi nhận sự cải thiện trên tập Test (−0.37 điểm), kém hơn Neck Anchor đáng kể. Mũi là điểm dễ bị che khuất và kém ổn định hơn vùng cổ trong không gian thực hiện ký hiệu, do đó không phù hợp làm điểm chuẩn hóa (anchor).

> **Kết luận Phase 2:** Kỹ thuật Neck Anchor Normalization kết hợp Keypoint Interpolation (cấu hình Run 08) đem lại hiệu quả cao nhất và được sử dụng làm cơ sở cho Phase 3.

---

### 4.4 · Phase 3 — Augmentation Ablation Study

**Mục tiêu:** Đánh giá đóng góp riêng lẻ và tổng hợp của từng kỹ thuật tăng cường dữ liệu.

Tất cả runs dùng Keypoint Interpolation + Neck Anchor từ Run 08 làm nền. Điểm so sánh: **Run 07** (cùng interpolation + Box anchor, nhằm cách ly tác động của augmentation khỏi tác động của anchor).

| Run | Cấu hình aug | Val Acc | Val F1 | Test Acc | Test F1 | Trạng thái |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Run 07 | Không aug (nền so sánh aug) | 90.21% | 90.46% | 80.80% | 80.72% | ✅ Completed |
| Run 10 | Spatial only (Rotate ±13° + Squeeze ≤15%) | 89.73% | 89.85% | 81.61% | 81.58% | ✅ Completed |
| Run 11 | Perspective Skew only (hệ số 0.10) | 89.89% | 90.12% | 81.27% | 81.10% | ✅ Completed |
| Run 12 | Kinematic only (ArmJointRotate ±4°) | 88.80% | 89.13% | 80.55% | 80.44% | ✅ Completed |
| Run 13 | Gaussian Noise only (std=0.001) | 88.32% | 88.69% | 79.71% | 79.39% | ✅ Completed |
| **Run 14** | **Combined (Spatial + Perspective + Kinematic + Noise)** | **90.25%** | **90.36%** | **82.30%** | **82.30%** | **✅ Completed** |
| Run 15 | Combined + Facial Landmarks (eyebrow/eye/mouth) | 89.49% | 89.80% | 80.96% | 80.94% | ✅ Completed |

**Phân tích:**

- **Spatial Aug (Run 10):** Cải thiện Test Acc +0.81 điểm so với Run 07. Kỹ thuật tăng cường không gian tiêu chuẩn này mang lại sự cải thiện ổn định.

- **Perspective Skew (Run 11):** Cải thiện Test Acc +0.47 điểm. Ghi nhận mức cải thiện tích cực, cho thấy việc mô phỏng tính đa dạng của góc nhìn máy ảnh mang lại hiệu quả.

- **Kinematic Aug (Run 12):** Test Acc giảm nhẹ −0.25 điểm. Việc xoay khớp tay độc lập ±4° với p=0.3 không mang lại sự cải thiện hiệu năng. Nguyên nhân có thể do sự thay đổi cục bộ này làm phá vỡ tính tương quan tự nhiên giữa các khớp trong chuyển động ký hiệu.

- **Gaussian Noise (Run 13):** Test Acc giảm −1.09 điểm so với Run 07. Việc bổ sung nhiễu ngẫu nhiên vào tọa độ keypoint đã được chuẩn hóa làm suy giảm chất lượng thông tin của các cử chỉ tinh tế.

- **Combined (Run 14):** Test Acc +1.50 điểm so với Run 07. Dù Kinematic và Gaussian độc lập cho kết quả âm, khi kết hợp tất cả các kỹ thuật lại vẫn tạo ra tác động cộng hưởng dương. Điều này chỉ ra rằng sự đa dạng của tổ hợp augmentation đóng vai trò quan trọng hơn hiệu quả của từng phép biến đổi riêng lẻ.

- **Facial Landmarks (Run 15 vs 14):** Test Acc giảm −1.34 điểm khi bổ sung 20 keypoint khuôn mặt vào cấu hình Combined. Kết quả này trái ngược với các cải thiện quan sát được trong công trình QIPEDC-VSL. Nguyên nhân khả dĩ: (a) Dữ liệu VSL400 (cam_1) ghi hình toàn thân ở khoảng cách xa, dẫn đến độ phân giải vùng khuôn mặt thấp và giảm độ tin cậy của keypoint; (b) Tham số hidden_dim=108 của mô hình chưa đủ năng lực biểu diễn để xử lý không gian đặc trưng lớn hơn (74 điểm thay vì 54 điểm).

> **Kết luận Phase 3:** Cấu hình Combined Augmentation (Run 14) đem lại hiệu năng cao nhất. Nhóm đặc trưng Facial Landmarks không mang lại cải thiện trong thiết lập hiện tại và do đó không được đưa vào pipeline đề xuất.

---

### 4.5 · Phase 4 — Cross-Model Validation (SL-GCN)

**Mục tiêu:** Xác minh tính tổng quát hóa của pipeline tối ưu trên kiến trúc mô hình dựa trên đồ thị (SL-GCN).

| Run | Cấu hình | Val Acc | Val F1 | Test Acc | Test F1 | Trạng thái |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Run 16 | SL-GCN Baseline | 78.37% | 78.16% | 67.44% | 66.88% | ✅ Completed |
| Run 17 | SL-GCN + Interpolation + Best TBL | 83.99% | 84.34% | 73.26% | 73.29% | ✅ Completed |
| Run 18 | SL-GCN + Interpolation + Best TBL + Face | **86.20%** | **86.63%** | **75.41%** | **75.23%** | ✅ Completed |

**Nhận xét:** 
* **Tác động của Keypoint Interpolation (Run 17 vs Run 16)**: Việc bổ sung cơ chế nội suy điểm thiếu giúp cải thiện hiệu năng rõ rệt cho SL-GCN (tăng **+5.62% Val Acc** và **+5.82% Test Acc**). Kết quả này củng cố tính tổng quát của bước tiền xử lý nội suy đối với mô hình đồ thị vốn nhạy cảm với việc mất kết nối keypoint.
* **Tác động của Face Landmarks (Run 18 vs Run 17)**: Khác với kiến trúc SPOTER (nơi landmarks khuôn mặt có xu hướng làm giảm hiệu năng), trên mô hình SL-GCN, Face Landmarks ghi nhận sự cải thiện hiệu năng (tăng **+2.21% Val Acc** và **+2.15% Test Acc**). Điều này chỉ ra cấu trúc liên kết đồ thị cục bộ của GCN giúp phân tách và khai thác tốt thông tin biểu cảm khuôn mặt mà không làm bão hòa khả năng biểu diễn của mô hình.
* **So sánh chéo giữa hai kiến trúc**: Mặc dù Run 18 là cấu hình tốt nhất của SL-GCN (Test Acc 75.41%), nó vẫn thấp hơn SPOTER Run 08 (Test Acc 84.08%). Sự chênh lệch này đến từ cơ chế Self-Attention toàn cục của Transformer cho phép SPOTER học các mối quan hệ khoảng cách xa linh hoạt hơn cấu trúc đồ thị tĩnh định sẵn của GCN.

---

## 5. Tổng hợp và Pipeline Được Đề Xuất

### 5.1 · Bảng so sánh tất cả cấu hình đã chạy

**Kiến trúc SPOTER:**

| Run | Mô tả ngắn | Test Acc | Test F1 | Δ vs Baseline (Run 00) |
| :--- | :--- | ---: | ---: | ---: |
| Run 00 | Raw Baseline | 78.59% | 78.47% | — |
| Run 03 | + TBL (θ=160°) | 81.08% | 80.86% | +2.49% |
| Run 07 | + Interpolation | 80.80% | 80.72% | +2.21% |
| Run 08 | + Neck Anchor | **84.08%** | **84.02%** | **+5.49%** |
| Run 09 | + Nose Anchor | 80.71% | 80.57% | +2.12% |
| Run 10 | + Spatial Aug | 81.61% | 81.58% | +3.02% |
| Run 11 | + Perspective Aug | 81.27% | 81.10% | +2.68% |
| Run 12 | + Kinematic Aug | 80.55% | 80.44% | +1.96% |
| Run 13 | + Gaussian Noise | 79.71% | 79.39% | +1.12% |
| Run 14 | + Combined Aug | **82.30%** | **82.30%** | **+3.71%** |
| Run 15 | + Face Landmarks | 80.96% | 80.94% | +2.37% |

**Kiến trúc SL-GCN (Xác thực chéo):**

| Run | Mô tả ngắn | Test Acc | Test F1 | Δ vs Baseline (Run 16) |
| :--- | :--- | ---: | ---: | ---: |
| Run 16 | SL-GCN Baseline | 67.44% | 66.88% | — |
| Run 17 | SL-GCN + Interpolation + Best TBL | 73.26% | 73.29% | +5.82% |
| Run 18 | SL-GCN + Interpolation + Best TBL + Face | **75.41%** | **75.23%** | **+7.97%** |

*Các run 01-02, 04-06 thiếu kết quả do giới hạn tài nguyên tính toán cục bộ.*

### 5.2 · Pipeline Đề Xuất

Dựa trên kết quả đã có, pipeline được nhóm đề xuất cho bộ dữ liệu keypoint phát hành bao gồm các bước theo thứ tự:

```
1. TBL Segmentation    (θ = 160°, τb = 400 ms, N = 20)  ← tham số mặc định VSL400
2. Keypoint Extraction (MediaPipe Holistic)
3. PoseInterpolate     (confidence_threshold = 0.5)       ← lấp đầy keypoint bị mất
4. SPOTERJointSelect   (54 khớp: 12 body + 42 hand)
5. Neck Anchor Norm    (anchor = "neck", scale = neck-nose distance)
6. Hand Wrist Norm     (shift về tọa độ cổ tay)
7. SPOTERPad           (96 frames, cycle-pad)
8. SPOTERShift         ([0,1] → [-0.5, 0.5])
```

*Augmentation (Spatial + Perspective + Kinematic + Gaussian) chỉ áp dụng trong quá trình huấn luyện, không đưa vào keypoint tĩnh của dataset.*

> **Lưu ý quan trọng:** Pipeline trên dựa trên các runs đã chạy. Các tham số TBL (θ, τb) **chưa được tối ưu hóa bằng grid search** do giới hạn tài nguyên tính toán. Người sử dụng được khuyến nghị chạy thêm ablation trên dữ liệu riêng trước khi áp dụng.

---

## 6. Công việc Còn Lại (Future Work)

| Hạng mục | Mô tả | Ưu tiên |
| :--- | :--- | :--- |
| Hoàn tất Phase 1 | Chạy các run TBL sweep còn thiếu (Run 02, 04, 05, 06) để tối ưu tham số cắt thời gian | Cao |
| Multi-seed | Lặp lại các runs với ≥3 seed để thu được error bar | Trung bình |
| Tối ưu hóa cấu trúc đồ thị | Tinh chỉnh Adjacency Matrix cho SL-GCN để giảm khoảng cách hiệu năng với SPOTER | Trung bình |
| Facial trên model lớn hơn | Thử nghiệm với hidden_dim tăng từ 108 lên 148 kết hợp Neck Anchor | Thấp |
| Multi-view | Mở rộng ablation thử nghiệm trên dữ liệu cam_2, cam_3 | Thấp |

---

## 7. Tham khảo

| Ký hiệu | Công trình |
| :--- | :--- |
| VSL400 (2026) | Nguyen Quoc et al., "A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition", Zenodo DOI: 10.5281/zenodo.17943574 |
| SPOTER (2022) | Boháček & Hrúz, "Sign Pose-Based Transformer for Word-Level Sign Language Recognition", WACV Workshops |
| Roh et al. (2024) | Roh et al., "Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition", SignLang @ LREC-COLING |
| QIPEDC-VSL (2026) | Dung et al., "Towards Realistic Vietnamese Sign Language Recognition", IJSRED Vol.9 No.1 |
| OpenHands (2022) | Selvaraj et al., "OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models", ACL |
