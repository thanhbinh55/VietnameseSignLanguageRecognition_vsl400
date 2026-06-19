# Hướng dẫn Điều hướng Tài nguyên Dự án

Tài liệu này giúp bạn xác định nên đọc gì trước và tìm tài nguyên ở đâu.

---

## Thứ tự đọc đề xuất

### Nếu bạn mới tiếp cận bài toán VSL

```
[1] vsl_linguistics.md
    └─ Hiểu VSL là gì, khác ASL thế nào, tại sao bài toán khó

[2] khao_sat_ky_thuat.md
    └─ Dataset nào đang có, model nào đã được thử, framework nào dùng được
    └─ Cách đọc và so sánh kết quả từ các paper khác nhau
```

### Nếu bạn muốn hiểu pipeline của dự án này

```
[3] architecture_analysis.md
    └─ Sơ đồ 8 module từ video thô đến prediction
    └─ Bảng mapping file nguồn ↔ chức năng
    └─ Phân loại: cố định gốc / tùy chỉnh / có thể thử nghiệm

[4] ablation_study_report.md
    └─ Kết quả 11/19 cấu hình đã chạy
    └─ Tại sao chọn Neck Anchor thay vì Box
    └─ Tại sao không đưa Facial Landmarks vào pipeline chính
```

### Nếu bạn muốn dùng ngay

```
[5] README.md (gốc repo)
    └─ Cài đặt → chạy nhanh với .npy có sẵn → hoặc full pipeline từ video

[6] DATASET_CARD.md (trong Drive)
    └─ Schema thư mục dataset
    └─ Cách load .npy và .pose
    └─ Điều khoản sử dụng
```

---

## Bản đồ Tài nguyên

| Tài nguyên | Vị trí | Ghi chú |
| :--- | :--- | :--- |
| Source code pipeline | [GitHub](https://github.com/YOUR_ORG/vsl-keypoint-pipeline) | Repo chính |
| Tài liệu kỹ thuật (docs/) | [GitHub/docs/](https://github.com/YOUR_ORG/vsl-keypoint-pipeline/tree/main/docs) | Cùng repo |
| Keypoint processed (.npy) | [Google Drive](https://drive.google.com/YOUR_LINK) | Dùng ngay để train |
| Keypoint raw (.pose) | [Google Drive](https://drive.google.com/YOUR_LINK) | Cần pose-format để đọc |
| Metadata (json, csv) | [Google Drive](https://drive.google.com/YOUR_LINK) | Schema: xem DATASET_CARD |
| Video curation tools | [Google Drive](https://drive.google.com/YOUR_LINK) | Script tách video nhiều lượt |
| Video gốc | ❌ Không phân phối | Liên hệ QIPEDC trực tiếp |

---

## Quan hệ giữa các tài liệu

```
khao_sat_ky_thuat.md          → nêu lý do chọn SPOTER và SL-GCN làm baseline
        ↓
architecture_analysis.md       → giải thích chi tiết cách 2 mô hình được tích hợp
        ↓
ablation_study_report.md       → chứng minh bằng số liệu cấu hình nào hoạt động tốt hơn
        ↓
DATASET_CARD.md (Drive)        → mô tả keypoint đã xử lý theo cấu hình tốt nhất đó
        ↓
README.md                      → hướng dẫn dùng pipeline để reproduce hoặc extend
```
