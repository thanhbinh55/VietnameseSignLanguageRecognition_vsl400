# KHẢO SÁT CÁC PHƯƠNG PHÁP NHẬN DẠNG NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT

**Báo cáo được cập nhật đến tháng 6/2026**

## Tóm tắt nội dung

Báo cáo này được tổng hợp từ các bài báo khoa học, bộ dữ liệu, kho mã nguồn và tài liệu kỹ thuật công khai liên quan đến bài toán nhận dạng ngôn ngữ ký hiệu. Mục tiêu của tài liệu là cung cấp nguồn tham khảo ban đầu và đề xuất một hướng benchmark phù hợp cho dự án nhận dạng Ngôn ngữ ký hiệu tiếng Việt.

Các số liệu về quy mô dữ liệu, độ chính xác và kết quả thực nghiệm được giữ theo cách báo cáo của từng nguồn. Báo cáo không xem các số liệu này là kết quả thực nghiệm của nhóm thực hiện tài liệu, trừ khi có ghi rõ kết quả thử nghiệm nội bộ. Do khác biệt về ngôn ngữ ký hiệu, bộ dữ liệu, số lớp, số người ký hiệu, cách chia dữ liệu, phương pháp tiền xử lý, mô hình pretrained và chỉ số đánh giá, kết quả từ các công trình khác nhau không nên được so sánh trực tiếp như trên cùng một bảng xếp hạng.

Mọi khuyến nghị triển khai trong tài liệu cần được kiểm chứng lại bằng thực nghiệm trên dữ liệu VSL mà dự án thực sự sử dụng. Mặc dù nhóm thực hiện đã cố gắng đối chiếu thông tin với các nguồn gốc, báo cáo vẫn có thể tồn tại sai sót trong quá trình tổng hợp hoặc diễn giải.

**Từ khóa:** Ngôn ngữ ký hiệu tiếng Việt; Isolated Sign Language Recognition; ISLR; gloss; RGB video; keypoint; pose-based recognition; multi-view; data augmentation; benchmark; VSL400; Multi-VSL.

---

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
  - [1.1. Thuật ngữ sử dụng trong báo cáo](#11-thuật-ngữ-sử-dụng-trong-báo-cáo)
  - [1.2. Nguyên tắc tổng hợp, gắn nhãn và so sánh bằng chứng](#12-nguyên-tắc-tổng-hợp-gắn-nhãn-và-so-sánh-bằng-chứng)
- [2. Phân loại các tài nguyên liên quan](#2-phân-loại-các-tài-nguyên-liên-quan)
  - [2.1. Dataset](#21-dataset)
  - [2.2. Mô hình](#22-mô-hình)
  - [2.3. Framework và công cụ](#23-framework-và-công-cụ)
  - [2.4. Công trình về preprocessing](#24-công-trình-về-preprocessing)
- [3. Khảo sát các hướng biểu diễn dữ liệu](#3-khảo-sát-các-hướng-biểu-diễn-dữ-liệu)
  - [3.1. Video RGB thô](#31-video-rgb-thô)
  - [3.2. Keypoint hoặc pose-based](#32-keypoint-hoặc-pose-based)
  - [3.3. Optical flow và đặc trưng chuyển động](#33-optical-flow-và-đặc-trưng-chuyển-động)
  - [3.4. Multi-view](#34-multi-view)
- [4. Tiền xử lý keypoint](#4-tiền-xử-lý-keypoint)
  - [4.1. Normalization](#41-normalization)
  - [4.2. Keypoint reconstruction](#42-keypoint-reconstruction)
  - [4.3. Smoothing và filtering](#43-smoothing-và-filtering)
  - [4.4. Facial landmarks và non-manual markers](#44-facial-landmarks-và-non-manual-markers)
- [5. Data augmentation](#5-data-augmentation)
- [6. Khuyến nghị pipeline benchmark cho dự án VSL](#6-khuyến-nghị-pipeline-benchmark-cho-dự-án-vsl)
  - [6.1. Giai đoạn 1 - Kiểm tra dữ liệu](#61-giai-đoạn-1---kiểm-tra-dữ-liệu)
  - [6.2. Giai đoạn 2 - Chia dữ liệu](#62-giai-đoạn-2---chia-dữ-liệu)
  - [6.3. Giai đoạn 3 - Baseline tối thiểu](#63-giai-đoạn-3---baseline-tối-thiểu)
  - [6.4. Ablation cho keypoint](#64-ablation-cho-keypoint)
  - [6.5. Multi-view benchmark](#65-multi-view-benchmark)
  - [6.6. Chỉ số đánh giá](#66-chỉ-số-đánh-giá)
- [7. Cách trình bày kết quả](#7-cách-trình-bày-kết-quả)
- [8. Khuyến nghị cho dự án](#8-khuyến-nghị-cho-dự-án)
- [9. Kết luận](#9-kết-luận)
- [10. Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

# 1. Giới thiệu

Ngôn ngữ ký hiệu là một hệ thống ngôn ngữ thị giác, trong đó thông tin được truyền đạt thông qua hình dạng bàn tay, vị trí tay, hướng lòng bàn tay, chuyển động, tư thế cơ thể và các tín hiệu phi thủ công như biểu cảm khuôn mặt, hướng nhìn hoặc chuyển động đầu.

Bài toán tự động phân tích ngôn ngữ ký hiệu thường gồm nhiều mức độ khác nhau:

- **Isolated Sign Language Recognition - ISLR:** nhận dạng ký hiệu rời rạc, trong đó mỗi video thường chứa một ký hiệu cần phân loại.
- **Continuous Sign Language Recognition - CSLR:** nhận dạng chuỗi ký hiệu liên tục và xác định trình tự các ký hiệu xuất hiện.
- **Sign Language Translation - SLT:** chuyển nội dung ngôn ngữ ký hiệu sang câu trong ngôn ngữ nói hoặc ngôn ngữ viết.

Báo cáo này tập trung chủ yếu vào ISLR ở cấp độ từ, vì đây là hướng phù hợp hơn cho giai đoạn xây dựng baseline khi dữ liệu và tài nguyên tính toán còn hạn chế.

Một ký hiệu trong dữ liệu thường được gắn với một **gloss**. Gloss là nhãn văn bản được sử dụng để đại diện cho một đơn vị ký hiệu trong quá trình chú thích. Gloss hỗ trợ tổ chức dữ liệu và huấn luyện mô hình, nhưng không nên được hiểu đơn giản là một từ tương đương hoàn toàn trong tiếng Việt, vì cấu trúc ngôn ngữ ký hiệu và ngôn ngữ nói có thể khác nhau [7].

Về phương thức biểu diễn đầu vào, các hệ thống ISLR hiện nay thường sử dụng một hoặc nhiều loại dữ liệu sau:

1. Video RGB thô.
2. Keypoint hoặc skeleton.
3. Optical flow hoặc đặc trưng chuyển động.
4. Nhiều góc quay.
5. Kết hợp nhiều phương thức bằng mô hình multi-stream hoặc multimodal.

Không có một phương thức biểu diễn nào tối ưu tuyệt đối. Hiệu quả của mỗi hướng phụ thuộc vào chất lượng dữ liệu, số lượng mẫu trên mỗi lớp, số người ký hiệu, góc quay, độ phân giải, tài nguyên tính toán và yêu cầu triển khai.

## 1.1. Thuật ngữ sử dụng trong báo cáo

**Bảng 1. Thuật ngữ sử dụng trong báo cáo**

| Thuật ngữ | Giải thích |
|---|---|
| Gloss | Nhãn văn bản đại diện cho một ký hiệu trong dữ liệu chú thích. |
| Signer | Người thực hiện ngôn ngữ ký hiệu trong video. |
| ISLR | Nhận dạng từng ký hiệu rời rạc. |
| CSLR | Nhận dạng chuỗi nhiều ký hiệu liên tục. |
| TBL | Một bước tiền xử lý temporal segmentation có vai trò xác định vùng thời gian chứa ký hiệu chính, loại bỏ frame dư và giúp mô hình keypoint-based học tập trung hơn vào chuyển động thực sự mang thông tin ngôn ngữ. |
| Signer-dependent | Người xuất hiện trong tập kiểm tra có thể đã xuất hiện trong tập huấn luyện. |
| Signer-independent | Người trong tập kiểm tra không xuất hiện trong tập huấn luyện. |
| LOSO | Leave-One-Signer-Out: lần lượt giữ lại một signer để kiểm tra và huấn luyện bằng các signer còn lại. |
| Keypoint | Tọa độ của các khớp hoặc điểm mốc trên cơ thể, bàn tay và khuôn mặt. |
| Non-manual markers | Các tín hiệu không xuất phát trực tiếp từ chuyển động tay, chẳng hạn biểu cảm khuôn mặt, miệng, mắt hoặc tư thế đầu. |
| Jitter | Dao động nhỏ và không ổn định của keypoint giữa các frame liên tiếp. |
| Early fusion | Kết hợp dữ liệu hoặc đặc trưng từ nhiều nguồn ở giai đoạn đầu của mô hình. |
| Late fusion | Xử lý từng nguồn bằng các nhánh riêng rồi kết hợp đặc trưng hoặc dự đoán ở giai đoạn cuối. |
| Missing view | Một hoặc nhiều góc camera không có sẵn trong quá trình suy luận. |
| Top-1 accuracy | Tỉ lệ nhãn có xác suất cao nhất trùng với nhãn đúng. |
| Top-5 accuracy | Tỉ lệ nhãn đúng nằm trong năm dự đoán có xác suất cao nhất. |
| Macro F1-score | F1 được tính riêng cho từng lớp rồi lấy trung bình, phù hợp hơn accuracy khi dữ liệu mất cân bằng. |
| Ablation study | Thí nghiệm loại bỏ hoặc bổ sung từng thành phần để đo đóng góp thực sự của thành phần đó. |

## 1.2. Nguyên tắc tổng hợp, gắn nhãn và so sánh bằng chứng

Báo cáo này tổng hợp thông tin từ nhiều nguồn khác nhau, bao gồm bài báo khoa học, bộ dữ liệu, framework, kho mã nguồn và tài liệu kỹ thuật. Do các công trình có thể khác nhau về ngôn ngữ ký hiệu, bộ dữ liệu, số lớp, số người ký hiệu, cách chia tập dữ liệu train/validation/test, mô hình, tiền xử lý, pretrained weights và metric đánh giá, các kết quả được trích dẫn trong báo cáo không nên được hiểu như một bảng xếp hạng trực tiếp.

Để tránh nhầm lẫn giữa kết quả đã được công bố, kết quả do nhóm tự thử nghiệm và khuyến nghị triển khai, báo cáo sử dụng ba mức độ bằng chứng sau:

**Bảng 2. Ba mức độ phát biểu được sử dụng trong báo cáo**

| Ký hiệu | Loại phát biểu | Ý nghĩa | Cách diễn giải |
|---|---|---|---|
| [M1] | Kết quả được công trình công bố | Kết quả, số liệu hoặc nhận định được trích từ paper, dataset, framework hoặc tài liệu kỹ thuật đã công bố. | Chỉ phản ánh kết quả trong thiết lập của công trình gốc, không mặc định đúng cho dữ liệu VSL của dự án. |
| [M2] | Quan sát hoặc thử nghiệm nội bộ | Kết quả do nhóm tự chạy lại, tự kiểm tra hoặc tự đo trên dữ liệu của dự án. | Đi kèm mô tả dữ liệu, số lượng mẫu, split, metric, mô hình, cấu hình chạy và điều kiện đánh giá. |
| [M3] | Khuyến nghị cho dự án | Đề xuất, nhận định hoặc hướng thiết kế thí nghiệm dựa trên tổng hợp tài liệu và nhu cầu của dự án. | Là định hướng cần kiểm chứng, không được xem là kết luận thực nghiệm nếu chưa có benchmark tương ứng. |

Trong các phần sau, những phát biểu [M1] trích kết quả từ công trình trước sẽ được gắn ứng với thứ tự của bài báo đó trong mục **Tài liệu tham khảo**. Những phát biểu dựa trên kết quả nhóm tự chạy lại sẽ được gắn nhãn [M2]. Những đoạn đưa ra nhận định, đề xuất pipeline, đề xuất benchmark hoặc khuyến nghị triển khai sẽ được gắn nhãn [M3].

**Ví dụ phát biểu mức [M1]:**

> Roh và cộng sự báo cáo rằng pipeline anchor-based normalization kết hợp keypoint reconstruction cải thiện accuracy thêm 6,05 điểm phần trăm trong thiết lập của họ trên WLASL [M1].

Phát biểu này chỉ mô tả kết quả của công trình gốc. Nó không có nghĩa rằng cùng mức cải thiện sẽ xuất hiện trên dữ liệu VSL của dự án.

**Ví dụ phát biểu mức [M2]:**

> [M2] Trong thử nghiệm nội bộ trên tập dữ liệu VSL của dự án VSL400, MediaPipe không phát hiện đủ keypoint bàn tay ở một số video có hiện tượng hai tay che nhau.

Loại phát biểu này chỉ nên được đưa vào báo cáo khi có thông tin đi kèm như số lượng video được kiểm tra, tiêu chí xác định mất keypoint, tỉ lệ frame bị mất keypoint và điều kiện chạy thử nghiệm. Nếu chưa có các số liệu này, báo cáo không nên trình bày phát biểu đó như một kết quả thực nghiệm.

**Ví dụ phát biểu mức [M3]:**

> [M3] Dự án nên benchmark keypoint trước và sau normalization, reconstruction và smoothing bằng cùng một mô hình, cùng split và cùng metric.

Đây là đề xuất thiết kế thí nghiệm, không phải kết luận rằng normalization, reconstruction hoặc smoothing chắc chắn sẽ cải thiện kết quả trên mọi tập dữ liệu.

Khi trình bày accuracy hoặc các chỉ số đánh giá, báo cáo chỉ so sánh trực tiếp các mô hình nếu chúng thỏa mãn các điều kiện sau:

- Sử dụng cùng dataset và cùng phiên bản dữ liệu.
- Có cùng số lớp hoặc cùng tập gloss được đánh giá.
- Sử dụng cùng train/validation/test split.
- Có cùng protocol signer-dependent hoặc signer-independent.
- Sử dụng cùng metric đánh giá.
- Điều kiện pretrained tương đương hoặc được mô tả rõ.
- Quy trình preprocessing và augmentation được mô tả rõ.
- Số seed, số epoch và cấu hình huấn luyện được báo cáo đầy đủ nếu là kết quả thực nghiệm nội bộ.

Nếu các điều kiện trên không được thỏa mãn, kết quả chỉ nên được sử dụng để tham khảo xu hướng, không nên dùng để kết luận mô hình này tốt hơn mô hình kia.

Trong phạm vi báo cáo hiện tại, phần lớn các số liệu từ paper và dataset được xem là [M1]. Các phần **Nhận định cho dự án**, **Khuyến nghị pipeline** hoặc **Cần benchmark thêm** được xem là [M3]. Kết quả báo cáo thử nghiệm nội bộ được mô tả đầy đủ là [M2].

---

# 2. Phân loại các tài nguyên liên quan

## 2.1. Dataset

**Bảng 3. Các bộ dữ liệu liên quan đến nhận dạng ngôn ngữ ký hiệu**

| Dataset | Ngôn ngữ | Quy mô chính | Góc quay | Vai trò tham khảo |
|---|---:|---|---|---|
| WLASL [1] | ASL | 2.000 gloss, 21.083 video, 119 signer | Chủ yếu single-view | Benchmark RGB và pose trên nhiều kích thước từ vựng. |
| MS-ASL [2] | ASL | 1.000 gloss, 25.513 video, hơn 200 signer | Single-view | Nhấn mạnh đánh giá trên signer chưa xuất hiện trong train. |
| ASL Citizen [3] | ASL | 2.731 sign, 83.399 video, 52 signer | Webcam, môi trường đa dạng | Tham khảo cách thu thập crowdsourced và dictionary retrieval. |
| Multi-VSL [7] | VSL | 1.000 gloss, 84.764 video, 30 signer | Ba view | Benchmark single-view, multi-view và missing view. |
| VSL400 [8] | VSL | 400 gloss, 74.259 clip, 28 signer | Ba view đồng bộ | Phù hợp xây dựng và kiểm thử prototype VSL multi-view. |
| QIPEDC-derived VSL [9] | VSL | 3.782 gloss, 6.046 video, 11 signer | Không được nêu rõ | Nghiên cứu từ vựng lớn, mất cân bằng lớp và protocol vocabulary-complete. |

## 2.2. Mô hình

**Bảng 4. Các mô hình thường gặp trong nhận dạng ngôn ngữ ký hiệu**

| Mô hình | Loại đầu vào | Đặc điểm | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| I3D | RGB | 3D convolution học đặc trưng không gian-thời gian. | Mạnh cho video RGB, tận dụng pretrained action recognition. | Nặng hơn keypoint; dễ học nền, quần áo hoặc ánh sáng nếu dữ liệu ít. |
| Video Swin Transformer | RGB | Video Transformer dùng shifted windows. | Mạnh với RGB, học attention không gian-thời gian. | Chi phí cao, cần dữ liệu hoặc pretraining. |
| MViT / MViTv2 | RGB | Transformer đa tỉ lệ cho video. | Hiệu quả với video dài hoặc dữ liệu đa tỉ lệ. | Tốn tài nguyên; cần tuning nhiều. |
| LSTM / BiLSTM | Keypoint hoặc embedding | Baseline tuần tự theo thời gian. | Dễ triển khai, nhẹ, mobile-friendly. | Có thể yếu với tương tác không gian phức tạp giữa các khớp. |
| Transformer Encoder | Keypoint | Projection keypoint, positional encoding và self-attention temporal. | Mobile-compatible, mạnh hơn BiLSTM trong benchmark nội bộ. | Cần đủ epoch; kết quả 1-epoch chỉ phản ánh ranking tương đối. |
| ST-GCN | Skeleton graph | Graph convolution theo topology cơ thể kết hợp temporal modeling. | Khai thác cấu trúc xương rõ ràng. | Graph operations có thể khó tối ưu trên mobile; nhạy với topology và keypoint noise. |
| SL-GCN | Skeleton graph | Graph model điều chỉnh cho dữ liệu ngôn ngữ ký hiệu. | Có sẵn trong OpenHands, phù hợp với hướng pose-based SLR. | Trong thử nghiệm nội bộ, SL-GCN hưởng lợi từ nội suy keypoint và facial landmarks, nhưng hiệu năng tổng thể hiện vẫn thấp hơn SPOTER. |
| SPOTER [5] | Pose 2D | Transformer encoder-decoder cho nhận dạng ký hiệu cấp từ. | Chi phí thấp, có normalization và augmentation riêng cho pose. | Phụ thuộc chất lượng pose; cần tuning keypoint subset. |
| MMPose + Transformer | Keypoint | Tách pose estimation và sequence modeling. | MMPose/RTMPose có COCO-WholeBody schema; dễ thay extractor. | Lỗi extractor ảnh hưởng trực tiếp đến mô hình; tọa độ 2D có thể mất thông tin chiều sâu. |
| VTNPF | RGB kết hợp pose flow | Baseline multi-source trong Multi-VSL. | Kết hợp appearance và pose/motion. | Pipeline phức tạp, khó triển khai nhẹ. |

**Ghi chú về SL-GCN.** Trong thử nghiệm nội bộ, SL-GCN hưởng lợi từ việc nội suy keypoint, với Test Accuracy tăng 5,82 điểm phần trăm và đạt 73,26%. Khi kết hợp thêm facial landmarks, mô hình tăng thêm 2,15 điểm phần trăm và đạt 75,41%. Điều này cho thấy cấu trúc đồ thị cục bộ của GCN có thể khai thác tốt biểu cảm khuôn mặt. Tuy nhiên, hiệu năng tổng thể hiện tại vẫn thấp hơn kiến trúc SPOTER, với Test Accuracy 84,08%.

## 2.3. Framework và công cụ

**Bảng 5. Framework và công cụ hỗ trợ nhận dạng ngôn ngữ ký hiệu**

| Framework/công cụ | Chức năng |
|---|---|
| OpenHands [4] | Chuẩn hóa pose dataset, cung cấp baseline, checkpoint và pipeline cho ISLR. |
| MediaPipe | Trích xuất pose, hand và face landmarks. |
| OpenPose | Trích xuất keypoint cơ thể và bàn tay. |
| MMPose | Toolbox pose estimation với nhiều backbone và cấu hình. |
| PyTorch/TensorFlow | Huấn luyện và triển khai mô hình. |
| Repository của dataset | Cung cấp script tải dữ liệu, metadata, split hoặc baseline. Repository không đồng nghĩa với dataset hay paper. |

**Lưu ý.** OpenHands là framework/library, không phải một mô hình duy nhất. Trong OpenHands có nhiều mô hình như LSTM, Transformer, ST-GCN và SL-GCN [4].

## 2.4. Công trình về preprocessing

Bảng dưới đây không nhằm liệt kê toàn bộ các công trình preprocessing trong nhận dạng ngôn ngữ ký hiệu. Báo cáo chỉ chọn bốn nhóm công trình có liên hệ trực tiếp nhất với hướng dự án: pose normalization/augmentation, keypoint reconstruction/anchor, framework pose-based và ablation VSL cho hand/pose/face/augmentation.

**Bảng 6. Các công trình liên quan đến preprocessing trong nhận dạng ngôn ngữ ký hiệu**

| Công trình | Đóng góp chính | Ưu điểm khi tham khảo | Hạn chế khi chuyển sang VSL | Kết quả cần lưu ý |
|---|---|---|---|---|
| SPOTER [5] | Pose normalization và augmentation cho Transformer. | Gần với pipeline keypoint nhẹ; có ablation normalization. | Kết quả trên WLASL/LSA64, không mặc định chuyển sang VSL. | Normalization tăng hơn 14 điểm trong ablation WLASL100 của công trình. |
| Roh et al. [6] | Anchor-based normalization và reconstruction cho MediaPipe keypoints. | Trực tiếp liên quan đến missing hand và anchor normalization. | Cần kiểm tra lại với VSL, extractor và split của dự án. | Pipeline cải thiện 6,05 điểm và đạt 83,26% trong protocol của công trình. |
| OpenHands [4] | Chuẩn hóa pose representation trên nhiều dataset/ngôn ngữ. | Tốt để tổ chức baseline và so sánh model pose-based. | Không thay thế ablation trên dataset VSL thật. | Cung cấp pipeline và baseline thống nhất. |
| QIPEDC-derived VSL study [9] | Ablation hand, pose, face và augmentation trên VSL. | Gần ngôn ngữ mục tiêu, có vocabulary lớn và protocol rõ. | Dataset mất cân bằng mạnh; không nên kỳ vọng cùng mức tăng trên VSL400. | Facial landmarks tăng 3,67 điểm; augmentation tăng từ 34,52% lên 57,83% trong thiết lập được báo cáo. |

---

# 3. Khảo sát các hướng biểu diễn dữ liệu

## 3.1. Video RGB thô

Video RGB thô giữ lại gần như toàn bộ thông tin quan sát được trong mỗi frame, bao gồm:

- Hình dạng và độ xoay của bàn tay.
- Chi tiết các ngón tay.
- Vị trí tay so với cơ thể.
- Biểu cảm khuôn mặt.
- Chuyển động đầu và thân người.
- Thông tin về vật thể hoặc ngữ cảnh xung quanh.

Pipeline RGB thường gồm chuẩn hóa định dạng video, lấy mẫu frame, resize hoặc crop vùng signer, chuẩn hóa pixel và đưa chuỗi frame vào mô hình.

Các kiến trúc thường gặp gồm:

- CNN kết hợp LSTM hoặc GRU.
- 3D CNN như I3D.
- Video Transformer như Video Swin Transformer hoặc MViT.

Trong WLASL, I3D sử dụng RGB đạt Top-1 lần lượt là 65,89%, 56,14%, 47,33% và 32,48% trên các subset WLASL100, WLASL300, WLASL1000 và WLASL2000. Trong cùng protocol, Pose-TGCN đạt lần lượt 55,43%, 38,32%, 34,86% và 23,65% [1]. Kết quả này là bằng chứng cho thấy RGB có thể vượt pose trong thiết lập cụ thể của WLASL, nhưng không đủ để kết luận rằng RGB luôn tốt hơn pose trên mọi dataset.

Cần lưu ý rằng I3D trong thí nghiệm WLASL sử dụng trọng số pretrained từ ImageNet và Kinetics, trong khi pipeline pose phụ thuộc vào một pose estimator có sẵn. Vì vậy, chênh lệch không chỉ đến từ dữ liệu RGB hay keypoint mà còn liên quan đến pretraining, capacity của mô hình và lỗi tích lũy từ quá trình trích xuất keypoint [1].

Trong công trình VSL sử dụng dữ liệu QIPEDC, các tác giả báo cáo Video Swin Transformer đạt 58,92%, I3D đạt 54,73% và MMPose kết hợp Transformer đạt 56,41% trên protocol vocabulary-complete gồm 3.782 lớp [9]. Các kết quả này được tạo ra trong cùng công trình nên có thể dùng để tham khảo tương đối, nhưng vẫn cần tái lập trước khi sử dụng làm baseline chính thức.

**Ưu điểm**

- Không làm mất trực tiếp thông tin về texture, bàn tay và khuôn mặt.
- Có thể học đặc trưng end-to-end.
- Tận dụng được pretrained model từ video action recognition.
- Phù hợp khi dữ liệu đủ lớn và GPU đáp ứng được yêu cầu.

**Hạn chế**

- Tốn dung lượng lưu trữ và VRAM.
- Thời gian huấn luyện và suy luận cao.
- Có nguy cơ học các yếu tố không liên quan như nền, quần áo hoặc ánh sáng.
- Dễ overfit nếu số mẫu trên mỗi gloss thấp.
- Khó giải thích chính xác mô hình đang dựa vào tay, khuôn mặt hay background.

### [M2] Thử nghiệm nội bộ với biểu diễn keypoint trên VSL400

Trong phạm vi thử nghiệm nội bộ, nhóm đã đánh giá hướng biểu diễn keypoint trên tập VSL400, sử dụng subset **cam_1** - góc máy quay từ phía trước mặt người ký hiệu (chính diện), gồm 400 gloss và 28 signer, với protocol signer-disjoint split và seed = 42. Mô hình chính trong các phase đã hoàn tất là SPOTER, sử dụng Top-1 Accuracy và Macro F1 làm metric đánh giá.

Kết quả Raw Baseline, tức cấu hình không dùng TBL, không nội suy keypoint và không augmentation, đạt **78,59% Test Accuracy** và **78,47% Test Macro F1**. Khi bổ sung các bước tiền xử lý keypoint, cấu hình tốt nhất trong các run đã hoàn tất là **TBL + Keypoint Interpolation + Neck Anchor Normalization**, đạt **84,08% Test Accuracy** và **84,02% Test Macro F1**. Đây là mức cao nhất đạt được, vượt qua cả các cấu hình có áp dụng thêm augmentation tổng hợp, vốn chỉ đạt 82,30% Test Accuracy do thực hiện trên Box Anchor.

Kết quả này cho thấy hướng keypoint-based có tiềm năng làm baseline thực nghiệm cho VSL400 trong thiết lập cam_1. Tuy nhiên, kết quả này chưa chứng minh keypoint-based tốt hơn RGB, vì nhóm chưa chạy RGB baseline trên cùng split. Do đó, trong báo cáo này, kết quả trên chỉ được xem là quan sát nội bộ ở mức [M2] cho hướng keypoint-based, còn so sánh RGB-keypoint vẫn cần được thực hiện trong các benchmark tiếp theo.

> [M3] Trong phạm vi thực nghiệm nội bộ hiện tại, nhóm chưa chạy RGB baseline trên VSL400 cam_1, nên chưa có cơ sở kết luận RGB tốt hơn hay kém hơn keypoint trên dữ liệu của dự án.

## 3.2. Keypoint hoặc pose-based

Hướng pose-based chuyển mỗi frame thành tập tọa độ biểu diễn bàn tay, thân người và khuôn mặt. Một video khi đó trở thành chuỗi tensor có dạng gần với:

```text
T x K x D
```

Trong đó:

- `T` là số frame.
- `K` là số keypoint.
- `D` là số chiều tọa độ, thường gồm `(x, y)`, đôi khi có thêm `(z)` và confidence.

Các mô hình có thể xử lý dữ liệu này gồm:

- LSTM hoặc BiLSTM.
- Transformer encoder.
- ST-GCN hoặc các graph neural network.
- SPOTER.
- Các kiến trúc fusion giữa pose, hand và face.

OpenHands sử dụng pose như một phương thức chuẩn hóa cho các bài toán ISLR tài nguyên thấp. Framework này cung cấp các phiên bản pose của nhiều dataset, bốn nhóm mô hình gồm LSTM, Transformer, ST-GCN và SL-GCN, đồng thời phát hành checkpoint trên sáu ngôn ngữ ký hiệu [4].

Trong WLASL2000, OpenHands báo cáo SL-GCN đạt 30,6% accuracy với pipeline chuẩn hóa của framework, cao hơn kết quả Pose-TGCN 23,65% được dẫn lại từ công trình WLASL. Tuy nhiên, cần kiểm tra chi tiết cách xử lý và protocol trước khi xem đây là một phép so sánh hoàn toàn tương đương [4].

**Ưu điểm**

1. Kích thước đầu vào nhỏ hơn nhiều so với video RGB. Thay vì xử lý toàn bộ pixel của mỗi frame, mô hình chỉ xử lý tọa độ của các điểm mốc quan trọng. Điều này giúp giảm dung lượng dữ liệu, giảm chi phí huấn luyện và thuận lợi hơn cho triển khai thời gian thực.
2. Keypoint giúp giảm ảnh hưởng của các yếu tố không liên quan như nền, màu quần áo, ánh sáng hoặc vật thể xung quanh. Vì dữ liệu đầu vào chủ yếu là tọa độ hình học, mô hình có xu hướng tập trung hơn vào chuyển động và tư thế của người ký hiệu.
3. Biểu diễn keypoint có cấu trúc rõ ràng. Các điểm mốc trên bàn tay, cánh tay, vai và khuôn mặt có quan hệ hình học với nhau, nên phù hợp với các mô hình graph như ST-GCN hoặc SL-GCN.
4. Hướng này dễ phân tích lỗi hơn so với RGB. Nhóm có thể đo được tỉ lệ mất keypoint, jitter, độ tin cậy trung bình của hand landmarks, hoặc những frame mà công cụ trích xuất pose không phát hiện được bàn tay.

**Hạn chế**

1. Hạn chế lớn nhất của keypoint-based là phụ thuộc mạnh vào pose estimator, tức công cụ trích xuất điểm mốc. Nếu MediaPipe, OpenPose hoặc MMPose phát hiện sai bàn tay, mất ngón tay, nhầm tay trái/tay phải hoặc không phát hiện được bàn tay trong một số frame, mô hình nhận dạng phía sau sẽ nhận đầu vào sai.
2. Keypoint có thể làm mất một phần thông tin quan trọng của ký hiệu. Ví dụ, RGB có thể giữ lại texture, hình dạng thật của bàn tay, độ cong ngón tay, hướng lòng bàn tay hoặc chi tiết khuôn mặt. Trong khi đó, keypoint chỉ giữ lại tọa độ rời rạc nên có thể không đủ để phân biệt những gloss có handshape gần giống nhau.
3. Chuyển động nhanh, motion blur, tay bị che khuất hoặc bàn tay quá nhỏ trong khung hình có thể làm keypoint bị thiếu hoặc dao động mạnh. Khi lỗi này xảy ra ở bước trích xuất keypoint, mô hình nhận dạng không thể sửa lỗi theo kiểu end-to-end như khi học trực tiếp từ RGB.

Li et al. nhận xét rằng do các mô hình pose-based trong WLASL sử dụng bộ ước lượng tư thế có sẵn như OpenPose, sai số trong quá trình ước lượng pose có thể làm suy giảm hiệu năng nhận dạng ký hiệu; do đó, huấn luyện các mô hình pose-based theo hướng end-to-end có thể tiếp tục cải thiện kết quả [1]. Roh và cộng sự cũng mô tả trường hợp MediaPipe không phát hiện được bàn tay ở một số frame, khiến mô hình nhầm các ký hiệu có chuyển động cơ thể tương tự nhưng khác handshape [6].

**Nhận định cho dự án.** Pose-based phù hợp để xây dựng baseline đầu tiên khi tài nguyên hạn chế. Tuy nhiên, dự án không nên xem file keypoint đã trích xuất là dữ liệu hoàn chỉnh. Cần đo riêng chất lượng trích xuất và thực hiện ablation cho normalization, reconstruction, smoothing và face landmarks.

> [M3] Từ các kết quả đã công bố, keypoint-based nên được đưa vào như một baseline quan trọng cho dự án VSL, đặc biệt trong giai đoạn xây dựng prototype hoặc benchmark ban đầu. Trong phạm vi dự án, nhóm không khẳng định keypoint-based tốt hơn RGB.

Dự án nên kiểm chứng hướng keypoint-based bằng một benchmark có kiểm soát. Trước tiên, cần chạy một baseline keypoint đơn giản, ví dụ BiLSTM hoặc Transformer encoder, sử dụng hand landmarks và upper-body pose. Sau đó, dự án có thể bổ sung dần các bước preprocessing như normalization, missing-keypoint reconstruction, confidence masking và smoothing. Tất cả các biến thể cần được so sánh trên cùng dataset, cùng split, cùng metric và cùng mô hình nhận dạng để đo đúng tác động của từng bước xử lý.

## 3.3. Optical flow và đặc trưng chuyển động

Optical flow biểu diễn sự dịch chuyển của pixel giữa các frame liên tiếp. Hướng này có thể bổ sung thông tin về:

- Hướng chuyển động.
- Tốc độ tương đối.
- Quỹ đạo tay.
- Những khác biệt động học khó quan sát từ một frame đơn lẻ.

Optical flow thường được sử dụng như một stream bổ sung thay vì thay thế hoàn toàn RGB hoặc keypoint. Một hệ thống có thể gồm:

- Stream RGB.
- Stream optical flow.
- Stream keypoint.
- Một module fusion kết hợp các đặc trưng.

Optical flow có chi phí tiền xử lý tương đối cao và có thể bị ảnh hưởng bởi chuyển động camera, background động hoặc motion blur. Vì báo cáo hiện chưa có bằng chứng trực tiếp trên cùng dữ liệu VSL cho thấy optical flow cải thiện bao nhiêu, hướng này nên được xem là một baseline mở rộng sau khi RGB và pose đã được thiết lập ổn định.

## 3.4. Multi-view

Multi-view sử dụng các video đồng bộ từ nhiều góc camera. Hướng này có thể hỗ trợ trong các trường hợp:

- Hai tay che nhau ở góc chính diện.
- Chuyển động diễn ra theo chiều sâu.
- Hướng lòng bàn tay khó quan sát từ một camera.
- Hai gloss có chuyển động tương tự khi nhìn từ trước nhưng khác nhau khi nhìn từ bên.

Ba cách kết hợp phổ biến gồm:

1. **Early fusion:** kết hợp dữ liệu hoặc đặc trưng thấp từ nhiều view trước khi đưa vào phần chính của mô hình.
2. **Late fusion:** mỗi view có encoder riêng, sau đó kết hợp embedding hoặc xác suất dự đoán.
3. **Cross-view learning:** dùng nhiều view trong huấn luyện nhằm cải thiện biểu diễn, kể cả khi suy luận chỉ có một số view.

Multi-VSL gồm 1.000 gloss, 84.764 video và 30 signer với nhiều góc nhìn [7]. Trong thử nghiệm Multi-VSL1000:

**Bảng 7. Kết quả so sánh single-view và multi-view trên Multi-VSL1000**

| Mô hình | Top-1 một view | Top-1 ba view | Chênh lệch |
|---|---:|---:|---:|
| I3D | 40,60% | 60,35% | +19,75 điểm |
| Video Swin Transformer | 76,43% | 76,73% | +0,30 điểm |
| MViTv2 | 81,52% | 84,71% | +3,19 điểm |
| VTNPF | 66,21% | 76,32% | +10,11 điểm |

Kết quả cho thấy ba view cải thiện Top-1 đối với tất cả các baseline được báo cáo, nhưng mức cải thiện khác nhau đáng kể giữa các mô hình [7]. Do đó, kết luận phù hợp không phải là "multi-view luôn cải thiện khoảng 20%", mà là:

> Trong protocol của Multi-VSL, multi-view cải thiện kết quả của tất cả baseline được thử nghiệm; mức tăng lớn nhất là 19,75 điểm phần trăm với I3D, trong khi Video Swin Transformer chỉ tăng 0,30 điểm [7].

Multi-VSL cũng đánh giá missing view. Với I3D trên Multi-VSL1000, mô hình một view đạt 40,60% Top-1. Mô hình ba view được huấn luyện với dữ liệu có view bị thiếu đạt 57,04% khi tỉ lệ thiếu view là 50%, cao hơn 16,44 điểm so với baseline một view trong bảng kết quả [7]. Đây là bằng chứng cho thấy huấn luyện có mô phỏng missing view có thể tăng độ bền trong thiết lập của công trình này.

> [M3] Multi-view chỉ nên được ưu tiên khi dữ liệu thực sự có các camera được đồng bộ. Nếu sản phẩm cuối chỉ sử dụng webcam một góc, dự án cần báo cáo rõ kết quả khi train và test một view, train và test nhiều view, train nhiều view nhưng test thiếu view, cũng như chi phí tăng thêm của nhiều encoder và nhiều camera.

---

# 4. Tiền xử lý keypoint

## 4.1. Normalization

Nếu sử dụng tọa độ ảnh gốc, mô hình có thể học vị trí tuyệt đối của signer, khoảng cách tới camera hoặc kích thước cơ thể. Normalization nhằm giảm các biến thiên không liên quan nhưng vẫn giữ thông tin ngôn ngữ cần thiết.

Các cách thường được sử dụng gồm:

- Đưa gốc tọa độ về vai, cổ, hông hoặc cổ tay.
- Scale skeleton theo khoảng cách hai vai.
- Chuẩn hóa bàn tay trong hệ tọa độ cục bộ.
- Sử dụng vector xương giữa các cặp khớp.
- Tách biểu diễn hình dạng và vị trí bàn tay so với cơ thể.

Các cách trên không phải đều được dùng trong cùng một công trình. Bảng dưới đây tóm tắt một số cách normalization được nhắc đến trong các tài liệu liên quan.

**Bảng 8. Một số cách normalization được sử dụng trong các tài liệu liên quan**

| Nguồn | Cách normalization được sử dụng | Mục tiêu chính |
|---|---|---|
| OpenHands [4] | Chuẩn hóa dữ liệu pose/keypoint để đưa các dataset từ nhiều ngôn ngữ ký hiệu về biểu diễn thống nhất hơn. Framework này hỗ trợ các pipeline pose-based và các mô hình như LSTM, Transformer, ST-GCN và SL-GCN. | Giúp dữ liệu pose từ nhiều nguồn có thể được huấn luyện và benchmark trong cùng một framework. |
| SPOTER [5] | Sử dụng normalization cho pose 2D trước khi đưa vào Transformer. Công trình tập trung vào việc chuẩn hóa không gian ký hiệu và tọa độ bàn tay để giảm ảnh hưởng của vị trí signer, kích thước cơ thể và vùng xuất hiện của bàn tay. | Giúp mô hình tập trung hơn vào quan hệ hình học và chuyển động của keypoint thay vì vị trí tuyệt đối trong ảnh. |
| Roh et al. [6] | Sử dụng anchor-based normalization. Với body keypoints, công trình dùng điểm neo trên cơ thể để đưa skeleton về hệ tọa độ ổn định hơn. Với hand keypoints, công trình chuẩn hóa bàn tay bằng điểm neo cục bộ nhằm giữ lại handshape và giảm ảnh hưởng của vị trí tuyệt đối. | Giảm nhiễu do vị trí người ký và cải thiện biểu diễn handshape trong pipeline MediaPipe keypoints. |
| MediaPipe [10] | MediaPipe chủ yếu là công cụ trích xuất pose, hand và face landmarks. Bản thân MediaPipe không phải là một phương pháp normalization cho ISLR, nhưng thường được dùng để tạo keypoint đầu vào cho các bước normalization phía sau. | Cung cấp keypoint đầu vào cho các pipeline pose-based/keypoint-based. |

SPOTER báo cáo baseline không normalization và augmentation đạt 44,96% trên WLASL100. Khi bổ sung normalization, kết quả tăng hơn 14 điểm phần trăm trong ablation của công trình [5]. Kết quả này cho thấy normalization có ảnh hưởng lớn trong thiết lập SPOTER, nhưng không nên mặc định mức tăng tương tự cho mọi pipeline.

Roh và cộng sự sử dụng các anchor riêng cho cơ thể và bàn tay nhằm loại bỏ thông tin vị trí không cần thiết nhưng giữ hình dạng bàn tay [6]. Công trình báo cáo toàn bộ pipeline preprocessing cải thiện accuracy 6,05 điểm và đạt 83,26% khi kết hợp augmentation trên WLASL-100 trong protocol của họ [6].

Hai kết quả của SPOTER và Roh và cộng sự không nên được so sánh trực tiếp vì có thể khác subset, kiến trúc, cách chia dữ liệu, số keypoint và augmentation.

### [M2] Thực nghiệm nội bộ trên VSL400

Trong thử nghiệm nội bộ của nhóm, normalization được đánh giá trên bộ dữ liệu VSL400, subset cam_1, gồm 400 gloss và 28 signer. Thử nghiệm sử dụng signer-disjoint split, seed = 42, mô hình SPOTER, metric chính là Top-1 Accuracy và Macro F1 trên validation/test set.

**Bảng 9. Kết quả thực nghiệm các cấu hình normalization trên VSL400**

| Run | Cấu hình | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|
| Run 00 | Raw Baseline, không TBL, không nội suy, không augmentation | 78,59% | 78,47% |
| Run 03 | TBL + Box Anchor, không nội suy | 81,08% | 80,86% |
| Run 07 | TBL + Keypoint Interpolation + Box Anchor | 80,80% | 80,72% |
| Run 08 | TBL + Keypoint Interpolation + Neck Anchor | 84,08% | 84,02% |
| Run 09 | TBL + Keypoint Interpolation + Nose Anchor | 80,71% | 80,57% |

Trong các cấu hình đã hoàn tất, Neck Anchor Normalization kết hợp Keypoint Interpolation đạt kết quả cao nhất, với **84,08% Test Accuracy** và **84,02% Test Macro F1**. Cấu hình này tăng 3,00 điểm phần trăm so với baseline TBL dùng Box Anchor ở Run 03 và tăng 5,49 điểm phần trăm so với Raw Baseline ở Run 00.

Ngược lại, Nose Anchor không cải thiện trong thiết lập này, khi chỉ đạt 80,71% Test Accuracy. Keypoint Interpolation khi kết hợp với Box Anchor cũng không tạo ra cải thiện rõ rệt, vì Run 07 đạt 80,80% Test Accuracy, thấp hơn nhẹ so với Run 03.

Kết quả này gợi ý rằng việc chọn điểm neo có ảnh hưởng đáng kể đến chất lượng biểu diễn keypoint. Trong thiết lập VSL400 cam_1 với SPOTER, cổ hoặc vùng neck là điểm neo phù hợp hơn so với mũi và Box Anchor. Một lý do có thể là neck ổn định hơn trong không gian ký hiệu, ít bị che khuất hơn mũi và vẫn giữ được quan hệ tương đối giữa tay, thân trên và vùng ký hiệu.

## 4.2. Keypoint reconstruction

Keypoint reconstruction nhằm khôi phục các điểm không được pose estimator phát hiện. Một cách đơn giản là nội suy giữa các frame trước và sau.

Roh và cộng sự sử dụng bilinear interpolation để tái tạo keypoint bàn tay bị thiếu. Case study của công trình cho thấy reconstruction có thể sửa dự đoán ở những video mà một số frame không phát hiện được bàn tay [6].

> [M2] Ở thực nghiệm được trình bày ở Bảng 9, việc thêm Keypoint Interpolation trong khi vẫn dùng Box Anchor không tạo ra cải thiện rõ rệt. Test Accuracy giảm nhẹ từ 81,08% xuống 80,80%. Điều này cho thấy interpolation đơn lẻ chưa đủ để cải thiện hiệu quả nhận dạng trong thiết lập này.

Ngược lại, khi kết hợp Keypoint Interpolation với Neck Anchor Normalization, kết quả tăng rõ rệt. Run 08 đạt 84,08% Test Accuracy và 84,02% Test Macro F1, cao hơn Run 03 lần lượt 3,00 điểm phần trăm về Test Accuracy và 3,16 điểm phần trăm về Test Macro F1. Đây là cấu hình tốt nhất trong các run đã hoàn tất liên quan đến preprocessing keypoint.

> [M3] Đối với dự án VSL, keypoint reconstruction nên được xem là một biến số cần benchmark thay vì một bước mặc định.

Các chỉ số cần báo cáo gồm:

- Tỉ lệ frame mất keypoint bàn tay.
- Tỉ lệ frame mất toàn bộ một bàn tay.
- Tỉ lệ frame mất cả hai bàn tay.
- Độ dài trung bình của các đoạn mất keypoint liên tục.
- Confidence trung bình của hand keypoints.
- Tỉ lệ keypoint được nội suy.
- Kết quả nhận dạng trước và sau interpolation.

## 4.3. Smoothing và filtering

Trong các hệ thống nhận dạng ngôn ngữ ký hiệu dựa trên keypoint, một vấn đề thường gặp là tọa độ điểm mốc có thể dao động không ổn định giữa các frame liên tiếp. Hiện tượng này thường được gọi là **jitter**. Jitter có thể xuất hiện ngay cả khi chuyển động thật của người ký không thay đổi quá nhanh, do sai số của pose estimator, motion blur, tay bị che khuất, độ phân giải thấp hoặc confidence của keypoint không ổn định.

Smoothing và filtering là nhóm kỹ thuật làm mượt chuỗi keypoint theo thời gian nhằm giảm dao động nhiễu. Mục tiêu của bước này là giúp mô hình nhận dạng quan sát được quỹ đạo chuyển động ổn định hơn, thay vì phải học trên các chuỗi tọa độ bị rung hoặc nhảy đột ngột.

**Bảng 10. Một số bộ lọc làm mượt keypoint thường dùng**

| Bộ lọc | Ý tưởng chính | Khi nào phù hợp |
|---|---|---|
| Moving average | Lấy trung bình keypoint trong một cửa sổ thời gian ngắn, ví dụ 3 hoặc 5 frame. | Baseline đơn giản để giảm jitter nhẹ. |
| Exponential moving average | Làm mượt bằng trung bình có trọng số, trong đó frame gần hiện tại có trọng số lớn hơn. | Khi muốn giảm jitter nhưng vẫn phản ứng nhanh hơn moving average. |
| Savitzky-Golay | Fit đa thức bậc thấp trên cửa sổ lân cận để làm mượt nhưng vẫn giữ một phần hình dạng tín hiệu. | Chuỗi keypoint có chuyển động cong hoặc nhịp nhàng, cần giữ đỉnh và quỹ đạo tương đối. |
| Kalman filter | Ước lượng trạng thái ẩn của keypoint dựa trên mô hình chuyển động và nhiễu đo. | Khi muốn dự đoán hoặc ổn định quỹ đạo trong điều kiện keypoint mất ngắn hạn hoặc nhiễu đo. |
| One Euro Filter | Low-pass filter có cutoff thay đổi theo tốc độ: khi chuyển động chậm thì làm mượt mạnh, khi chuyển động nhanh thì giảm smoothing để hạn chế lag. | Rất phù hợp với tracking tương tác và tín hiệu tọa độ thời gian thực. |

Smoothing quá mạnh có thể làm mất chuyển động nhanh ở ngón tay. Do báo cáo hiện chưa có kết quả VSL chứng minh một bộ lọc cụ thể là tối ưu, smoothing nên được xem là một biến số cần ablation thay vì bước bắt buộc.

## 4.4. Facial landmarks và non-manual markers

Ngôn ngữ ký hiệu không chỉ chứa thông tin ở bàn tay. Biểu cảm khuôn mặt, chuyển động lông mày, miệng, mắt và đầu có thể bổ sung thông tin phân biệt.

Trong công trình [9], các tác giả báo cáo:

**Bảng 11. Kết quả ablation theo nhóm keypoint đầu vào**

| Cấu hình đầu vào | Accuracy |
|---|---:|
| Chỉ bàn tay | 42,18% |
| Bàn tay và pose | 48,67% |
| Bàn tay, pose và toàn bộ facial landmarks | 52,34% |
| Bàn tay, pose và facial landmarks được chọn | 51,89% |

Trong cùng protocol, thêm pose vào hand-only tăng 6,49 điểm; thêm toàn bộ facial landmarks vào hand-plus-pose tăng thêm 3,67 điểm [9].

Kết quả này hỗ trợ việc benchmark face landmarks trên dữ liệu VSL, nhưng không đồng nghĩa rằng sử dụng toàn bộ face mesh luôn tốt nhất. Dự án cần so sánh giữa:

- Không dùng mặt.
- Một nhóm điểm quanh mắt, lông mày và miệng.

Ví dụ, nếu dùng MediaPipe Holistic, mỗi frame có thể xuất ra 543 landmarks gồm 33 pose landmarks, 468 face landmarks và 21 landmarks cho mỗi bàn tay [10], [15]. Tuy nhiên, dùng toàn bộ 468 điểm mặt có thể làm tăng nhiễu và chiều dữ liệu, đặc biệt khi video toàn thân có mặt nhỏ. Vì vậy, có thể chọn một subset đại diện cho các vùng có liên quan ngôn ngữ:

**Bảng 12. Nhóm facial landmarks được chọn để biểu diễn tín hiệu phi thủ công**

| Vùng mặt | Số điểm ví dụ | Ý nghĩa biểu diễn | Ví dụ chỉ số MediaPipe Face Mesh |
|---|---:|---|---|
| Miệng/môi | 10 | Mở/khép miệng, hình miệng, khẩu hình hỗ trợ phân biệt gloss. | 61, 291, 0, 17, 13, 14, 78, 308, 81, 311 |
| Mắt trái | 6 | Độ mở mắt, hướng nhìn cục bộ, nháy/mở mắt. | 33, 133, 159, 145, 160, 144 |
| Mắt phải | 6 | Độ mở mắt, hướng nhìn cục bộ, nháy/mở mắt. | 362, 263, 386, 374, 387, 373 |
| Lông mày trái | 5 | Nhướng/hạ lông mày, biểu cảm nhấn mạnh. | 70, 63, 105, 66, 107 |
| Lông mày phải | 5 | Nhướng/hạ lông mày, biểu cảm nhấn mạnh. | 336, 296, 334, 293, 300 |
| Tổng | 32 | Subset face nhỏ gọn thay cho full face mesh. | - |

Ngoài ra, dự án có thể so sánh thêm:

- Toàn bộ face mesh.
- Embedding khuôn mặt được học bằng một encoder riêng.

Ý tưởng là xây một nhánh riêng cho vùng mặt hoặc keypoint mặt, ví dụ:

1. Nhánh hand/body xử lý keypoint tay và pose.
2. Nhánh face xử lý crop mặt RGB hoặc face landmarks theo thời gian.
3. Mỗi nhánh tạo một embedding riêng.
4. Hai embedding được fusion ở cuối bằng concat, attention hoặc gating.

Lợi ích là mô hình không ép face landmarks vào cùng không gian với hand/body ngay từ đầu. Hạn chế là cần nhiều dữ liệu hơn, dễ overfit nếu số mẫu trên mỗi gloss thấp, và tăng chi phí huấn luyện/suy luận.

---

# 5. Data augmentation

Augmentation trong ngôn ngữ ký hiệu cần được thiết kế thận trọng vì các phép biến đổi hình ảnh có thể thay đổi ý nghĩa của ký hiệu.

Các phép biến đổi có thể xem xét gồm:

- Crop nhẹ.
- Zoom.
- Rotation nhỏ.
- Thay đổi độ sáng hoặc độ tương phản.
- Perspective transformation nhỏ.
- Temporal crop.
- Thay đổi tốc độ ở mức giới hạn.
- Gaussian noise trên keypoint.
- Joint dropout.
- Frame dropping hoặc temporal masking.

Có khá nhiều tranh luận về phép biến đổi horizontal flip so với các phép biến đổi nhỏ như crop, zoom, ... Với ảnh/video thông thường, lật ngang thường được xem là label-preserving. Nhưng trong ngôn ngữ ký hiệu, lật ngang có thể đổi tay thuận, đổi trái/phải, đổi hướng chuyển động hoặc vị trí tay so với cơ thể. Những yếu tố này có thể mang ý nghĩa ngôn ngữ. Vì vậy, không nên đưa ra quy tắc tuyệt đối rằng horizontal flip luôn đúng hoặc luôn sai.

WLASL và Multi-VSL có sử dụng horizontal flipping trong pipeline augmentation của họ [1], [7]. Ngược lại, công trình [9] không sử dụng horizontal hoặc vertical flip vì cho rằng hướng tay và hướng chuyển động có thể mang ý nghĩa ngôn ngữ.

Do đó, khuyến nghị phù hợp là:

> Horizontal flip chỉ nên được sử dụng sau khi người có chuyên môn ngôn ngữ ký hiệu xác nhận phép biến đổi không làm thay đổi nhãn, hoặc sau khi có thí nghiệm ablation riêng cho từng nhóm gloss.

Trong công trình [9], BiLSTM không augmentation đạt 34,52%, trong khi sử dụng đầy đủ năm nhóm augmentation đạt 57,83%, tương ứng mức tăng tuyệt đối 23,31 điểm trong protocol của công trình. Dữ liệu có 64% lớp chỉ chứa một mẫu, vì vậy mức cải thiện lớn này gắn với một thiết lập đặc biệt mất cân bằng và ít mẫu [9]. Không nên dùng con số 23,31 điểm như mức cải thiện kỳ vọng trên các dataset khác.

Nguyên tắc bắt buộc là chia dữ liệu gốc thành train, validation và test trước, sau đó chỉ augmentation tập train. Nếu các phiên bản augmentation của cùng một video xuất hiện ở cả train và test, kết quả sẽ bị data leakage.

---

# 6. Khuyến nghị pipeline benchmark cho dự án VSL

Từ các dataset, framework, mô hình và kết quả thực nghiệm nội bộ, có thể tóm tắt các khoảng trống nghiên cứu quan trọng cho dự án VSL như sau:

1. **Thiếu benchmark VSL có kiểm soát giữa RGB, keypoint và keypoint có preprocessing.** Nhiều kết quả trong ASL hoặc dataset khác không thể chuyển trực tiếp sang VSL vì khác ngôn ngữ, gloss, signer và protocol.
2. **Thiếu ablation VSL cho từng bước tiền xử lý.** Báo cáo nội bộ [I1] cho thấy TBL, neck anchor và augmentation có ảnh hưởng lớn. Việc ứng dụng TBL (Run 03: theta=160 deg, taub=400ms) đã giúp tăng 2,49 điểm Test Acc so với video thô, đạt 81,08%. Tuy nhiên, do mỗi run tốn 5,8 giờ trên Mac MPS, các cấu hình góc theta khác (140 deg, 150 deg, 170 deg) chưa hoàn tất nên chưa đủ cơ sở kết luận 160 deg là tối ưu. Thực nghiệm Phase 4 xác thực chéo với SL-GCN cũng đã có kết quả sơ bộ.
3. **Facial landmarks chưa có kết luận ổn định.** QIPEDC-derived VSL [9] báo cáo face giúp tăng accuracy, nhưng ablation nội bộ VSL400 cam_1 cho thấy thêm facial landmarks vào Combined Aug làm giảm Test Acc. Sự sụt giảm 1,34 điểm Test Acc trên SPOTER khi thêm facial landmarks có thể do video cam_1 quay toàn thân ở khoảng xa khiến keypoint mặt kém tin cậy, hoặc do mô hình với hidden_dim=108 chưa đủ capacity để xử lý lượng điểm ảnh tăng lên. Ngược lại, facial landmarks lại có tác động tích cực đối với mô hình kiến trúc SL-GCN [I1]. Điều này tạo research gap về chất lượng face landmarks theo góc quay, độ phân giải và mô hình.
4. **Multi-view hữu ích nhưng chưa phản ánh sản phẩm một webcam.** Multi-VSL và VSL400 có nhiều view, nhưng nếu hệ thống cuối dùng webcam đơn, cần benchmark front-view, missing-view và train-many-test-one.
5. **Chưa có chuẩn chung cho smoothing/filtering keypoint trong VSL.** Dự án có thể đo jitter và thử Moving average, Savitzky-Golay, Kalman hoặc One Euro Filter, nhưng cần ablation trực tiếp thay vì chọn theo kinh nghiệm.
6. **Thiếu đánh giá triển khai.** Nhiều bảng chỉ báo cáo accuracy; dự án cần thêm latency, FPS, kích thước checkpoint, VRAM và thời gian preprocessing.

Dựa trên sáu research gap ở trên, có thể phân ra ba giai đoạn.

## 6.1. Giai đoạn 1 - Kiểm tra dữ liệu

Trước khi huấn luyện, cần tạo một báo cáo thống kê gồm:

- Tổng số gloss.
- Tổng số video.
- Số signer.
- Số view.
- Số mẫu trên mỗi gloss.
- Số gloss chỉ có một mẫu.
- Độ dài video trung bình.
- FPS và độ phân giải.
- Tỉ lệ video lỗi hoặc thiếu.
- Mức độ chồng lấn signer giữa train và test.
- Tỉ lệ keypoint bàn tay bị thiếu nếu đã trích xuất pose.

Không nên chọn mô hình trước khi biết phân phối số mẫu trên mỗi lớp.

## 6.2. Giai đoạn 2 - Chia dữ liệu

Nên xây dựng ít nhất hai protocol.

### Protocol A - Vocabulary-covered split

Mọi gloss trong validation và test phải xuất hiện trong train. Protocol này đánh giá khả năng nhận dạng bộ từ vựng cố định.

### Protocol B - Signer-independent split

Signer trong test không xuất hiện trong train. Có thể dùng:

- Group split theo signer.
- Leave-One-Signer-Out nếu số signer đủ nhỏ.
- K-fold group cross-validation.

Trong công trình [9], LOSO thấp hơn stratified split khoảng 5-7 điểm phần trăm đối với các baseline được báo cáo. Đây là bằng chứng cho thấy khả năng tổng quát hóa sang signer mới cần được đánh giá riêng [9].

Hai protocol trả lời hai câu hỏi khác nhau và không thay thế cho nhau.

## 6.3. Giai đoạn 3 - Baseline tối thiểu

Dự án nên huấn luyện ít nhất ba baseline trên cùng split.

### Baseline A - RGB

- Input: 16 hoặc 32 frame RGB.
- Mô hình khởi đầu: I3D, Video Swin-T hoặc MViTv2.
- Ghi nhận: accuracy, macro-F1, VRAM, thời gian train và latency.

### Baseline B - Keypoint

- Input: hand và upper-body keypoint.
- Mô hình khởi đầu: BiLSTM hoặc Transformer encoder.
- Không áp dụng preprocessing phức tạp trong phiên bản đầu.

### Baseline C - Keypoint có preprocessing

- Anchor-based normalization.
- Missing-point reconstruction.
- Confidence masking.
- Smoothing nhẹ.
- Cùng kiến trúc với Baseline B.

So sánh B và C giúp đo trực tiếp giá trị của preprocessing mà không bị nhiễu bởi việc thay đổi mô hình.

## 6.4. Ablation cho keypoint

Một ma trận ablation đề xuất:

**Bảng 13. Các cấu hình ablation cho pipeline keypoint-based**

| Cấu hình | Hand | Body | Face | Normalize | Reconstruction | Smoothing | Augmentation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| K0 | x |  |  |  |  |  |  |
| K1 | x | x |  |  |  |  |  |
| K2 | x | x | x |  |  |  |  |
| K3 | x | x | x | x |  |  |  |
| K4 | x | x | x | x | x |  |  |
| K5 | x | x | x | x | x | x |  |
| K6 | x | x | x | x | x | x | x |

Tất cả cấu hình cần giữ nguyên:

- Train/test split.
- Mô hình.
- Hyperparameter chính.
- Random seed.
- Số epoch.
- Metric.

Nên chạy nhiều seed và báo cáo trung bình cùng độ lệch chuẩn.

## 6.5. Multi-view benchmark

Nếu sử dụng Multi-VSL hoặc VSL400, nên thực hiện:

1. Front view riêng.
2. Left view riêng.
3. Right view riêng.
4. Ba view với late fusion.
5. Ba view với shared-weight encoder.
6. Train ba view, test hai view.
7. Train ba view, test một view.
8. View dropout trong quá trình train.

Không nên chỉ báo cáo kết quả ba view vì điều đó không phản ánh tình huống sản phẩm sử dụng một webcam.

## 6.6. Chỉ số đánh giá

### Chỉ số nhận dạng

- Top-1 accuracy.
- Top-5 accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Balanced accuracy.
- Confusion matrix.
- Accuracy theo từng nhóm số mẫu trên lớp.

### Chỉ số tổng quát hóa

- Accuracy theo từng signer.
- Trung bình và độ lệch chuẩn qua các fold LOSO.
- Chênh lệch giữa signer-dependent và signer-independent.

### Chỉ số preprocessing keypoint

- Missing-hand frame rate.
- Missing-keypoint rate.
- Độ dài đoạn mất liên tục.
- Tỉ lệ frame phải reconstruction.
- Jitter trước và sau smoothing.
- Confidence trung bình của hand keypoints.

### Chỉ số triển khai

- Số tham số.
- FLOPs hoặc MACs.
- VRAM.
- Thời gian preprocessing.
- Latency cho một video.
- FPS.
- Dung lượng checkpoint.
- Mức sử dụng CPU và GPU.

---

# 7. Cách trình bày kết quả

Mỗi bảng kết quả cần ghi rõ:

- Dataset và phiên bản.
- Số gloss.
- Số video.
- Số signer.
- Split protocol.
- Signer-dependent hay signer-independent.
- Loại input.
- Pretrained weights.
- Augmentation.
- Metric.
- Số seed.
- Phần cứng.

**Ví dụ phát biểu phù hợp:**

> Trên split signer-independent của VSL400, mô hình A đạt Macro F1 cao hơn mô hình B 2,4 điểm khi cả hai sử dụng cùng keypoint, augmentation và random seed.

**Ví dụ phát biểu chưa đủ cơ sở:**

> Mô hình A tốt hơn mô hình B vì paper A có accuracy 90% còn paper B chỉ có 70%.

Phát biểu thứ hai không hợp lệ nếu hai paper dùng dataset hoặc protocol khác nhau.

---

# 8. Khuyến nghị cho dự án

Dựa trên tài liệu khảo sát và hai báo cáo thực nghiệm nội bộ, hướng triển khai ưu tiên được đề xuất như sau:

**Bảng 14. Các khuyến nghị triển khai cho dự án VSL**

| Ưu tiên | Khuyến nghị | Lý do |
|---:|---|---|
| 1 | Hoàn tất data audit và chuẩn hóa metadata trước khi chọn mô hình. | Tránh chọn pipeline sai khi chưa biết rõ phân phối gloss, signer, view và lỗi keypoint. |
| 2 | Giữ RGB baseline và keypoint baseline trên cùng split. | RGB là đối chứng cần thiết; keypoint phù hợp cho prototype nhẹ nhưng có nguy cơ mất thông tin handshape hoặc face. |
| 3 | Ưu tiên Transformer/SPOTER cho baseline keypoint chính. | Nội bộ [I1] cho thấy Transformer hand-focus converged Macro-F1 cao; [I1] cũng cho thấy SPOTER + neck anchor mạnh trên VSL400 cam_1. |
| 4 | Tách preprocessing thành module độc lập. | Dễ ablation TBL, interpolation, anchor, smoothing, augmentation và face subset. |
| 5 | Ưu tiên neck anchor và hand-focused normalization trong vòng thí nghiệm kế tiếp. | Run 08 là cấu hình tốt nhất hiện có với Test Accuracy 84,08%. Nên kết hợp chung với Keypoint Interpolation, vì riêng Neck Anchor + Interpolation giúp mô hình SPOTER đạt Test Accuracy 84,08%, tăng 5,49 điểm so với baseline chưa qua xử lý [I1]. |
| 6 | Không mặc định thêm facial landmarks vào pipeline cuối. | QIPEDC-derived VSL [9] cho kết quả dương, nhưng nội bộ VSL400 cam_1 cho thấy thêm face làm giảm kết quả; cần ablation theo face_10, face_32, full face hoặc face encoder. Ảnh hưởng của facial landmarks phụ thuộc mạnh vào kiến trúc mô hình: chúng làm giảm hiệu suất ở mạng Transformer như SPOTER nhưng lại tăng độ chính xác đáng kể, thêm 2,15 điểm Test Accuracy, khi dùng kiến trúc đồ thị cục bộ như SL-GCN [I1]. |
| 7 | Không mặc định horizontal flip. | Flip có thể đổi hướng hoặc tay thuận; chỉ dùng khi có xác nhận chuyên môn hoặc ablation theo gloss. |
| 8 | Chạy smoothing/filtering như ablation riêng. | Có jitter benchmark nhưng chưa có kết quả trực tiếp; cần thử Moving average, Savitzky-Golay, One Euro Filter và Kalman filter. |
| 9 | Hoàn tất Phase 1 và Phase 4 trong ablation nội bộ. | TBL sweep chưa đủ để kết luận theta = 160° và taub = 400 ms là tối ưu. Ưu tiên hoàn tất các run TBL còn lại, gồm theta = 140°, theta = 150° và theta = 170°, do trước đó gặp lỗi thiếu dữ liệu và giới hạn tài nguyên. Đã có kết quả Phase 4 cho SL-GCN nhưng cần tinh chỉnh Adjacency Matrix để thu hẹp khoảng cách hiệu năng với SPOTER. |
| 10 | Chỉ triển khai multi-view sau khi single-view ổn định. | Nếu sản phẩm dùng webcam một góc, single-view và missing-view quan trọng hơn kết quả ba view lý tưởng. |

Với tài nguyên hạn chế, pose-based/keypoint-based vẫn là lựa chọn hợp lý để xây dựng prototype vì đầu vào nhỏ và mô hình có thể nhẹ hơn [4], [5]. Tuy nhiên, RGB cần được giữ làm baseline đối chứng, bởi WLASL và QIPEDC-derived VSL cho thấy mô hình RGB có thể đạt kết quả cao trong một số protocol [1], [9]. Kết luận cuối cùng cần dựa trên benchmark trực tiếp với dữ liệu VSL của dự án.

## Pipeline đề xuất

Từ kết quả thực nghiệm [M2] trên VSL400 cam_1, pipeline được đề xuất như sau:

- **Temporal Boundary Localization (TBL)** để cắt bỏ frame thừa, tạm dùng ngưỡng `theta=160 deg`, `taub=400ms`.
- **Keypoint Interpolation** để lấp các điểm mốc bị thiếu.
- **Neck Anchor Normalization**, vì cấu hình này mang lại Test Accuracy cao nhất, đạt 84,08% trên SPOTER.
- **Combined Augmentation** gồm Spatial + Perspective + Kinematic + Gaussian, chỉ áp dụng trong quá trình huấn luyện, không lưu trên dataset tĩnh. Tập hợp đa dạng này giúp Test Accuracy tăng +1,50 điểm.

## Future Work

Dự án cần bổ sung các thực nghiệm:

1. Lặp lại các run với ít nhất 3 random seed thay vì chỉ dùng seed = 42 để đánh giá độ lệch chuẩn, hay còn gọi là error bar.
2. Thử nghiệm tăng `hidden_dim` của SPOTER từ 108 lên 148 để tối ưu thêm Facial Landmarks.
3. Mở rộng kiểm thử trên các view khác như `cam_2`, `cam_3`.

---

# 9. Kết luận

Khảo sát cho thấy bài toán nhận dạng Ngôn ngữ ký hiệu tiếng Việt chịu ảnh hưởng đồng thời bởi phương thức biểu diễn, chất lượng dữ liệu, số mẫu trên mỗi lớp, độ đa dạng của signer, preprocessing và protocol đánh giá.

RGB giữ lại nhiều thông tin thị giác và đã vượt một số baseline pose trong các thử nghiệm cùng protocol trên WLASL [1]. Tuy nhiên, RGB có chi phí cao và dễ phụ thuộc vào dữ liệu lớn hoặc pretrained model. Pose-based giảm kích thước đầu vào và hỗ trợ triển khai nhẹ, nhưng hiệu quả phụ thuộc trực tiếp vào chất lượng pose estimator và quy trình preprocessing [4], [6].

Normalization và reconstruction không nên được xem là các bước phụ. SPOTER báo cáo normalization tạo ra mức cải thiện lớn trong ablation WLASL100 [5], trong khi Roh và cộng sự báo cáo anchor normalization kết hợp reconstruction tăng 6,05 điểm trong thiết lập của họ [6].

Facial landmarks cũng cần được benchmark. Công trình VSL [9] báo cáo việc bổ sung face vào hand-plus-pose tăng accuracy từ 48,67% lên 52,34% trong cùng protocol. Kết quả này hỗ trợ việc xem non-manual markers là một thành phần có thể mang thông tin bổ sung, thay vì chỉ tập trung vào bàn tay.

Multi-view có tiềm năng cải thiện nhận dạng, nhưng mức tăng phụ thuộc mô hình. Trên Multi-VSL1000, mức tăng Top-1 dao động từ 0,30 điểm với Video Swin Transformer đến 19,75 điểm với I3D [7]. Vì vậy, mọi nhận định về lợi ích của multi-view cần nêu rõ model, subset, metric và protocol.

Khuyến nghị cuối cùng là bắt đầu bằng benchmark có kiểm soát gồm RGB baseline, keypoint baseline và keypoint có preprocessing. Dự án cần đánh giá bằng cả vocabulary-covered split và signer-independent split, đồng thời báo cáo macro-F1, hiệu năng theo signer, lỗi keypoint và chi phí triển khai. Chỉ sau các thí nghiệm này mới có đủ cơ sở lựa chọn pipeline phù hợp cho hệ thống VSL thực tế.

---

# 10. Tài liệu tham khảo

[1] D. Li, C. Rodriguez, X. Yu và H. Li, "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison," WACV, 2020.

[2] H. R. V. Joze và O. Koller, "MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language," BMVC, 2019.

[3] A. Desai và cộng sự, "ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition," NeurIPS Datasets and Benchmarks, 2023.

[4] P. Selvaraj, G. Nc, P. Kumar và M. Khapra, "OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages," ACL, 2022.

[5] M. Bohácek và M. Hrúz, "Sign Pose-Based Transformer for Word-Level Sign Language Recognition," WACV Workshops, 2022.

[6] K. Roh, H. Lee, E. J. Hwang, S. Cho và J. C. Park, "Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition," SignLang at LREC-COLING, 2024.

[7] N. S. Dinh và cộng sự, "Sign Language Recognition: A Large-Scale Multi-View Dataset and Comprehensive Evaluation," WACV, 2025.

[8] "VSL400: A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition," Zenodo dataset record, 2026.

[9] H. M. Dung, N. V. Hung, N. K. Dang và P. T. H. Nhai, "Towards Realistic Vietnamese Sign Language Recognition: A Large-Scale Dataset and Rigorous Evaluation Protocol," IJSRED, tập 9, số 1, 2026.

[10] C. Lugaresi và cộng sự, "MediaPipe: A Framework for Building Perception Pipelines," 2019.

[I1] Nguồn thực nghiệm nội bộ: <https://github.com/thanhbinh55/VietnameseSignLanguageRecognition>
