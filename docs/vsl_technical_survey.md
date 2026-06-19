# BÁO CÁO KHẢO SÁT KỸ THUẬT (Preview - 10:41 19/06/2026)
## Các hướng tiếp cận và khuyến nghị xây dựng hệ thống nhận dạng Ngôn ngữ ký hiệu tiếng Việt

### Tuyên bố về phạm vi và nguồn thông tin
Báo cáo này được tổng hợp từ các bài báo khoa học, bộ dữ liệu, kho mã nguồn và tài liệu kỹ thuật công khai liên quan đến bài toán nhận dạng ngôn ngữ ký hiệu. Mục tiêu của tài liệu là cung cấp nguồn tham khảo ban đầu và đề xuất một hướng benchmark phù hợp cho dự án nhận dạng Ngôn ngữ ký hiệu tiếng Việt.

Các số liệu về quy mô dữ liệu, độ chính xác và kết quả thực nghiệm được giữ theo cách báo cáo của từng nguồn. Báo cáo không xem các số liệu này là kết quả thực nghiệm của nhóm thực hiện tài liệu, trừ khi có ghi rõ “kết quả thử nghiệm nội bộ”. Do khác biệt về ngôn ngữ ký hiệu, bộ dữ liệu, số lớp, số người ký hiệu, cách chia dữ liệu, phương pháp tiền xử lý, mô hình pretrained và chỉ số đánh giá, kết quả từ các công trình khác nhau không nên được so sánh trực tiếp như trên cùng một bảng xếp hạng.

Mọi khuyến nghị triển khai trong tài liệu cần được kiểm chứng lại bằng thực nghiệm trên dữ liệu VSL mà dự án thực sự sử dụng. Mặc dù nhóm thực hiện đã cố gắng đối chiếu thông tin với các nguồn gốc, báo cáo vẫn có thể tồn tại sai sót trong quá trình tổng hợp hoặc diễn giải.

---

## 1. Giới thiệu
Ngôn ngữ ký hiệu là một hệ thống ngôn ngữ thị giác, trong đó thông tin được truyền đạt thông qua hình dạng bàn tay, vị trí tay, hướng lòng bàn tay, chuyển động, tư thế cơ thể và các tín hiệu phi thủ công như biểu cảm khuôn mặt, hướng nhìn hoặc chuyển động đầu.

Bài toán tự động phân tích ngôn ngữ ký hiệu thường gồm nhiều mức độ khác nhau:
* **Isolated Sign Language Recognition – ISLR:** nhận dạng ký hiệu rời rạc, trong đó mỗi video thường chứa một ký hiệu cần phân loại.
* **Continuous Sign Language Recognition – CSLR:** nhận dạng chuỗi ký hiệu liên tục và xác định trình tự các ký hiệu xuất hiện.
* **Sign Language Translation – SLT:** chuyển nội dung ngôn ngữ ký hiệu sang câu trong ngôn ngữ nói hoặc ngôn ngữ viết.

Báo cáo này tập trung chủ yếu vào ISLR ở cấp độ từ, vì đây là hướng phù hợp hơn cho giai đoạn xây dựng baseline khi dữ liệu và tài nguyên tính toán còn hạn chế.

Một ký hiệu trong dữ liệu thường được gắn với một *gloss*. Gloss là nhãn văn bản được sử dụng để đại diện cho một đơn vị ký hiệu trong quá trình chú thích. Gloss hỗ trợ tổ chức dữ liệu và huấn luyện mô hình, nhưng không nên được hiểu đơn giản là một từ tương đương hoàn toàn trong tiếng Việt, vì cấu trúc ngôn ngữ ký hiệu và ngôn ngữ nói có thể khác nhau [7].

Về phương thức biểu diễn đầu vào, các hệ thống ISLR hiện nay thường sử dụng một hoặc nhiều loại dữ liệu sau:
1. Video RGB thô.
2. Keypoint hoặc skeleton.
3. Optical flow hoặc đặc trưng chuyển động.
4. Nhiều góc quay.
5. Kết hợp nhiều phương thức bằng mô hình multi-stream hoặc multimodal.

Không có một phương thức biểu diễn nào tối ưu tuyệt đối. Hiệu quả của mỗi hướng phụ thuộc vào chất lượng dữ liệu, số lượng mẫu trên mỗi lớp, số người ký hiệu, góc quay, độ phân giải, tài nguyên tính toán và yêu cầu triển khai.

---

## 2. Thuật ngữ sử dụng trong báo cáo

| Thuật ngữ | Giải thích |
| :--- | :--- |
| **Gloss** | Nhãn văn bản đại diện cho một ký hiệu trong dữ liệu chú thích. |
| **Signer** | Người thực hiện ngôn ngữ ký hiệu trong video. |
| **ISLR** | Nhận dạng từng ký hiệu rời rạc. |
| **CSLR** | Nhận dạng chuỗi nhiều ký hiệu liên tục. |
| **Signer-dependent** | Người xuất hiện trong tập kiểm tra có thể đã xuất hiện trong tập huấn luyện. |
| **Signer-independent** | Người trong tập kiểm tra không xuất hiện trong tập huấn luyện. |
| **LOSO** | Leave-One-Signer-Out: lần lượt giữ lại một signer để kiểm tra và huấn luyện bằng các signer còn lại. |
| **Keypoint** | Tọa độ của các khớp hoặc điểm mốc trên cơ thể, bàn tay và khuôn mặt. |
| **Non-manual markers** | Các tín hiệu không xuất phát trực tiếp từ chuyển động tay, chẳng hạn biểu cảm khuôn mặt, miệng, mắt hoặc tư thế đầu. |
| **Jitter** | Dao động nhỏ và không ổn định của keypoint giữa các frame liên tiếp. |
| **Early fusion** | Kết hợp dữ liệu hoặc đặc trưng từ nhiều nguồn ở giai đoạn đầu của mô hình. |
| **Late fusion** | Xử lý từng nguồn bằng các nhánh riêng rồi kết hợp đặc trưng hoặc dự đoán ở giai đoạn cuối. |
| **Missing view** | Một hoặc nhiều góc camera không có sẵn trong quá trình suy luận. |
| **Top-1 accuracy** | Tỉ lệ nhãn có xác suất cao nhất trùng với nhãn đúng. |
| **Top-5 accuracy** | Tỉ lệ nhãn đúng nằm trong năm dự đoán có xác suất cao nhất. |
| **Macro F1-score** | F1 được tính riêng cho từng lớp rồi lấy trung bình, phù hợp hơn accuracy khi dữ liệu mất cân bằng. |
| **Ablation study** | Thí nghiệm loại bỏ hoặc bổ sung từng thành phần để đo đóng góp thực sự của thành phần đó. |

---

## 3. Nguyên tắc tổng hợp và so sánh bằng chứng
Báo cáo phân biệt ba mức độ phát biểu.

* **Mức 1 – Kết quả được một công trình công bố**
    > *Ví dụ:* Roh và cộng sự báo cáo rằng pipeline anchor-based normalization kết hợp keypoint reconstruction cải thiện accuracy thêm 6,05 điểm phần trăm trong thiết lập của họ trên WLASL [6].
    >
    > Phát biểu này chỉ mô tả kết quả của công trình [6], không mặc định rằng mức cải thiện tương tự sẽ xuất hiện trên VSL.
* **Mức 2 – Quan sát từ thử nghiệm nội bộ**
    > *Ví dụ:* Trong thử nghiệm nội bộ của dự án, MediaPipe thường không phát hiện đủ keypoint bàn tay khi hai tay che nhau.
    >
    > Loại phát biểu này cần đi kèm dữ liệu, số lượng video được kiểm tra, tiêu chí xác định mất keypoint và kết quả đo cụ thể. Báo cáo hiện tại chưa đưa ra các kết quả nội bộ như vậy.
* **Mức 3 – Khuyến nghị cho dự án**
    > *Ví dụ:* Dự án nên benchmark keypoint trước và sau normalization, reconstruction và smoothing bằng cùng một mô hình và cùng cách chia dữ liệu.
    >
    > Đây là đề xuất thiết kế thí nghiệm, không phải kết luận rằng một kỹ thuật chắc chắn cải thiện kết quả.

Khi trình bày accuracy, chỉ nên so sánh trực tiếp nếu các mô hình sử dụng:
* Cùng dataset và phiên bản dữ liệu.
* Cùng số lớp.
* Cùng train/validation/test split.
* Cùng protocol signer-dependent hoặc signer-independent.
* Cùng metric.
* Điều kiện pretrained tương đương.
* Quy trình preprocessing và augmentation được mô tả rõ.

---

## 4. Khảo sát các hướng biểu diễn dữ liệu

### 4.1. Video RGB thô
Video RGB thô giữ lại gần như toàn bộ thông tin quan sát được trong mỗi frame, bao gồm:
* Hình dạng và độ xoay của bàn tay.
* Chi tiết các ngón tay.
* Vị trí tay so với cơ thể.
* Biểu cảm khuôn mặt.
* Chuyển động đầu và thân người.
* Thông tin về vật thể hoặc ngữ cảnh xung quanh.

Pipeline RGB thường gồm chuẩn hóa định dạng video, lấy mẫu frame, resize hoặc crop vùng signer, chuẩn hóa pixel và đưa chuỗi frame vào mô hình.

Các kiến trúc thường gặp gồm:
* CNN kết hợp LSTM hoặc GRU.
* 3D CNN như I3D.
* Video Transformer như Video Swin Transformer hoặc MViT.

Trong WLASL, I3D sử dụng RGB đạt Top-1 lần lượt là **65,89%**, **56,14%**, **47,33%** và **32,48%** trên các subset WLASL100, WLASL300, WLASL1000 và WLASL2000. Trong cùng protocol, Pose-TGCN đạt lần lượt **55,43%**, **38,32%**, **34,86%** và **23,65%** [1]. Kết quả này là bằng chứng cho thấy RGB có thể vượt pose trong thiết lập cụ thể của WLASL, nhưng không đủ để kết luận rằng RGB luôn tốt hơn pose trên mọi dataset.

Cần lưu ý rằng I3D trong thí nghiệm WLASL sử dụng trọng số pretrained từ ImageNet và Kinetics, trong khi pipeline pose phụ thuộc vào một pose estimator có sẵn. Vì vậy, chênh lệch không chỉ đến từ dữ liệu RGB hay keypoint mà còn liên quan đến pretraining, capacity của mô hình và lỗi tích lũy từ quá trình trích xuất keypoint [1].

Trong công trình VSL sử dụng dữ liệu QIPEDC, các tác giả báo cáo Video Swin Transformer đạt **58,92%**, I3D đạt **54,73%** và MMPose kết hợp Transformer đạt **56,41%** trên protocol vocabulary-complete gồm 3.782 lớp [9]. Các kết quả này được tạo ra trong cùng công trình nên có thể dùng để tham khảo tương đối, nhưng vẫn cần tái lập trước khi sử dụng làm baseline chính thức.

* **Ưu điểm:**
    * Không làm mất trực tiếp thông tin về texture, bàn tay và khuôn mặt.
    * Có thể học đặc trưng end-to-end.
    * Tận dụng được pretrained model từ video action recognition.
    * Phù hợp khi dữ liệu đủ lớn và GPU đáp ứng được yêu cầu.
* **Hạn chế:**
    * Tốn dung lượng lưu trữ và VRAM.
    * Thời gian huấn luyện và suy luận cao.
    * Có nguy cơ học các yếu tố không liên quan như nền, quần áo hoặc ánh sáng.
    * Dễ overfit nếu số mẫu trên mỗi gloss thấp.
    * Khó giải thích chính xác mô hình đang dựa vào tay, khuôn mặt hay background.

> **Nhận định cho dự án:** RGB nên được xem là một baseline quan trọng, không phải mặc định là phương án tốt nhất. Dự án cần so sánh RGB và pose trong cùng dataset, cùng split và cùng metric. Không nên dùng kết quả trên WLASL hoặc ASL để khẳng định trước rằng RGB sẽ vượt pose trên VSL.

### 4.2. Keypoint hoặc pose-based
Hướng pose-based chuyển mỗi frame thành tập tọa độ biểu diễn bàn tay, thân người và khuôn mặt. Một video khi đó trở thành chuỗi tensor có dạng gần với: $T \times K \times D$

Trong đó:
* $T$ là số frame.
* $K$ là số keypoint.
* $D$ là số chiều tọa độ, thường gồm $(x, y)$, đôi khi có thêm $(z)$ và confidence.

Các mô hình có thể xử lý dữ liệu này gồm:
* LSTM hoặc BiLSTM.
* Transformer encoder.
* ST-GCN hoặc các graph neural network.
* SPOTER.
* Các kiến trúc fusion giữa pose, hand và face.

OpenHands sử dụng pose như một phương thức chuẩn hóa cho các bài toán ISLR tài nguyên thấp. Framework này cung cấp các phiên bản pose của nhiều dataset, bốn nhóm mô hình gồm LSTM, Transformer, ST-GCN và SL-GCN, đồng thời phát hành checkpoint trên sáu ngôn ngữ ký hiệu [4].

Trong WLASL2000, OpenHands báo cáo SL-GCN đạt **30,6%** accuracy với pipeline chuẩn hóa của framework, cao hơn kết quả Pose-TGCN **23,65%** được dẫn lại từ công trình WLASL. Tuy nhiên, cần kiểm tra chi tiết cách xử lý và protocol trước khi xem đây là một phép so sánh hoàn toàn tương đương [4].

* **Ưu điểm:**
    * Kích thước đầu vào nhỏ hơn RGB.
    * Giảm ảnh hưởng của nền, màu quần áo và ánh sáng.
    * Thuận lợi cho mô hình nhẹ và triển khai thời gian thực.
    * Có cấu trúc hình học rõ ràng để áp dụng graph neural network.
    * Dễ phân tích missing point, jitter và chuyển động của từng khớp.
* **Hạn chế:**
    * Phụ thuộc vào pose estimator.
    * Có thể mất thông tin về handshape, orientation và texture.
    * Bàn tay nhỏ, bị che, chuyển động nhanh hoặc mờ có thể làm mất keypoint.
    * Lỗi trích xuất xảy ra trước mô hình nhận dạng nên không được sửa end-to-end.
    * Số lượng keypoint lớn không đồng nghĩa với thông tin hữu ích hơn nếu confidence thấp.

Công trình WLASL nhận xét rằng lỗi từ pose estimator có thể làm giảm hiệu quả của các mô hình pose-based [1]. Roh và cộng sự cũng mô tả trường hợp MediaPipe không phát hiện được bàn tay ở một số frame, khiến mô hình nhầm các ký hiệu có chuyển động cơ thể tương tự nhưng khác handshape [6].

> **Nhận định cho dự án:** Pose-based phù hợp để xây dựng baseline đầu tiên khi tài nguyên hạn chế. Tuy nhiên, dự án không nên xem file keypoint đã trích xuất là dữ liệu hoàn chỉnh. Cần đo riêng chất lượng trích xuất và thực hiện ablation cho normalization, reconstruction, smoothing và face landmarks.

### 4.3. Optical flow và đặc trưng chuyển động
Optical flow biểu diễn sự dịch chuyển của pixel giữa các frame liên tiếp. Hướng này có thể bổ sung thông tin về:
* Hướng chuyển động.
* Tốc độ tương đối.
* Quỹ đạo tay.
* Những khác biệt động học khó quan sát từ một frame đơn lẻ.

Optical flow thường được sử dụng như một stream bổ sung thay vì thay thế hoàn toàn RGB hoặc keypoint. Một hệ thống có thể gồm:
* Stream RGB.
* Stream optical flow.
* Stream keypoint.
* Một module fusion kết hợp các đặc trưng.

Optical flow có chi phí tiền xử lý tương đối cao và có thể bị ảnh hưởng bởi chuyển động camera, background động hoặc motion blur. Vì báo cáo hiện chưa có bằng chứng trực tiếp trên cùng dữ liệu VSL cho thấy optical flow cải thiện bao nhiêu, hướng này nên được xem là một baseline mở rộng sau khi RGB và pose đã được thiết lập ổn định.

### 4.4. Multi-view
Multi-view sử dụng các video đồng bộ từ nhiều góc camera. Hướng này có thể hỗ trợ trong các trường hợp:
* Hai tay che nhau ở góc chính diện.
* Chuyển động diễn ra theo chiều sâu.
* Hướng lòng bàn tay khó quan sát từ một camera.
* Hai gloss có chuyển động tương tự khi nhìn từ trước nhưng khác nhau khi nhìn từ bên.

Ba cách kết hợp phổ biến gồm:
* **Early fusion:** kết hợp dữ liệu hoặc đặc trưng thấp từ nhiều view trước khi đưa vào phần chính của mô hình.
* **Late fusion:** mỗi view có encoder riêng, sau đó kết hợp embedding hoặc xác suất dự đoán.
* **Cross-view learning:** dùng nhiều view trong huấn luyện nhằm cải thiện biểu diễn, kể cả khi suy luận chỉ có một số view.

Multi-VSL gồm 1.000 gloss, 84.764 video và 30 signer với nhiều góc nhìn [7]. Trong thử nghiệm Multi-VSL1000:

| Mô hình | Top-1 một view | Top-1 ba view | Chênh lệch |
| :--- | :---: | :---: | :---: |
| I3D | 40,60% | 60,35% | +19,75 điểm |
| Video Swin Transformer | 76,43% | 76,73% | +0,30 điểm |
| MViTv2 | 81,52% | 84,71% | +3,19 điểm |
| VTNPF | 66,21% | 76,32% | +10,11 điểm |

Kết quả cho thấy ba view cải thiện Top-1 đối với tất cả các baseline được báo cáo, nhưng mức cải thiện khác nhau đáng kể giữa các mô hình [7]. Do đó, kết luận phù hợp không phải là “multi-view luôn cải thiện khoảng 20%”, mà là:
> Trong protocol của Multi-VSL, multi-view cải thiện kết quả của tất cả baseline được thử nghiệm; mức tăng lớn nhất là 19,75 điểm phần trăm với I3D, trong khi Video Swin Transformer chỉ tăng 0,30 điểm [7].

Multi-VSL cũng đánh giá missing view. Với I3D trên Multi-VSL1000, mô hình một view đạt **40,60%** Top-1. Mô hình ba view được huấn luyện với dữ liệu có view bị thiếu đạt **57,04%** khi tỉ lệ thiếu view là 50%, cao hơn 16,44 điểm so với baseline một view trong bảng kết quả [7]. Đây là bằng chứng cho thấy huấn luyện có mô phỏng missing view có thể tăng độ bền trong thiết lập của công trình này.

> **Nhận định cho dự án:** Multi-view chỉ nên được ưu tiên khi dữ liệu thực sự có các camera được đồng bộ. Nếu sản phẩm cuối chỉ sử dụng webcam một góc, dự án cần báo cáo rõ:
> * Kết quả khi train và test một view.
> * Kết quả khi train và test nhiều view.
> * Kết quả khi train nhiều view nhưng test thiếu view.
> * Chi phí tăng thêm của nhiều encoder và nhiều camera.

---

## 5. Tiền xử lý keypoint

### 5.1. Normalization
Nếu sử dụng tọa độ ảnh gốc, mô hình có thể học vị trí tuyệt đối của signer, khoảng cách tới camera hoặc kích thước cơ thể. Normalization nhằm giảm các biến thiên không liên quan nhưng vẫn giữ thông tin ngôn ngữ cần thiết.

Các cách thường được sử dụng gồm:
* Đưa gốc tọa độ về vai, cổ, hông hoặc cổ tay.
* Scale skeleton theo khoảng cách hai vai.
* Chuẩn hóa bàn tay trong hệ tọa độ cục bộ.
* Sử dụng vector xương giữa các cặp khớp.
* Tách biểu diễn handshape và vị trí bàn tay so với cơ thể.

SPOTER báo cáo baseline không normalization và augmentation đạt **44,96%** trên WLASL100. Khi bổ sung normalization, kết quả tăng hơn 14 điểm phần trăm trong ablation của công trình [5]. Kết quả này cho thấy normalization có ảnh hưởng lớn trong thiết lập SPOTER, nhưng không nên mặc định mức tăng tương tự cho mọi pipeline.

Roh và cộng sự sử dụng các anchor riêng cho cơ thể và bàn tay nhằm loại bỏ thông tin vị trí không cần thiết nhưng giữ hình dạng bàn tay [6]. Công trình báo cáo toàn bộ pipeline preprocessing cải thiện accuracy 6,05 điểm và đạt **83,26%** khi kết hợp augmentation trên WLASL trong protocol của họ [6].

Hai kết quả của SPOTER và Roh và cộng sự không nên được so sánh trực tiếp vì có thể khác subset, kiến trúc, cách chia dữ liệu, số keypoint và augmentation.

### 5.2. Keypoint reconstruction
Keypoint reconstruction nhằm khôi phục các điểm không được pose estimator phát hiện. Một cách đơn giản là nội suy giữa các frame trước và sau.

Roh và cộng sự sử dụng bilinear interpolation để tái tạo keypoint bàn tay bị thiếu. Case study của công trình cho thấy reconstruction có thể sửa dự đoán ở những video mà một số frame không phát hiện được bàn tay [6].

Tuy nhiên, nội suy không tạo ra thông tin thật trong trường hợp bàn tay bị che trong một đoạn dài. Vì vậy, dự án cần báo cáo:
* Tỉ lệ frame mất một tay.
* Tỉ lệ frame mất cả hai tay.
* Độ dài trung bình của đoạn mất liên tục.
* Tỉ lệ keypoint được nội suy.
* Kết quả trước và sau reconstruction.

### 5.3. Smoothing và filtering
Jitter là sự dao động của keypoint dù chuyển động thật không thay đổi tương ứng. Các bộ lọc có thể thử nghiệm gồm:
* Moving average.
* Exponential moving average.
* Savitzky–Golay.
* Kalman filter.
* One Euro Filter.

Smoothing quá mạnh có thể làm mất chuyển động nhanh ở ngón tay. Do báo cáo hiện chưa có kết quả VSL chứng minh một bộ lọc cụ thể là tối ưu, smoothing nên được xem là một biến số cần ablation thay vì bước bắt buộc.

Ví dụ, dự án có thể so sánh:
* Không smoothing.
* Moving average với cửa sổ 3 frame.
* Moving average với cửa sổ 5 frame.
* One Euro Filter.
* Savitzky–Golay với nhiều cấu hình.

Tất cả cấu hình phải sử dụng cùng mô hình, seed và split.

### 5.4. Facial landmarks và non-manual markers
Ngôn ngữ ký hiệu không chỉ chứa thông tin ở bàn tay. Biểu cảm khuôn mặt, chuyển động lông mày, miệng, mắt và đầu có thể bổ sung thông tin phân biệt.

Trong công trình [9], các tác giả báo cáo:

| Đặc trưng | Accuracy |
| :--- | :---: |
| Chỉ bàn tay | 42,18% |
| Bàn tay và pose | 48,67% |
| Bàn tay, pose và toàn bộ facial landmarks | 52,34% |
| Bàn tay, pose và facial landmarks được chọn | 51,89% |

Trong cùng protocol, thêm pose vào hand-only tăng 6,49 điểm; thêm toàn bộ facial landmarks vào hand-plus-pose tăng thêm 3,67 điểm [9].

Kết quả này hỗ trợ việc benchmark face landmarks trên dữ liệu VSL, nhưng không đồng nghĩa rằng sử dụng toàn bộ face mesh luôn tốt nhất. Dự án cần so sánh giữa:
* Không dùng mặt.
* Một nhóm điểm quanh mắt, lông mày và miệng.
* Toàn bộ face mesh.
* Embedding khuôn mặt được học bằng một encoder riêng.

---

## 6. Data augmentation
Augmentation trong ngôn ngữ ký hiệu cần được thiết kế thận trọng vì các phép biến đổi hình ảnh có thể thay đổi ý nghĩa của ký hiệu.

Các phép biến đổi có thể xem xét gồm:
* Crop nhẹ.
* Zoom.
* Rotation nhỏ.
* Thay đổi độ sáng hoặc độ tương phản.
* Perspective transformation nhỏ.
* Temporal crop.
* Thay đổi tốc độ ở mức giới hạn.
* Gaussian noise trên keypoint.
* Joint dropout.
* Frame dropping hoặc temporal masking.

Không nên đưa ra quy tắc tuyệt đối rằng horizontal flip luôn đúng hoặc luôn sai. WLASL và Multi-VSL có sử dụng horizontal flipping trong pipeline augmentation của họ [1], [7]. Ngược lại, công trình [9] không sử dụng horizontal hoặc vertical flip vì cho rằng hướng tay và hướng chuyển động có thể mang ý nghĩa ngôn ngữ.

Do đó, khuyến nghị phù hợp là:
> Horizontal flip chỉ nên được sử dụng sau khi người có chuyên môn ngôn ngữ ký hiệu xác nhận phép biến đổi không làm thay đổi nhãn, hoặc sau khi có thí nghiệm ablation riêng cho từng nhóm gloss.

Trong công trình [9], BiLSTM không augmentation đạt **34,52%**, trong khi sử dụng đầy đủ năm nhóm augmentation đạt **57,83%**, tương ứng mức tăng tuyệt đối 23,31 điểm trong protocol của công trình. Dữ liệu có 64% lớp chỉ chứa một mẫu, vì vậy mức cải thiện lớn này gắn với một thiết lập đặc biệt mất cân bằng và ít mẫu [9]. Không nên dùng con số 23,31 điểm như mức cải thiện kỳ vọng trên các dataset khác.

Nguyên tắc bắt buộc là chia dữ liệu gốc thành train, validation và test trước, sau đó chỉ augmentation tập train. Nếu các phiên bản augmentation của cùng một video xuất hiện ở cả train và test, kết quả sẽ bị data leakage.

---

## 7. Phân loại các tài nguyên liên quan

### 7.1. Dataset
| Dataset | Ngôn ngữ | Quy mô chính | Góc quay | Vai trò tham khảo |
| :--- | :---: | :--- | :--- | :--- |
| **WLASL** [1] | ASL | 2.000 gloss, 21.083 video, 119 signer | Chủ yếu single-view | Benchmark RGB và pose trên nhiều kích thước từ vựng. |
| **MS-ASL** [2] | ASL | 1.000 gloss, 25.513 video, hơn 200 signer | Single-view | Nhấn mạnh đánh giá trên signer chưa xuất hiện trong train. |
| **ASL Citizen** [3] | ASL | 2.731 sign, 83.399 video, 52 signer | Webcam, môi trường đa dạng | Tham khảo cách thu thập crowdsourced và dictionary retrieval. |
| **Multi-VSL** [7] | VSL | 1.000 gloss, 84.764 video, 30 signer | Ba view | Benchmark single-view, multi-view và missing view. |
| **VSL400** [8] | VSL | 400 gloss, 74.259 clip, 28 signer | Ba view đồng bộ | Phù hợp xây dựng và kiểm thử prototype VSL multi-view. |
| **QIPEDC-derived VSL** [9] | VSL | 3.782 gloss, 6.046 video, 11 signer | Theo dữ liệu nguồn | Nghiên cứu từ vựng lớn, mất cân bằng lớp và protocol vocabulary-complete. |

### 7.2. Mô hình
| Mô hình | Loại đầu vào | Đặc điểm |
| :--- | :--- | :--- |
| **I3D** | RGB | 3D convolution học đặc trưng không gian–thời gian. |
| **Video Swin Transformer** | RGB | Video Transformer sử dụng shifted windows. |
| **MViT/MViTv2** | RGB | Transformer đa tỉ lệ cho video. |
| **LSTM/BiLSTM** | Keypoint hoặc embedding | Baseline tuần tự, tương đối dễ triển khai. |
| **ST-GCN** | Skeleton graph | Biểu diễn khớp và liên kết cơ thể dưới dạng graph không gian–thời gian. |
| **SL-GCN** | Skeleton graph | Graph model được điều chỉnh cho dữ liệu ngôn ngữ ký hiệu. |
| **SPOTER** [5] | Pose 2D | Transformer cho nhận dạng ký hiệu cấp từ với chi phí thấp. |
| **MMPose + Transformer** | Keypoint | Tách pose estimation và sequence modeling. |
| **VTNPF** | RGB kết hợp pose flow | Baseline kết hợp nhiều nguồn đặc trưng trong Multi-VSL. |

> SPOTER là một mô hình, không phải dataset hay framework. Công trình báo cáo SPOTER đạt **63,18%** trên WLASL100 và **43,78%** trên WLASL300, cao hơn các pose baseline được so sánh trong cùng bảng, nhưng vẫn thấp hơn một số phương pháp appearance-based [5].

### 7.3. Framework và công cụ
| Framework/công cụ | Chức năng |
| :--- | :--- |
| **OpenHands** [4] | Chuẩn hóa pose dataset, cung cấp baseline, checkpoint và pipeline cho ISLR. |
| **MediaPipe** | Trích xuất pose, hand và face landmarks. |
| **OpenPose** | Trích xuất keypoint cơ thể và bàn tay. |
| **MMPose** | Toolbox pose estimation với nhiều backbone và cấu hình. |
| **PyTorch/TensorFlow** | Huấn luyện và triển khai mô hình. |
| **Repository của dataset** | Cung cấp script tải dữ liệu, metadata, split hoặc baseline; repository không đồng nghĩa với dataset hay paper. |

> OpenHands là framework/library, không phải một mô hình duy nhất. Trong OpenHands có nhiều mô hình như LSTM, Transformer, ST-GCN và SL-GCN [4].

### 7.4. Công trình về preprocessing
| Công trình | Đóng góp chính | Kết quả cần lưu ý |
| :--- | :--- | :--- |
| **SPOTER** [5] | Pose normalization và augmentation cho Transformer | Normalization tăng hơn 14 điểm trong ablation WLASL100 của công trình. |
| **Roh et al.** [6] | Anchor-based normalization và reconstruction cho keypoint MediaPipe | Pipeline cải thiện 6,05 điểm và đạt **83,26%** trong protocol của công trình. |
| **OpenHands** [4] | Chuẩn hóa pose representation trên nhiều dataset và ngôn ngữ | Cung cấp cách tổ chức pipeline và baseline thống nhất. |
| **QIPEDC-derived VSL study** [9] | Ablation hand, pose, face và augmentation trên VSL | Facial landmarks tăng 3,67 điểm; augmentation tăng từ **34,52%** lên **57,83%** trong thiết lập được báo cáo. |

---

## 8. Khuyến nghị pipeline benchmark cho dự án VSL

### 8.1. Giai đoạn 1 – Kiểm tra dữ liệu
Trước khi huấn luyện, cần tạo một báo cáo thống kê gồm:
* Tổng số gloss.
* Tổng số video.
* Số signer.
* Số view.
* Số mẫu trên mỗi gloss.
* Số gloss chỉ có một mẫu.
* Độ dài video trung bình.
* FPS và độ phân giải.
* Tỉ lệ video lỗi hoặc thiếu.
* Mức độ chồng lấn signer giữa train và test.
* Tỉ lệ keypoint bàn tay bị thiếu nếu đã trích xuất pose.

Không nên chọn mô hình trước khi biết phân phối số mẫu trên mỗi lớp.

### 8.2. Giai đoạn 2 – Chia dữ liệu
Nên xây dựng ít nhất hai protocol.

* **Protocol A – Vocabulary-covered split:** Mọi gloss trong validation và test phải xuất hiện trong train. Protocol này đánh giá khả năng nhận dạng bộ từ vựng cố định.
* **Protocol B – Signer-independent split:** Signer trong test không xuất hiện trong train. Có thể dùng:
    * Group split theo signer.
    * Leave-One-Signer-Out (LOSO) nếu số signer đủ nhỏ.
    * K-fold group cross-validation.

Trong công trình [9], LOSO thấp hơn stratified split khoảng 5–7 điểm phần trăm đối với các baseline được báo cáo. Đây là bằng chứng cho thấy khả năng tổng quát hóa sang signer mới cần được đánh giá riêng [9].

Hai protocol trả lời hai câu hỏi khác nhau và không thay thế cho nhau.

### 8.3. Giai đoạn 3 – Baseline tối thiểu
Dự án nên huấn luyện ít nhất ba baseline trên cùng split.

* **Baseline A – RGB:**
    * Input: 16 hoặc 32 frame RGB.
    * Mô hình khởi đầu: I3D, Video Swin-T hoặc MViTv2.
    * Ghi nhận: accuracy, macro-F1, VRAM, thời gian train và latency.
* **Baseline B – Keypoint:**
    * Input: hand và upper-body keypoint.
    * Mô hình khởi đầu: BiLSTM hoặc Transformer encoder.
    * Không áp dụng preprocessing phức tạp trong phiên bản đầu.
* **Baseline C – Keypoint có preprocessing:**
    * Anchor-based normalization.
    * Missing-point reconstruction.
    * Confidence masking.
    * Smoothing nhẹ.
    * Cùng kiến trúc với Baseline B.

So sánh B và C giúp đo trực tiếp giá trị của preprocessing mà không bị nhiễu bởi việc thay đổi mô hình.

### 8.4. Ablation cho keypoint
Một ma trận ablation đề xuất:

| Cấu hình | Hand | Body | Face | Normalize | Reconstruction | Smoothing | Augmentation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K0** | ✓ | | | | | | |
| **K1** | ✓ | ✓ | | | | | |
| **K2** | ✓ | ✓ | ✓ | | | | |
| **K3** | ✓ | ✓ | ✓ | ✓ | | | |
| **K4** | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **K5** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **K6** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Tất cả cấu hình cần giữ nguyên:
* Train/test split.
* Mô hình.
* Hyperparameter chính.
* Random seed.
* Số epoch.
* Metric.

Nên chạy nhiều seed và báo cáo trung bình cùng độ lệch chuẩn.

### 8.5. Multi-view benchmark
If sử dụng Multi-VSL hoặc VSL400, nên thực hiện:
* Front view riêng.
* Left view riêng.
* Right view riêng.
* Ba view với late fusion.
* Ba view với shared-weight encoder.
* Train ba view, test hai view.
* Train ba view, test một view.
* View dropout trong quá trình train.

Không nên chỉ báo cáo kết quả ba view vì điều đó không phản ánh tình huống sản phẩm sử dụng một webcam.

### 8.6. Chỉ số đánh giá
* **Chỉ số nhận dạng:**
    * Top-1 accuracy.
    * Top-5 accuracy.
    * Macro precision.
    * Macro recall.
    * Macro F1-score.
    * Balanced accuracy.
    * Confusion matrix.
    * Accuracy theo từng nhóm số mẫu trên lớp.
* **Chỉ số tổng quát hóa:**
    * Accuracy theo từng signer.
    * Trung bình và độ lệch chuẩn qua các fold LOSO.
    * Chênh lệch giữa signer-dependent và signer-independent.
* **Chỉ số preprocessing keypoint:**
    * Missing-hand frame rate.
    * Missing-keypoint rate.
    * Độ dài đoạn mất liên tục.
    * Tỉ lệ frame phải reconstruction.
    * Jitter trước và sau smoothing.
    * Confidence trung bình của hand keypoints.
* **Chỉ số triển khai:**
    * Số tham số.
    * FLOPs hoặc MACs.
    * VRAM.
    * Thời gian preprocessing.
    * Latency cho một video.
    * FPS.
    * Dung lượng checkpoint.
    * Mức sử dụng CPU và GPU.

---

## 9. Cách trình bày kết quả
Mỗi bảng kết quả cần ghi rõ:
* Dataset và phiên bản.
* Số gloss.
* Số video.
* Số signer.
* Split protocol.
* Signer-dependent hay signer-independent.
* Loại input.
* Pretrained weights.
* Augmentation.
* Metric.
* Số seed.
* Phần cứng.

> **Ví dụ phát biểu phù hợp:** Trên split signer-independent của VSL400, mô hình A đạt Macro F1 cao hơn mô hình B 2,4 điểm khi cả hai sử dụng cùng keypoint, augmentation và random seed.
>
> **Ví dụ phát biểu chưa đủ cơ sở:** Mô hình A tốt hơn mô hình B vì paper A có accuracy 90% còn paper B chỉ có 70%. (Phát biểu thứ hai không hợp lệ nếu hai paper dùng dataset hoặc protocol khác nhau).

---

## 10. Khuyến nghị cho dự án
Dựa trên các tài liệu đã khảo sát, hướng triển khai ban đầu được đề xuất như sau:
1. Kiểm tra và thống kê lại dữ liệu VSL trước khi chọn mô hình.
2. Xây dựng một baseline RGB và một baseline keypoint trên cùng split.
3. Dùng BiLSTM hoặc Transformer encoder làm baseline keypoint đầu tiên.
4. Tách preprocessing keypoint thành một module độc lập.
5. Benchmark lần lượt hand, body và face landmarks.
6. Đo missing keypoint trước khi áp dụng reconstruction.
7. Thực hiện ablation normalization, reconstruction, smoothing và augmentation.
8. Sử dụng signer-independent split bên cạnh split bảo đảm vocabulary coverage.
9. Chỉ triển khai multi-view sau khi baseline single-view ổn định.
10. Báo cáo cả hiệu quả nhận dạng và chi phí tính toán.

Với tài nguyên hạn chế, pose-based là lựa chọn hợp lý để xây dựng prototype vì đầu vào nhỏ và mô hình có thể nhẹ hơn [4], [5]. Tuy nhiên, RGB vẫn cần được giữ làm baseline đối chứng, bởi các thử nghiệm trên WLASL và nghiên cứu VSL [9] cho thấy mô hình RGB có thể đạt kết quả cao hơn trong một số protocol [1], [9].

Không nên quyết định pipeline chỉ dựa trên kết quả từ ASL. Kết luận cuối cùng cần dựa trên benchmark trực tiếp với dữ liệu VSL của dự án.

---

## 11. Kết luận
Khảo sát cho thấy bài toán nhận dạng Ngôn ngữ ký hiệu tiếng Việt chịu ảnh hưởng đồng thời bởi phương thức biểu diễn, chất lượng dữ liệu, số mẫu trên mỗi lớp, độ đa dạng của signer, preprocessing và protocol đánh giá.

RGB giữ lại nhiều thông tin thị giác và đã vượt một số baseline pose trong các thử nghiệm cùng protocol trên WLASL [1]. Tuy nhiên, RGB có chi phí cao và dễ phụ thuộc vào dữ liệu lớn hoặc pretrained model. Pose-based giảm kích thước đầu vào và hỗ trợ triển khai nhẹ, nhưng hiệu quả phụ thuộc trực tiếp vào chất lượng pose estimator và quy trình preprocessing [4], [6].

Normalization và reconstruction không nên được xem là các bước phụ. SPOTER báo cáo normalization tạo ra mức cải thiện lớn trong ablation WLASL100 [5], trong khi Roh và cộng sự báo cáo anchor normalization kết hợp reconstruction tăng 6,05 điểm trong thiết lập của họ [6].

Facial landmarks cũng cần được benchmark. Công trình VSL [9] báo cáo việc bổ sung face vào hand-plus-pose tăng accuracy từ **48,67%** lên **52,34%** trong cùng protocol. Kết quả này hỗ trợ việc xem non-manual markers là một thành phần có thể mang thông tin bổ sung, thay vì chỉ tập trung vào bàn tay.

Multi-view có tiềm năng cải thiện nhận dạng, nhưng mức tăng phụ thuộc mô hình. Trên Multi-VSL1000, mức tăng Top-1 dao động từ 0,30 điểm với Video Swin Transformer đến 19,75 điểm với I3D [7]. Vì vậy, mọi nhận định về lợi ích của multi-view cần nêu rõ model, subset, metric và protocol.

Khuyến nghị cuối cùng là bắt chuỗi bằng benchmark có kiểm soát gồm RGB baseline, keypoint baseline và keypoint có preprocessing. Dự án cần đánh giá bằng cả vocabulary-covered split và signer-independent split, đồng thời báo cáo macro-F1, hiệu năng theo signer, lỗi keypoint và chi phí triển khai. Chỉ sau các thí nghiệm này mới có đủ cơ sở lựa chọn pipeline phù hợp cho hệ thống VSL thực tế.

---

## 12. Tài liệu tham khảo
* **[1]** D. Li, C. Rodriguez, X. Yu và H. Li, “Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison,” *WACV*, 2020.
* **[2]** H. R. V. Joze và O. Koller, “MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language,” *BMVC*, 2019.
* **[3]** A. Desai và cộng sự, “ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition,” *NeurIPS Datasets and Benchmarks*, 2023.
* **[4]** P. Selvaraj, G. Nc, P. Kumar và M. Khapra, “OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages,” *ACL*, 2022.
* **[5]** M. Boháček và M. Hrúz, “Sign Pose-Based Transformer for Word-Level Sign Language Recognition,” *WACV Workshops*, 2022.
* **[6]** K. Roh, H. Lee, E. J. Hwang, S. Cho và J. C. Park, “Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition,” *SignLang at LREC-COLING*, 2024.
* **[7]** N. S. Dinh và cộng sự, “Sign Language Recognition: A Large-Scale Multi-View Dataset and Comprehensive Evaluation,” *WACV*, 2025.
* **[8]** “VSL400: A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition,” *Zenodo dataset record*, 2026.
* **[9]** H. M. Dung, N. V. Hung, N. K. Dang và P. T. H. Nhai, “Towards Realistic Vietnamese Sign Language Recognition: A Large-Scale Dataset and Rigorous Evaluation Protocol,” *IJSRED*, tập 9, số 1, 2026.
* **[10]** C. Lugaresi và cộng sự, “MediaPipe: A Framework for Building Perception Pipelines,” 2019.