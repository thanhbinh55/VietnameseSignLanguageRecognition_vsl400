# BÁO CÁO KHẢO SÁT KỸ THUẬT (Preview - 10:41 19/06/2026)
## Phân tích đối sánh Ngôn ngữ ký hiệu và Thách thức xây dựng hệ thống nhận diện

### Link đã chuyển sang Latex
https://www.overleaf.com/3443147841xmmmdmwrbrcv#e736bf 

---

## TÓM TẮT
Ngôn ngữ Ký hiệu Việt Nam (VSL) là một hệ thống ngôn ngữ độc lập, hoàn chỉnh, sở hữu các đặc trưng âm vị học, ngữ pháp và cú pháp riêng biệt, không phụ thuộc vào hệ thống ngôn ngữ nói. Bài báo cáo này tổng hợp các đặc điểm cốt lõi của VSL dưới góc độ ngôn ngữ học đối sánh với Ngôn ngữ Ký hiệu Mỹ (ASL), phân tích cơ chế cấu tạo từ ghép, cấu trúc cấp câu và các hiện tượng biến đổi sinh học vận động liên quan. Đồng thời, báo cáo chỉ ra những thách thức kỹ thuật trọng yếu trong việc xây dựng các mô hình trí tuệ nhân tạo nhận diện VSL - bao gồm tính độc lập của người ký, rò rỉ dữ liệu và hiện tượng che khuất - từ đó đề xuất các định hướng tối ưu hóa kiến trúc mô hình đa phương thức.

---

## PHẦN I: KHÁI NIỆM & BẢN CHẤT CỦA NNKH

### 1. Tính độc lập của hệ thống ngôn ngữ
Ngôn ngữ Ký hiệu Việt Nam (VSL) được công nhận là một ngôn ngữ tự nhiên và chính thống, có lịch sử hình thành lâu đời và đóng vai trò là công cụ giao tiếp độc lập của cộng đồng người Điếc tại Việt Nam. VSL là một hệ thống hoàn chỉnh, sở hữu đầy đủ các quy luật ngữ pháp, cấu trúc cú pháp và kho từ vựng riêng biệt, hoàn toàn không phải là sự sao chép cơ học hay việc diễn dịch "từng từ một" (*word-by-word*) từ tiếng Việt nói. 

VSL vận dụng phương thức tư duy trực quan và hình tượng để diễn đạt các ý niệm. Hệ thống từ vựng của VSL mang tính cấp tiến, liên tục vận động và biến đổi theo thời gian để thích ứng với các khái niệm mới phát sinh trong xã hội. Đối với các khái niệm mới chưa được quy ước ký hiệu, người ký sẽ vận dụng phương pháp đánh vần ngón tay (*Fingerspelling*) để cấu trúc hóa từ ngữ theo từng chữ cái cấu thành. 

### 2. Kênh truyền tải và Không gian thực hiện
Khác với các ngôn ngữ nói sử dụng kênh "Thính giác - Phát âm", VSL hoạt động dựa trên kênh "Thị giác - Vận động", nghĩa là sử dụng cơ quan thị giác (mắt) để tiếp nhận thông tin và các chuyển động cơ thể (vận động) để truyền tải thông tin. Không gian thực hiện ký hiệu phần lớn được giới hạn trong khoảng không từ ngang rốn lên đến đỉnh đầu, trong đó khu vực trước ngực và khuôn mặt là nơi có mật độ xuất hiện cử chỉ cao nhất.

### 3. Văn hóa định danh trong cộng đồng
Dưới góc độ văn hóa và xã hội học, thuật ngữ **"Người Điếc" (Deaf person)** được ưu tiên sử dụng phổ biến trong cộng đồng và các công ước quốc tế nhằm khẳng định bản sắc văn hóa và ngôn ngữ độc lập. Ngược lại, thuật ngữ *"Người khiếm thính" (Hearing impaired)* thường chỉ giới hạn trong các văn bản hành chính hoặc ngữ cảnh y tế liên quan đến mức độ suy giảm thính lực. Trong cấu trúc tương tác xã hội, cộng đồng phân định rõ ràng thành hai nhóm đối tượng: "người điếc" và "người nghe". 

---

## PHẦN II: THÀNH TỐ CẤU THÀNH MỘT KÝ HIỆU
Mỗi một ký hiệu đơn lẻ (được gọi là một *Gloss*) trong VSL không chỉ đơn thuần là một động tác tay, mà là sự kết hợp đồng thời của 5 thành tố nền tảng (tương đương với các âm vị trong cấu trúc ngôn ngữ nói):

* **Hình dạng bàn tay:** Tư thế, cấu hình co duỗi của lòng bàn tay và các ngón tay khi thực hiện ký hiệu.
* **Vị trí đặt tay:** Vị trí tương quan của bàn tay so với các mốc cơ thể (trán, cằm, ngực, vai) hoặc trong không gian ba chiều trước mặt. 
    > *Ghi chú:* Có những kí hiệu có hình dạng bàn tay giống nhau nhưng đặt ở vị trí khác nhau thì có ý nghĩa khác nhau.
* **Chuyển động:** Hướng, chiều, tốc độ và quỹ đạo di chuyển (thẳng, cong, zic-zac) của bàn tay. 
* **Chiều hướng của lòng bàn tay:** Hướng lật của lòng bàn tay và các ngón tay (úp, ngửa, hướng vào trong hoặc hướng ra ngoài).
* **Biểu cảm phi thủ ngữ:** Các thành tố phi cử chỉ tay, bao gồm điệu bộ khuôn mặt (mắt, mày), khẩu hình miệng, sự nghiêng hoặc chuyển động của đầu và phần thân trên. Thành tố này đóng vai trò cốt lõi trong việc định hình cấu trúc cú pháp và biểu đạt ngữ pháp câu.

---

## PHẦN III: PHÂN TÍCH ĐỐI SÁNH NGÔN NGỮ HỌC GIỮA VSL VÀ ASL
Để xây dựng các hệ thống nhận diện tự động, việc áp dụng nguyên bản các mô hình huấn luyện từ Ngôn ngữ Ký hiệu Mỹ (ASL) sang VSL gặp phải rào cản lớn do những khác biệt mang tính bản chất sau:

| Tiêu chí | Ngôn ngữ Ký hiệu Việt Nam (VSL) | Ngôn ngữ Ký hiệu Mỹ (ASL) |
| :--- | :--- | :--- |
| **Dấu thanh** | Có 5 dấu thanh (Sắc, Huyền, Hỏi, Ngã, Nặng) được thể hiện bằng các cử chỉ vẽ quỹ đạo ngắn chuyển động trong không trung. | Không có dấu thanh, vì cấu trúc nền tảng dựa trên chữ cái Latinh không dấu. |
| **Chữ cái** | Có bộ ký hiệu phụ trợ riêng cho các chữ cái đặc hữu: Ă, Â, Đ, Ê, Ô, Ơ, Ư (ví dụ: thêm cử chỉ đội mũ, thêm móc). | Chỉ có 26 chữ cái Latinh cơ bản tĩnh. |
| **Phương thức phối hợp tay** | Sử dụng linh hoạt linh động cả hai tay, phân định rõ vai trò tay chính (thực hiện hành động cốt lõi) và tay phụ (làm bệ đỡ hoặc định vị không gian). | Xu hướng tập trung sử dụng nhiều ở một tay chủ đạo để diễn đạt ý nghĩa. |
| **Cấu trúc ngữ pháp** | Thường có xu hướng bám khá sát trật tự từ của câu tiếng Việt nói (Chủ ngữ + Vị ngữ + Tường thuật). | Sử dụng cấu trúc riêng biệt là Chủ đề - Bình luận (*Topic - Comment*) và biến đổi động từ mạnh bằng hướng tay (*Directional Verbs*). |
| **Khẩu hình** | Khẩu hình miệng và điệu bộ khuôn mặt được dùng liên tục nhằm phân biệt các từ đồng âm hoặc gần nghĩa trong đời sống. | Chủ yếu dùng biểu cảm để nhấn mạnh cảm xúc hoặc xác định loại câu (hỏi, mệnh lệnh). |
| **Tính đồng nhất địa phương** | Phân hóa mạnh theo vùng miền. Việt Nam có 3 nhóm vùng miền lớn: Hà Nội - Hải Phòng, TP.HCM, và Bình Dương. Sự trùng khớp từ vựng giữa các miền chỉ đạt khoảng 50-60%. | Có tính thống nhất tương đối cao và đồng đều trên toàn lãnh thổ nước Mỹ. |

---

## PHẦN IV: CẤU TRÚC TỪ GHÉP CÀ CƠ CHẾ BIỂU DIỄN CẤP CÂU

### 1. Cơ chế cấu tạo Từ ghép
Do VSL ưu tiên tính trực quan, khi biểu đạt một khái niệm trừu tượng không có hình thái rõ ràng, ngôn ngữ này vận dụng quy tắc mượn khái niệm trực quan tương đương bằng cách ghép các ký hiệu đơn lẻ có hình thái cụ thể. 
* *Ví dụ:* Khái niệm "Học bổng" được cấu thành bởi ký hiệu "Học" (hành động mô phỏng mở sách) kết hợp với ký hiệu "Tiền/Nhận" (ngón cái và ngón trỏ xoa vào nhau hoặc hai tay bưng đưa về phía trước). 
* Tương tự: "Trường học" được ghép từ `[Học] + [Nhà]`; "Học sinh" gồm `[Học] + [Người]`. 

Khi các ký hiệu đơn lẻ kết hợp thành một từ ghép, dưới góc độ sinh học vận động, chúng không giữ nguyên 100% hình thái ban đầu mà trải qua 3 hiện tượng biến đổi kỹ thuật sau:
* **Sự rút gọn biên độ:** Tốc độ thực hiện ký hiệu thành phần đầu tiên được đẩy nhanh, biên độ chuyển động thu hẹp để nhường chỗ cho hành động tiếp theo; hành động đầu tiên vừa kịp định hình đã nhanh chóng biến đổi sang tư thế của từ thứ hai.
* **Quỹ đạo chuyển động mượt mà:** Chuyển động của tay giữa các ký hiệu tuân theo một đường cong chuyển tiếp mượt mà, tránh hiện tượng đứt gãy cơ học. Đây là vùng dữ liệu thách thức nhất đối với các thuật toán phân đoạn thời gian tự động (*Temporal Segmentation*).
* **Trọng âm thị giác:** Trọng tâm ngữ nghĩa thường được nhấn mạnh vào hành động thứ hai, biểu hiện bằng việc thực hiện dứt khoát hơn, biên độ rõ ràng hơn hoặc kết hợp cử chỉ gật đầu khẳng định giá trị.

### 2. Biểu diễn cấp câu và ngữ cảnh
VSL có khả năng nén thông tin cao nhờ vào các “Ký hiệu bao hàm”, cho phép diễn đạt các cấu trúc câu phức tạp thông qua một hành động tích hợp duy nhất trong không gian 3D. Ngữ pháp cấp câu phụ thuộc nặng nề vào các biểu cảm phi thủ ngữ để định hình loại câu thay thế cho trợ từ:
* **Câu khẳng định:** Thực hiện chuỗi ký hiệu thông thường kết hợp một cái gật đầu nhẹ ở cuối câu để xác nhận thông tin.
* **Câu nghi vấn:** Khi kết thúc câu, người ký sẽ nhướn mày và hơi nghiêng đầu về phía trước để biểu thị ý định hỏi (thay thế cho từ "chưa?", "không?" trong tiếng nói).
* **Câu phủ định:** Người ký có thể thực hiện động từ kèm theo hành động lắc đầu đồng thời. Hành động lắc đầu tích hợp trực tiếp vào ký hiệu để mang ý nghĩa phủ định (ví dụ: Ký hiệu `[Ăn] + [Lắc đầu]` nghĩa là "Không ăn", không nhất thiết phải tách riêng từ "Không" và từ "Ăn").

> **Lưu ý:** Một ký hiệu tay có thể mang nhiều tầng ý nghĩa khác nhau. Ý nghĩa chính xác của ký hiệu hoàn toàn phụ thuộc vào vị trí đặt tay, biểu cảm đi kèm và ngữ cảnh văn hóa sinh hoạt cụ thể của cuộc trò chuyện.

---

## PHẦN V: KHÓ KHĂN KHI TỔNG HỢP DỮ LIỆU VSL
* **Thiếu hụt dữ liệu chuẩn hóa:** Mặc dù quy mô cộng đồng lớn, Việt Nam vẫn thiếu một bộ cơ sở dữ liệu từ điển số hóa toàn diện được thống nhất ở cấp quốc gia; các tập dữ liệu hiện tại chỉ dao động khoảng 2000 đến 4000 từ (*Gloss*) và gặp bài toán lớn về chi phí lưu trữ, vận hành máy chủ hệ thống.
* **Rào cản địa phương:** Do chưa có sự thống nhất hoàn toàn mang tính quốc gia, các vùng miền vẫn tồn tại các từ mang tính địa phương (thủ ngữ địa phương). Tuy nhiên, nhờ quá trình giao lưu ngôn ngữ giữa các cộng đồng Điếc Nam - Bắc, người Điếc vẫn có khả năng hiểu được các biến thể ký hiệu cơ bản của nhau (ví dụ: dùng chung bộ ký hiệu mũ của miền Nam thì miền Bắc vẫn có thể hiểu được).
* **Sự đa dạng về phong cách cá nhân:** Tốc độ thực hiện ký hiệu, biên độ rộng/hẹp của tay và sắc thái biểu cảm phụ thuộc rất lớn vào tính cách, thói quen giao tiếp cá nhân cũng như cảm xúc nhất thời của người nói tại thời điểm đó.
* **Nhận diện đặc trưng tinh vi:** Việc bắt được các chuyển động cực nhỏ để tạo dấu thanh và các chữ cái có mũ/móc đòi hỏi mô hình AI có độ chính xác rất cao.
* **Hiện tượng che khuất:** Trong quá trình ký tự nhiên, các hiện tượng tay che khuôn mặt hoặc hai tay chồng lấp lên nhau thường xuyên xảy ra, gây mất dấu các điểm mốc hình thể (*Keypoints*) khi trích xuất đặc trưng. 

---

## PHẦN VI: TÀI LIỆU THAM KHẢO
* **[1]** Trần Thị Thiệp. *Giáo trình ngôn ngữ ký hiệu thực hành*. NXB Đại học Sư Phạm, 2016.
* **[2]** Vương Hồng Tâm. *Đặc điểm biểu đạt ngôn ngữ ký hiệu của người điếc Việt Nam*. Tạp chí Khoa học Giáo dục Việt Nam, 2011.
* **[3]** Bộ Giáo dục và Đào tạo Việt Nam. *Từ điển Ngôn ngữ Ký hiệu Việt Nam dùng cho Giáo dục Tiểu học*. NXB Giáo dục Việt Nam, 2018.