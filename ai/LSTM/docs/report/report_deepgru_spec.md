# BÁO CÁO NGHIỆM THU: HOÀN TẤT BIÊN SOẠN TÀI LIỆU ĐẶC TẢ KỸ THUẬT MÔ HÌNH DEEPGRUCLASSIFIER VÀ CÁC KHỐI LIÊN QUAN

**Mã tài liệu:** `report_deepgru_spec.md`  
**Dự án:** Hệ Thống Giám Sát và Cảnh Báo Tài Xế Buồn Ngủ (Driver Guardian)  
**Tệp mục tiêu:** [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)  
**Tài liệu đặc tả đã tạo:** [`docs/spec_deepgru_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/spec_deepgru_classifier.md)  
**Căn cứ kế hoạch:** [`docs/plan/plan_deepgru_spec.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_deepgru_spec.md)  
**Căn cứ khảo sát:** [`docs/analsys/analsys_deepgru_spec.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_deepgru_spec.md)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 3: Thực hiện kế hoạch & Báo cáo tổng kết  
**Ngày hoàn tất:** 03/10/2026  

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN

Theo đúng kế hoạch tại [`docs/plan/plan_deepgru_spec.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_deepgru_spec.md) và toàn bộ các chỉ đạo điều chỉnh của người dùng:
1. **Biên soạn hoàn tất tài liệu đặc tả chuẩn hóa:** Đã tạo tệp [`docs/spec_deepgru_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/spec_deepgru_classifier.md) đặc tả chi tiết 3 khối cốt lõi trong [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py): [`CNNAdapter`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L9-L269), [`TemporalAttentionPooling`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L270-L308), và [`DeepGRUClassifier`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L309-L457).
2. **Tuân thủ $100\%$ các ràng buộc nội dung của người dùng:**
   - Đã loại bỏ hoàn toàn các chương mở rộng không cần thiết (Tổng quan chung, API Reference mở rộng, Thống kê FLOPs ngoài lề, Xuất khẩu ONNX).
   - Khối `CNNAdapter`: Đã loại bỏ cơ chế Chunking VRAM và đa dạng hóa giao diện đầu vào.
   - Khối `TemporalAttentionPooling`: Đã loại bỏ nội dung Explainable AI (XAI) / trích xuất weights timeline.
   - Khối `DeepGRUClassifier`: Đã loại bỏ chế độ Sequence / Streaming và các Factory Methods (`from_config`, `from_checkpoint`), tập trung duy nhất vào luồng phân loại Clip-level (Phương án A chuẩn).
   - **Mô tả trực tiếp và chuẩn hóa kiến trúc:** Toàn bộ tài liệu đặc tả và báo cáo tập trung mô tả trực tiếp bản chất phễu tích chập phân tầng và ma trận trọng số $5 \times 5$ học được, không sử dụng các khái niệm gom tụ trung bình toàn cục.
3. **Xác thực tự động qua mã nguồn kiểm thử:** Đã xây dựng và chạy thành công $100\%$ kịch bản kiểm thử độc lập (`verify_spec_models.py`), nghiệm chứng tính chính xác về kích thước tensor, đạo hàm ngược triệt tiêu gradient rác, và độ ổn định trong kiểu dữ liệu FP16.

---

## 2. NỘI DUNG ĐẶC TẢ ĐÃ HOÀN TẤT TRONG [`docs/spec_deepgru_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/spec_deepgru_classifier.md)

### 2.1. Phần 1: Khối `CNNAdapter` (Spatial Feature Adapter)
- **Mục tiêu:** Chuyển đổi và hợp nhất 3 tầng đặc trưng đa tỉ lệ từ Backbone-PAFPN:
  - $p_3 \in \mathbb{R}^{B \times T \times 64 \times 80 \times 80}$ (biên, chi tiết cục bộ).
  - $p_4 \in \mathbb{R}^{B \times T \times 128 \times 40 \times 40}$ (bộ phận mắt, mũi, miệng).
  - $p_5 \in \mathbb{R}^{B \times T \times 256 \times 20 \times 20}$ (ngữ nghĩa toàn thể khuôn mặt).
- **Căn chỉnh không gian:**
  - $p_3 \xrightarrow{\text{MaxPool2d}(k=2, s=2)} 40 \times 40$ (64 kênh).
  - $p_4$ giữ nguyên $40 \times 40$ (128 kênh).
  - $p_5 \xrightarrow{\text{Bilinear Upsample}(\text{scale}=2)} 40 \times 40$ (256 kênh).
  - Nối kênh: Tensor $[N, 448, 40, 40]$ với $N = B \times T$.
- **Phễu tích chập phân tầng 4 giai đoạn (`hierarchical_conv_pyramid`):**
  - **Stage 1:** `Conv2d(448, 512, k=3, s=2, p=1)` $\to$ `BatchNorm2d(512)` $\to$ `SiLU` $\to [N, 512, 20, 20]$ (Học tương quan cục bộ mí mắt, vành môi).
  - **Stage 2:** `Conv2d(512, 512, k=3, s=2, p=1)` $\to$ `BatchNorm2d(512)` $\to$ `SiLU` $\to [N, 512, 10, 10]$ (Học tương quan vùng: khoảng cách hai mắt, tam giác mắt-mũi-miệng).
  - **Stage 3:** `Conv2d(512, 512, k=3, s=2, p=1)` $\to$ `BatchNorm2d(512)` $\to$ `SiLU` $\to [N, 512, 5, 5]$ (Học tương quan toàn thể khuôn mặt và góc nghiêng đầu).
  - **Stage 4:** `Conv2d(512, 256, k=5, s=1, p=0)` $\to$ `BatchNorm2d(256)` $\to$ `SiLU` $\to [N, 256, 1, 1]$ (Ma trận trọng số $5 \times 5$ học được tổng hợp toàn thể).
  - Squeeze & Dropout: `Dropout(0.25)` $\to$ Khôi phục chuỗi thời gian $[B, T, 256]$.

### 2.2. Phần 2: Khối `TemporalAttentionPooling`
- **Công thức Additive Attention (Bahdanau MLP):**
  $$u_{i, t} = W_2 \tanh(W_1 h_{i, t} + b_1) + b_2 \in \mathbb{R}$$
  $$\alpha_{i, t} = \frac{\exp(u_{i, t})}{\sum_{k=1}^T \exp(u_{i, k})} \in [0, 1], \quad c_i = \sum_{t=1}^T \alpha_{i, t} h_{i, t} \in \mathbb{R}^{192}$$
  với $W_1 \in \mathbb{R}^{96 \times 192}$, $b_1 \in \mathbb{R}^{96}$, $W_2 \in \mathbb{R}^{1 \times 96}$, $b_2 \in \mathbb{R}^{1}$.
- **Cơ chế Masking triệt tiêu Zero-Padding:**
  - $\text{mask}_{i, t} = (t < L_i)$.
  - Gán giá trị âm cực hạn tại các khung hình padding ($t \ge L_i$):
    $$u_{i, t} \leftarrow \begin{cases} -10^4 & \text{nếu dtype} = \text{torch.float16} \\ -10^9 & \text{nếu dtype} = \text{torch.float32} \end{cases}$$
  - Dẫn đến $\alpha_{i, t} = 0.0$ tuyệt đối và đạo hàm ngược $\frac{\partial \mathcal{L}}{\partial h_{i, t}} = \mathbf{0.0}$ $\forall t \ge L_i$, triệt tiêu $100\%$ gradient rác.
- **Ổn định số học FP16:** Ngưỡng $-10^4$ nằm an toàn trong dải $[-65504, 65504]$ của half-precision, triệt tiêu nguy cơ tràn số / sinh giá trị NaN trong AMP.

### 2.3. Phần 3: Khối `DeepGRUClassifier`
- **Kiến trúc chuỗi Stacked Deep GRU:**
  - 2 lớp `nn.GRU` xếp chồng (`num_layers=2`), `input_size=256`, `hidden_size=192`, `batch_first=True`.
  - Regularization: `dropout=0.35` giữa lớp 1 và lớp 2 chống quá khớp.
- **Luồng phân loại Clip-level chuẩn hóa:**
  $$(p_3, p_4, p_5) \xrightarrow{\text{CNNAdapter}} [B, T, 256] \xrightarrow{\text{GRU}} [B, T, 192] \xrightarrow{\text{Attention Pooling}} [B, 192] \xrightarrow{\text{Dropout}(0.35) \to \text{Linear}(192, 2)} [B, 2]$$
  (Lớp 0: Tỉnh táo, Lớp 1: Buồn ngủ).

---

## 3. KẾT QUẢ KIỂM THỬ XÁC THỰC (VERIFICATION TESTS)

Đã khởi chạy kịch bản kiểm thử độc lập nghiệm chứng các khẳng định trong đặc tả kỹ thuật:

```text
[1/4] Testing CNNAdapter specification...
    [PASSED] Output shape: torch.Size([2, 4, 256])
[2/4] Testing TemporalAttentionPooling specification & Gradient Zeroing...
    [PASSED] 100% gradient elimination on padded frames verified!
[3/4] Testing FP16 numerical stability...
    [PASSED] FP16 half precision is completely stable (No NaN / Overflow)!
[4/4] Testing DeepGRUClassifier End-to-End Clip-level...
    [PASSED] End-to-End Logits shape: torch.Size([2, 2]), Weights shape: torch.Size([2, 8])

[SUCCESS] All verification tests passed flawlessly!
```

---

## 4. BẢNG ĐỐI CHIẾU TIÊU CHÍ HOÀN THÀNH (DEFINITION OF DONE)

| Tiêu chuẩn nghiệm thu | Trạng thái | Minh chứng thực tế |
| :--- | :---: | :--- |
| **Đặc tả đầy đủ 3 khối cốt lõi** | ĐẠT | [`docs/spec_deepgru_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/spec_deepgru_classifier.md) bao gồm Phần 1, Phần 2, Phần 3 |
| **Loại bỏ các chương ngoài lề (1, 5, 6, 7)** | ĐẠT | Chỉ giữ lại đúng 3 phần trọng tâm tương ứng 3 khối mã nguồn |
| **Loại bỏ Sequence/Streaming & Factory Methods** | ĐẠT | Phần 3 chỉ tập trung duy nhất vào kiến trúc 2-layer GRU và Clip-level flow |
| **Loại bỏ XAI & weights timeline** | ĐẠT | Phần 2 chỉ đặc tả công thức toán học và cơ chế Zero-Padding Masking |
| **Loại bỏ Chunking VRAM & Đa dạng hóa I/O** | ĐẠT | Phần 1 chỉ tập trung vào căn chỉnh không gian và Phễu tích chập phân tầng |
| **Kiến trúc tích chập học được** | ĐẠT | Sử dụng phễu tích chập phân tầng và ma trận trọng số $5 \times 5$ học được |
| **Nghiệm chứng qua kiểm thử PyTorch** | ĐẠT | $4/4$ kiểm thử passed với gradient zeroing và FP16 stability |
| **Tuân thủ quy trình 3 bước AGENTS.md** | ĐẠT | Hoàn thành Bước 1 (`analsys`), Bước 2 (`plan`), Bước 3 (`spec` + `report`) |
