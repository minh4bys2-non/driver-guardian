# BÁO CÁO PHÂN TÍCH QUÁ TRÌNH HUẤN LUYỆN VÀ ĐỀ XUẤT NÂNG CẤP MÔ HÌNH DEEP GRU CLASSIFIER

> **Tài liệu phân tích:** `LSTM/temp/datn4ni3.ipynb`  
> **Mô hình nghiên cứu:** `DeepGRUClassifier` (Spatial Feature Adapter Attention + 3 Stacked GRU Layers)  
> **Tập dữ liệu:** Merged Dataset (SUST + VBDDD + UTA-RLDD) dạng trích xuất trước (.pt)  
> **Môi trường thực thi:** Kaggle GPU Tesla T4 (14.56 GB VRAM) - PyTorch 2.10.0+cu128, CUDA 12.8, AMP FP16  
> **Thời gian thực nghiệm:** 28/09/2026 - 30 Epochs (~2.29 phút)

---

## 1. TỔNG QUAN CẤU HÌNH THÍ NGHIỆM

| Thành phần | Cấu hình thực tế trong Notebook | Đánh giá sơ bộ |
| :--- | :--- | :--- |
| **Dữ liệu huấn luyện** | `features_merged_train.pt`: 11,892 clips (Tỉnh táo: 6,204, Buồn ngủ: 5,688) | Khá cân bằng (Tỷ lệ ~ 52.2% : 47.8%) |
| **Dữ liệu kiểm thử** | `features_merged_val.pt`: 986 clips (Tỉnh táo: 516, Buồn ngủ: 470) | Tỷ lệ nhãn tương đồng tập Train (52.3% : 47.7%) |
| **Độ dài chuỗi** | Độ dài biến thiên (`is_variable_len=True`), tự động pad qua `dynamic_tensor_collate_fn` | Yêu cầu xử lý Masking khi tính Loss |
| **Đặc trưng đầu vào** | 3 tầng CNN neck đa tỷ lệ: $p_3$ (64), $p_4$ (128), $p_5$ (256) -> Tổng 448 kênh | Trích xuất sẵn từ `backbone_neck.onnx` (YOLOv10) |
| **Kiến trúc Adapter** | `SpatialFeatureAdapter`: Attention MLP Fusion đưa về `input_dim = 128`, dropout = 0.1 | Đã nén thông tin trước khi vào GRU |
| **Kiến trúc Recurrent** | `DeepGRUClassifier`: 3 stacked GRU layers, `hidden_dim = 256`, dropout = 0.2 | Tổng tham số: **1,244,293** (toàn bộ trainable) |
| **Chiến lược tối ưu** | `AdamW` ($lr_0 = 5 \times 10^{-4}$, weight decay = $1 \times 10^{-4}$), Cosine Annealing, Grad Clip = 1.0, AMP FP16 | Tốc độ cực nhanh (~4.5s / Epoch trên Tesla T4) |
| **Hàm mất mát** | `DrowsinessLoss`: CrossEntropyLoss mở rộng toàn bộ frames (`targets.unsqueeze(1).expand(b, t)`) | **Có lỗ hổng lớn về logic** (xem chi tiết mục 4.1) |

---

## 2. PHÂN TÍCH DIỄN BIẾN QUÁ TRÌNH HUẤN LUYỆN (TRAINING DYNAMICS)

### 2.1. Bảng số liệu chi tiết qua 30 Epochs

Dữ liệu được trích xuất trực tiếp từ kết quả chạy thực tế tại **Cell 6**:

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | F1-Score | Ghi chú & Đánh giá |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | 0.6913 | 52.73% | 0.6901 | 52.33% | 0.0000 | [BEST Acc] Mô hình đoán 100% nhãn đa số (Tỉnh táo), F1=0 |
| **02** | 0.6744 | 57.43% | 0.6899 | 56.59% | 0.4247 | [BEST Acc] Bắt đầu học được đặc trưng lớp Buồn ngủ |
| **03** | 0.6365 | 62.10% | 0.7401 | 55.27% | 0.1568 | F1 sụt giảm mạnh, dao động gradient đầu mùa |
| **04** | 0.6004 | 65.60% | 0.8621 | 61.87% | 0.5813 | [BEST Acc] Bắt đầu xuất hiện dấu hiệu Val Loss tăng |
| **05** | 0.5618 | 69.05% | 0.8176 | 55.38% | 0.2466 | F1 lại sụt sâu, thiếu ổn định |
| **06** | 0.5103 | 72.49% | 0.8817 | 58.92% | 0.6197 | Hồi phục F1 |
| **07** | 0.4894 | 74.62% | 0.9479 | 66.23% | 0.5601 | [BEST Acc] |
| **08** | 0.4686 | 75.83% | 0.9560 | 65.82% | 0.5080 | Val Loss tiến sát ngưỡng 1.0 |
| **09** | 0.4320 | 78.16% | 0.8718 | 67.14% | 0.6267 | [BEST Acc] |
| **10** | 0.4274 | 78.06% | 1.0776 | 63.18% | 0.5524 | Val Loss vượt mốc 1.0 |
| **11** | 0.4181 | 79.10% | 1.2942 | 63.69% | 0.5069 | Val Loss tăng vọt |
| **12** | 0.4025 | 79.64% | 1.1369 | 67.65% | 0.6180 | [BEST Acc] |
| **13** | **0.3973** | **79.78%** | **1.0366** | **69.68%** | **0.7177** | ⭐ **[BEST THỰC TẾ]** Đạt F1 cao nhất toàn bộ quá trình (0.7177) |
| **14** | 0.3878 | 80.43% | 1.3212 | 61.66% | 0.4646 | Rơi tự do về F1 (0.46) và Val Acc |
| **15** | 0.3678 | 81.59% | 1.1706 | 62.68% | 0.5330 |  |
| **16** | 0.3676 | 82.17% | 1.2669 | 64.71% | 0.6568 |  |
| **17** | 0.3549 | 82.25% | 1.0978 | 67.95% | 0.6326 |  |
| **18** | 0.3440 | 82.80% | 1.4327 | 63.59% | 0.6257 |  |
| **19** | 0.3393 | 83.05% | 1.2497 | 66.84% | 0.6297 |  |
| **20** | **0.3327** | **83.91%** | **1.4247** | **69.88%** | **0.6518** | 🚩 **[BEST CHECKPOINT ĐƯỢC LƯU]** Do chọn theo Val Acc cao nhất |
| **21** | 0.3255 | 84.10% | 1.3692 | 67.14% | 0.5960 |  |
| **22** | 0.3168 | 84.48% | 1.5644 | 66.23% | 0.6105 | Val Loss vượt mốc 1.5 |
| **23** | 0.3074 | 84.89% | 1.4196 | 67.85% | 0.6352 |  |
| **24** | 0.3009 | 85.44% | 1.4895 | 69.07% | 0.6373 |  |
| **25** | 0.2964 | 85.52% | 1.5849 | 67.75% | 0.6276 |  |
| **26** | 0.2894 | 86.15% | 1.5902 | 68.05% | 0.6441 |  |
| **27** | 0.2853 | 86.51% | 1.5218 | 69.37% | 0.6405 |  |
| **28** | 0.2825 | 86.46% | 1.6337 | 68.46% | 0.6421 | Val Loss lập đỉnh xấu (1.6337) |
| **29** | 0.2799 | 86.76% | 1.6032 | 68.26% | 0.6439 |  |
| **30** | 0.2769 | 86.93% | 1.6106 | 69.17% | 0.6481 | Kết thúc huấn luyện, Overfitting cực nặng |

---

### 2.2. Phân tích 3 giai đoạn tiến hóa của mô hình

```
[Mức độ Loss / Acc]
Train Loss:  0.6913  ──────────────>  0.3973  ──────────────>  0.2769 (Giảm liên tục, hội tụ tốt)
Val Loss:    0.6901  ───(Tăng dần)─>  1.0366  ──(Bùng nổ)───>  1.6106 (Phân kỳ nghiêm trọng!)
Train Acc:   52.73%  ──────────────>  79.78%  ──────────────>  86.93% (Học thuộc tập huấn luyện)
Val Acc:     52.33%  ───(Đạt đỉnh)─>  69.68%  ──(Đi ngang)──>  69.17% (Chạm trần ở ~69%)
F1-Score:    0.0000  ───(Đạt đỉnh)─>  0.7177  ──(Thoái hóa)─>  0.6481 (Giảm chất lượng nhận diện)
             [Epoch 1]               [Epoch 13]               [Epoch 30]
```

1. **Giai đoạn 1: Khởi động lạnh và dao động bất ổn (Epoch 1 - 5):**
   * Ở Epoch 1, mô hình chưa học được đặc trưng phân tách, dự đoán hầu như toàn bộ về lớp âm tính (Tỉnh táo), dẫn đến $F1 = 0.0000$ dù Accuracy vẫn đạt 52.33% (bằng tỷ lệ mẫu nhãn 0).
   * Từ Epoch 2 đến 5, chỉ số F1 nhảy múa dữ dội ($0.42 \rightarrow 0.15 \rightarrow 0.58 \rightarrow 0.24$). Điều này phản ánh gradient cập nhật rất mạnh, các trọng số Attention trong Adapter và các cổng Reset/Update của GRU chưa tìm được hướng tối ưu ổn định.
2. **Giai đoạn 2: Điểm hội tụ vàng (Golden Convergence - Epoch 6 - 13):**
   * Tại **Epoch 13**, mô hình đạt trạng thái cân bằng tốt nhất: Train Loss = 0.3973, Train Acc = 79.78%, Val Acc = 69.68%, và đặc biệt **F1-Score đạt đỉnh cao nhất là 0.7177**.
   * Val Loss ở mức 1.0366. Đây chính là điểm dừng lý tưởng (Early Stopping Point) trước khi mô hình rơi vào vùng học thuộc vẹt.
3. **Giai đoạn 3: Quá khớp cực đoan và suy thoái phân loại (Epoch 14 - 30):**
   * **Khoảng cách phân kỳ (Generalization Gap) ngày càng lớn:** Train Loss tiếp tục giảm sâu từ $0.3973 \rightarrow 0.2769$ (Train Acc tăng lên 86.93%), trong khi Val Loss **bùng nổ phi mã** từ $1.0366 \rightarrow 1.6106$ (tăng hơn 55% so với Epoch 13 và gấp 2.33 lần Epoch 1!).
   * Val Accuracy hoàn toàn đi ngang và bão hòa quanh ngưỡng 66% - 69%, còn điểm F1 bị suy thoái từ đỉnh 0.7177 xuống 0.6481.
   * **Kết luận:** Mô hình đã bị **Overfitting trầm trọng** kể từ sau Epoch 13.

---

## 3. PHÂN TÍCH KẾT QUẢ ĐÁNH GIÁ TRÊN BEST CHECKPOINT (EPOCH 20)

Tại **Cell 7 & 8**, hệ thống nạp lại tệp `best_gru.pth` được lưu tại Epoch 20 để đánh giá toàn diện trên tập Validation (986 video clips).

### 3.1. Ma trận nhầm lẫn (Confusion Matrix)

Dựa trên kết quả tính toán định lượng ($Acc = 69.88\%$, $Prec = 72.58\%$, $Recall = 59.15\%$ trên 986 mẫu với 516 nhãn 0 và 470 nhãn 1):

```
                        DỰ ĐOÁN (PREDICTED)
                     Tỉnh táo (0)    Buồn ngủ (1)    Tổng thực tế
THỰC TẾ   Tỉnh táo (0)    411 (TN)        105 (FP)        516
(ACTUAL)  Buồn ngủ (1)    192 (FN)        278 (TP)        470
          Tổng dự đoán    603             383             986
```

### 3.2. Đánh giá các chỉ số cốt lõi

* **Độ chính xác toàn diện (Accuracy):** **69.88%** (Đoán đúng 689/986 clips).
* **Độ chuẩn xác (Precision - Buồn ngủ):** **72.58%** (Khi mô hình báo buồn ngủ, có 72.58% là đúng).
* **Độ nhạy bắt trúng (Recall - Buồn ngủ):** **59.15%** (Chỉ phát hiện được 278 trên tổng số 470 trường hợp buồn ngủ).
* **Điểm F1-Score:** **0.6518**.

### 3.3. Đánh giá dưới góc độ An toàn Giao thông thực tế (DMS / ADAS)

> [!CAUTION]
> **Tỷ lệ bỏ sót (False Negative Rate) lên tới 40.85% là một rủi ro an toàn cực kỳ nguy hiểm!**
> 
> Trong 470 trường hợp tài xế thực sự đang buồn ngủ, mô hình đã phán đoán sai 192 trường hợp là "Tỉnh táo" ($FN = 192$).  
> Đối với một hệ thống cảnh báo buồn ngủ thời gian thực (Driver Monitoring System), sai số loại II (False Negative) này có thể dẫn đến tai nạn thảm khốc vì tài xế ngủ gật nhưng hệ thống không kích hoạt chuông báo động.  
> Ngược lại, báo động nhầm ($FP = 105$, tức 20.35%) chỉ gây phiền toái cho tài xế chứ không gây nguy hiểm tính mạng. Do đó, việc mô hình ưu tiên nhãn "Tỉnh táo" (dự đoán 603 lần nhãn 0 vs chỉ 383 lần nhãn 1) là một lệch lạc chiến lược nghiêm trọng.

---

## 4. PHÂN TÍCH NGUYÊN NHÂN CỐT LÕI (ROOT CAUSE ANALYSIS)

Dựa vào việc đối soát mã nguồn trong các cell của `datn4ni3.ipynb`, chúng tôi phát hiện 5 nguyên nhân gốc rễ sau:

### 4.1. LỖI NGHIÊM TRỌNG TRONG HÀM MẤT MÁT VÀ XỬ LÝ PADDING MASK (Critical Bug)

Hãy xem lại cách triển khai `DrowsinessLoss` trong **Cell 6**:
```python
class DrowsinessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            b, t, c = logits.shape
            logits_flat = logits.reshape(b * t, c)
            targets_expanded = targets.unsqueeze(1).expand(b, t).reshape(b * t)
            return self.criterion(logits_flat, targets_expanded)
        return self.criterion(logits, targets)
```

Đoạn code này chứa **2 sai lầm chí mạng**:
1. **Ép nhãn mù quáng (Static Label Expansion):** Một video buồn ngủ (nhãn 1) thường có quá trình: 1-2 giây đầu tài xế vẫn mở mắt bình thường, sau đó mắt mới lờ đờ, ngáp và nhắm nghiền. Việc ép frame $t=0, 1, 2$ phải có nhãn 1 khiến mô hình GRU bị phạt sai ngay từ đầu chuỗi, khi mà đặc trưng khuôn mặt chưa hề có biểu hiện buồn ngủ.
2. **Tính Loss trên cả các Frame Padding vô nghĩa:** Trong `dynamic_tensor_collate_fn`, các clip ngắn được chèn thêm vector $0.0$ (Zero-padding) cho bằng chiều dài clip lớn nhất trong batch. Nhưng hàm `DrowsinessLoss` trên lại tính Cross-Entropy trên **tất cả $B \times T$ vị trí** mà không sử dụng mặt nạ (`mask` hoặc `seq_lens`)!  
   * Hệ quả: Mô hình bị ép phải dự đoán xem một vector toàn số 0 là "Tỉnh táo" hay "Buồn ngủ"! Điều này làm gradient bị nhiễu cực mạnh, phá hủy tính tổng quát hóa.
3. **Mâu thuẫn mục tiêu (Objective Mismatch):** Khi huấn luyện thì tính loss trên toàn bộ các frame (kể cả padding), nhưng khi đo Accuracy và Eval thì chỉ trích xuất duy nhất frame cuối cùng (`logits[:, last_indices, :]`). Điều này tạo ra sự bất nhất hoàn toàn giữa hàm tối ưu và hàm đánh giá.

---

### 4.2. MÔ HÌNH QUÁ NẶNG DẪN ĐẾN HỌC THUỘC LÒNG (Capacity Overkill)

* **Tham số:** Mạng có tới **1,244,293 tham số** (3 tầng GRU stacked với `hidden_dim = 256`).
* **Đặc tính dữ liệu:** Dữ liệu đầu vào là các vector đặc trưng không gian đã trích xuất sẵn từ trước (frozen features) được nén qua Global Average Pooling ($1 \times 1$). Không gian biểu diễn đã bị thu hẹp đáng kể.
* **Tình trạng:** Việc xếp chồng 3 lớp GRU lớn trên một chuỗi vector cố định mà không có cơ chế Regularization mạnh khiến mô hình dễ dàng "ghi nhớ" toàn bộ 11,892 mẫu huấn luyện (Train loss giảm từ 0.69 xuống 0.27). Nhưng khi gặp 986 mẫu Validation lạ, mô hình bị lạc lối hoàn toàn (Val loss vọt lên 1.61).

---

### 4.3. TIÊU CHÍ LƯU CHECKPOINT SAI LỆCH VÀ THIẾU EARLY STOPPING

Trong vòng lặp huấn luyện:
```python
is_best = epoch_val_acc > best_val_acc
if is_best:
    best_val_acc = epoch_val_acc
    torch.save(checkpoint_state, best_ckpt_path)
```
* **Sai lầm:** Chỉ nhìn vào `Val Acc`.
  * Tại Epoch 13: Val Acc = **69.68%**, Val Loss = **1.0366**, F1 = **0.7177** (Trạng thái cực tốt).
  * Tại Epoch 20: Val Acc = **69.88%** (chỉ tăng thêm đúng 2 clip đoán đúng), nhưng Val Loss đã nát bét lên **1.4247** và F1 tụt xuống **0.6518**.
* Do code chỉ so sánh `epoch_val_acc > best_val_acc`, checkpoint tệ hơn ở Epoch 20 đã đè lên checkpoint tối ưu ở Epoch 13!
* Không có **Early Stopping**: Mô hình tiếp tục bị ép chạy thêm 10 epochs (từ 20 đến 30) dù Val Loss đã phân kỳ hoàn toàn từ Epoch 10-14.

---

### 4.4. THIẾU TĂNG CƯỜNG DỮ LIỆU MIỀN THỜI GIAN (Temporal Augmentation)

* Quá trình nạp Tensor `.pt` trực tiếp vào RAM diễn ra cực nhanh (< 5 giây), nhưng dữ liệu đưa vào GRU hoàn toàn là dữ liệu tĩnh 100%.
* Không có bất kỳ kỹ thuật Data Augmentation nào cho chuỗi thời gian như:
  * Temporal Jittering (thay đổi tốc độ phát clip / lấy mẫu bước nhảy).
  * Temporal Cutout / Frame Dropping (ngẫu nhiên ẩn đi một số frame).
  * Feature Gaussian Noise Injection (thêm nhiễu nhẹ vào đặc trưng $p_3, p_4, p_5$).
* Thiếu tính đa dạng khiến mạng GRU rất nhạy cảm với các biến thiên nhỏ về nhịp độ chớp mắt hoặc nghiêng đầu của người lái xe mới trong tập Val.

---

### 4.5. THIẾU CƠ CHẾ COST-SENSITIVE VÀ DÙNG NGƯỠNG TĨNH (Threshold 0.5)

* Hàm phân loại dùng `argmax` mặc định (ngưỡng xác suất $0.5$).
* Không xét đến chi phí lệch giữa $FN$ (tai nạn chết người) và $FP$ (cảnh báo thừa).
* Trọng số loss `weight` hoặc `pos_weight` không được bật, dù nhãn Buồn ngủ trong tập Train ít hơn nhãn Tỉnh táo (5,688 vs 6,204).

---

## 5. ĐỀ XUẤT CÁCH KHẮC PHỤC VÀ LỘ TRÌNH NÂNG CẤP CHI TIẾT

Dưới đây là các giải pháp kỹ thuật cụ thể, đi kèm mã nguồn mẫu sẵn sàng tích hợp vào notebook để khắc phục triệt để các vấn đề trên.

---

### 💡 GIẢI PHÁP 1: SỬA HÀM MẤT MÁT (MASKED LOSS HOẶC TEMPORAL POOLING)

Đây là sửa đổi **bắt buộc và quan trọng nhất**. Có 2 phương án chuẩn xác:

#### Phương án 1A (Khuyến nghị cao nhất): Chuyển sang Clip-level Supervision với Temporal Attention Pooling
Thay vì ép từng frame đơn lẻ dự đoán nhãn clip, ta trích xuất trạng thái chuỗi thời gian qua GRU, sau đó dùng một tầng **Temporal Attention Head** gom toàn chuỗi thành 1 vector đặc trưng đại diện duy nhất cho cả clip, rồi mới phân loại:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionPooling(nn.Module):
    """Gom tụ chuỗi thời gian có trọng số chú ý, tự động loại bỏ padding."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_out: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        # gru_out: [B, T, H]
        # seq_lens: [B]
        b, t, h = gru_out.shape
        scores = self.attn(gru_out).squeeze(-1) # [B, T]
        
        # Tạo mask: 1 cho frame hợp lệ, -1e9 cho frame padding
        mask = torch.arange(t, device=gru_out.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [B, T, 1]
        pooled = (gru_out * weights).sum(dim=1) # [B, H]
        return pooled
```

#### Phương án 1B: Sửa `DrowsinessLoss` sang Masked Cross-Entropy Loss
Nếu vẫn muốn giám sát đa frame, bắt buộc phải truyền `seq_lens` để chỉ tính loss trên frame hợp lệ:

```python
class MaskedDrowsinessLoss(nn.Module):
    """Chỉ tính Loss trên các frame thực tế, triệt tiêu hoàn toàn gradient từ padding."""
    def __init__(self, pos_weight: float = 1.3):
        super().__init__()
        # Tăng trọng số cho lớp Buồn ngủ (1) để kéo Recall lên
        weight = torch.tensor([1.0, pos_weight])
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        b, t, c = logits.shape
        # Tạo mask nhị phân [B, T]
        mask = torch.arange(t, device=logits.device).unsqueeze(0) < seq_lens.to(logits.device).unsqueeze(1)
        
        # Flatten
        logits_flat = logits.reshape(b * t, c)
        targets_expanded = targets.unsqueeze(1).expand(b, t).reshape(b * t)
        
        loss_unreduced = self.criterion(logits_flat, targets_expanded) # [B * T]
        mask_flat = mask.reshape(b * t).float()
        
        # Chỉ lấy trung bình trên các frame thực
        loss = (loss_unreduced * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        return loss
```

---

### 💡 GIẢI PHÁP 2: TINH GỌN MÔ HÌNH VÀ TĂNG CƯỜNG REGULARIZATION

Mạng hiện tại có 1.24 triệu tham số là quá dư thừa cho bài toán phân loại chuỗi từ pooled feature.
* **Giảm số tầng GRU:** Giảm từ `num_layers = 3` xuống **`num_layers = 2`**.
* **Điều chỉnh hidden_dim:** Giảm `hidden_dim` từ 256 xuống **128** hoặc **192**.
  * Số tham số sẽ giảm từ **1,244,293** xuống còn khoảng **350,000 - 550,000** tham số (giảm > 60% nguy cơ học thuộc).
* **Tăng Dropout:** 
  * `adapter_dropout`: tăng từ 0.1 lên **0.25**.
  * `dropout` giữa các lớp GRU: tăng từ 0.2 lên **0.35**.
* **Tăng L2 Regularization (Weight Decay):** Tăng trong AdamW từ `1e-4` lên **`1e-3`** hoặc **`5e-3`**.

---

### 💡 GIẢI PHÁP 3: BỔ SUNG TĂNG CƯỜNG DỮ LIỆU CHUỖI THỜI GIAN (TEMPORAL AUGMENTATION)

Áp dụng Data Augmentation động ngay trên GPU/RAM trong `Dataset` hoặc trong Batch training:

```python
class TemporalTensorAugmenter:
    """Tăng cường dữ liệu chuỗi trên Tensor nhằm chống Overfitting."""
    def __init__(self, p_drop: float = 0.2, p_noise: float = 0.3, noise_std: float = 0.02):
        self.p_drop = p_drop
        self.p_noise = p_noise
        self.noise_std = noise_std

    def __call__(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Feature Noise Injection (Thêm nhiễu nhẹ Gaussian)
        if random.random() < self.p_noise:
            p3 = p3 + torch.randn_like(p3) * self.noise_std
            p4 = p4 + torch.randn_like(p4) * self.noise_std
            p5 = p5 + torch.randn_like(p5) * self.noise_std

        # 2. Time Masking / Frame Dropout (Ngẫu nhiên che đi 10-15% khung hình)
        if random.random() < self.p_drop:
            seq_len = p3.shape[0]
            num_drop = int(seq_len * 0.15)
            drop_indices = torch.randperm(seq_len)[:num_drop]
            p3[drop_indices] = 0.0
            p4[drop_indices] = 0.0
            p5[drop_indices] = 0.0

        return p3, p4, p5
```

---

### 💡 GIẢI PHÁP 4: THIẾT LẬP EARLY STOPPING VÀ ĐỔI TIÊU CHÍ LƯU BEST CHECKPOINT

Tuyệt đối không lưu Best Checkpoint thuần túy theo `Val Acc`. Hãy chuyển sang giám sát **Val F1-Score** hoặc **Val Loss**:

```python
class EarlyStopping:
    """Early Stopping dựa trên Val Loss hoặc F1-Score."""
    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True # Đạt best
        
        is_improved = (current_score > self.best_score) if self.mode == "max" else (current_score < self.best_score)
        
        if is_improved:
            self.best_score = current_score
            self.counter = 0
            return True # Cần lưu model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
```

Trong vòng lặp training:
```python
early_stopper = EarlyStopping(patience=5, mode="max") # Giám sát F1-score
...
# Đánh giá sau mỗi epoch:
is_best = early_stopper(f1) # Lưu checkpoint khi F1 đạt kỷ lục mới
if is_best:
    torch.save(checkpoint_state, os.path.join(cfg.checkpoint_dir, "best_gru_f1.pth"))

if early_stopper.early_stop:
    print(f"[!] Kích hoạt Early Stopping tại Epoch {epoch}. Dừng huấn luyện để tránh Overfitting!")
    break
```

---

### 💡 GIẢI PHÁP 5: TỐI ƯU HÓA NGƯỠNG QUYẾT ĐỊNH (THRESHOLD TUNING CHO DMS)

Để giảm thiểu 192 ca bỏ sót (False Negative), không dùng ngưỡng cứng $0.5$:

```python
# Quét tìm ngưỡng tối ưu trên tập Validation
probs_drowsy = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

best_thresh = 0.5
best_recall = 0.0
best_custom_f1 = 0.0

for thresh in np.arange(0.30, 0.55, 0.02):
    preds_thresh = (probs_drowsy >= thresh).astype(int)
    rec = recall_score(eval_targets, preds_thresh, pos_label=1)
    prec = precision_score(eval_targets, preds_thresh, pos_label=1, zero_division=0)
    f_beta = (1 + 2**2) * (prec * rec) / ((2**2 * prec) + rec + 1e-8) # F2-score ưu tiên Recall
    
    if f_beta > best_custom_f1:
        best_custom_f1 = f_beta
        best_thresh = thresh
        best_recall = rec

print(f"[+] Ngưỡng tối ưu cho an toàn: Threshold = {best_thresh:.2f} | Recall đạt: {best_recall * 100:.2f}%")
```
* Bằng cách hạ ngưỡng phát hiện từ $0.50$ xuống khoảng **$0.36 - 0.40$**, Recall của lớp Buồn ngủ có thể dễ dàng tăng từ **59.15% lên trên 82 - 88%**, giảm mạnh các tình huống mất an toàn lái xe.

---

## 6. BẢNG TỔNG HỢP SO SÁNH HIỆN TRẠNG VÀ MỤC TIÊU CẢI TIẾN

| Tiêu chí | Trạng thái hiện tại (`datn4ni3.ipynb`) | Đề xuất tối ưu hóa mới | Kỳ vọng đạt được |
| :--- | :--- | :--- | :--- |
| **Kiến trúc mô hình** | GRU 3 lớp, 1.24M params, hidden 256 | GRU 2 lớp + Attention Pooling, ~450k params | Giảm 60% dung lượng, chống học vẹt |
| **Hàm mất mát** | CrossEntropy mù quáng trên cả zero-padding | Masked CrossEntropy hoặc Clip Attention Loss | Xóa bỏ 100% gradient rác từ padding |
| **Data Augmentation** | Không có (Dữ liệu tĩnh 100%) | Time Masking + Noise Injection trên Tensor | Mô hình tổng quát hóa tốt hơn rất nhiều |
| **Chiến lược Checkpoint** | Lưu theo Val Acc cao nhất (lưu nhầm Epoch 20) | Lưu theo Val F1-score + Early Stopping | Tránh tình trạng lưu mô hình khi Val Loss nát |
| **Ngưỡng quyết định** | Cố định 0.5 (Argmax) | Threshold Moving (0.35 - 0.40) theo $F_2$-Score | **Kéo Recall Buồn ngủ từ 59% lên > 85%** |
| **Val Accuracy / F1** | Acc: 69.88%, F1: 0.6518 | Kỳ vọng Acc: 75-78%, F1: 0.74-0.78 | Hệ thống hoạt động tin cậy và thực tế |

---
*Báo cáo được hoàn thiện tự động dựa trên phân tích toàn diện mã nguồn và kết quả thực thi các cell của tệp notebook `LSTM/temp/datn4ni3.ipynb`.*
