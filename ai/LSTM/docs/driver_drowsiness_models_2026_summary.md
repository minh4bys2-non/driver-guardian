# Tổng hợp kiến trúc và kết quả các mô hình phát hiện buồn ngủ lái xe (2026)

Tài liệu này tổng hợp ba bài báo được cung cấp, tập trung vào **kiến trúc**, **dữ liệu và giao thức đánh giá**, **chỉ số định lượng**, **ablation/robustness/deployment**, cùng **ưu - nhược điểm kỹ thuật**. Các con số được lấy từ bảng và phần kết quả của từng bài; không nên so sánh Accuracy giữa ba bài như cùng một benchmark vì giao thức chia dữ liệu, miền dữ liệu và đầu vào khác nhau đáng kể.

## 1. Light-VTD - Domain-robust Vision Transformer with Hierarchical Swin Encoding

**Nguồn:** `s41598-026-41847-y.pdf`, Scientific Reports (2026), 16:28136.

### 1.1. Bài toán và đầu vào

Light-VTD là mô hình phân loại nhị phân trạng thái buồn ngủ từ **ROI khuôn mặt RGB 224×224**, hoạt động theo **từng frame**, không có mô hình hóa chuỗi thời gian trực tiếp.

Pipeline tổng quát:

```text
Ảnh khuôn mặt 224×224
    ↓
Tiền xử lý: ROI/crop → CLAHE → normalize
    ↓
Fused-MBConv / MobileOne-derived convolutional stem
    ↓
Patch partition + linear projection
    ↓
Hierarchical Swin encoder (4 stages)
    ↓
{T1, T2, T3, T4}
    ↓
Adaptive multi-path token fusion
    ↓
LayerNorm → Global Average Pooling → FC → Softmax
    ↓
Drowsy / Non-drowsy
```

### 1.2. Kiến trúc

#### Convolutional stem

Mục tiêu là giữ lại các cấu trúc cục bộ như biên mí mắt, iris và texture quanh mắt trước khi token hóa:

```math
F_{mob} = BN(Conv_{3×3}(ReLU(Conv_{1×1}(I))))
```

So với patchify trực tiếp từ ảnh, stem CNN cung cấp **local inductive bias** và giảm độ nhạy với nhiễu nền / artefact cảm biến.

#### Hierarchical Swin encoder

Mô hình có 4 stage S1-S4. Qua mỗi stage:

- độ phân giải token giảm bằng PatchMerge;
- số kênh tăng;
- sử dụng Window-based MSA và Shifted Window MSA;
- tránh chi phí attention toàn cục theo bình phương số token.

Patch merging:

```math
P_{merge} = W_{merge}[p_1 || p_2 || p_3 || p_4]
```

#### Adaptive multi-path token fusion

Đầu ra đa tỉ lệ `{T1, T2, T3, T4}` được đưa về cùng kích thước, rồi trộn bằng trọng số học được:

```math
T_{fused} = \sum_i \alpha_i T_i,
\qquad
\alpha_i = \frac{e^{w_i}}{\sum_j e^{w_j}}
```

Ý nghĩa:

- tầng sớm giữ chi tiết quanh mắt;
- tầng sâu mang ngữ cảnh ổn định hơn trước pose/illumination;
- mô hình tự học tỉ lệ đóng góp của từng scale.

#### Classification head

```text
LayerNorm → GlobalAvgPool → Linear → Softmax
```

#### Explainability

Grad-CAM được dùng để kiểm tra vùng mô hình dựa vào khi ra quyết định. Bài báo báo cáo **IoU trung bình > 0.71** giữa heatmap và vùng quanh mắt được chú thích thủ công.

### 1.3. Dữ liệu và chia tập

Bảng phân bổ chi tiết của bài báo liệt kê:

| Dataset | Số ảnh gốc | Train 80% | Validation 5% | Test 15% |
|---|---:|---:|---:|---:|
| MRL Eye | 84,898 | 67,918 | 4,245 | 12,735 |
| nthuDDD2 | 66,521 | 53,217 | 3,327 | 9,977 |
| UTA-RLDD | 11,787 | 9,429 | 590 | 1,768 |
| **Tổng** | **163,206** | **130,564** | **8,162** | **24,480** |

Training set được mở rộng ở ba mức augmentation: **×2, ×4, ×6**; validation và test giữ nguyên.

> **Lưu ý về tính nhất quán của bài báo:** abstract nói "hơn 167,000 ảnh" và "bốn public datasets" nhưng phần phân bổ chi tiết chỉ liệt kê ba dataset trên, tổng 163,206 ảnh gốc. Vì vậy các số liệu dataset trong tài liệu này ưu tiên bảng phân bổ chi tiết.

### 1.4. Tiền xử lý và augmentation

Pipeline có:

- resize / ROI localization / cropping;
- CLAHE;
- normalization;
- curriculum augmentation từ đơn giản đến khó hơn;
- các biến đổi được thảo luận gồm geometric scaling, contrast, blur, occlusion, patch drop, shear, gamma correction.

CLAHE tạo cải thiện nhất quán:

| Dataset | Không CLAHE: Macro-F1 | CLAHE: Macro-F1 | MCC với CLAHE | Macro-F1 dưới illumination stress | Drop |
|---|---:|---:|---:|---:|---:|
| MRL Eye | 97.8±0.4 | **98.6±0.3** | **0.983±0.004** | 96.1±0.5 | 2.5 pp |
| nthuDDD2 | 95.8±0.6 | **97.1±0.5** | **0.976±0.006** | 93.4±0.7 | 3.7 pp |
| UTA-RLDD | 96.4±0.5 | **97.7±0.4** | **0.970±0.006** | 94.6±0.6 | 3.1 pp |

So với không CLAHE, mức giảm Macro-F1 khi stress ánh sáng được giảm khoảng **1.7-2.7 điểm phần trăm**.

### 1.5. Cấu hình huấn luyện

| Hyperparameter | Giá trị được chọn |
|---|---:|
| Input | 224×224 |
| Learning rate | 5×10⁻⁵ |
| Batch size | 24 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Weight decay | 5×10⁻⁴ |
| LR scheduler | OneCycleLR |
| Warm-up | 750 steps |
| Max epochs | 50 |
| Early stopping patience | 5 |

Bài báo còn nêu class-weighted loss, label smoothing và stochastic depth regularization trong protocol huấn luyện.

### 1.6. Kết quả in-domain

Các con số đại diện được bài báo dùng khi so sánh với các nghiên cứu trước:

| Dataset | Accuracy | Ghi chú |
|---|---:|---|
| MRL Eye | **98.7%** | cấu hình tốt được báo cáo trong phần so sánh SOTA |
| nthuDDD2 | **97.2%** | CLAHE / cấu hình tốt |
| UTA-RLDD | **97.6%** | cấu hình tốt |

Một số kết quả chi tiết theo augmentation cho Light-VTD:

| Dataset / augmentation | Accuracy | F1 | PR-AUC | MCC |
|---|---:|---:|---:|---:|
| MRL ×2 | 97.1±1.0 | 96.3±1.1 | 98.7±0.9 | 97.5±0.9 |
| MRL ×4 | **98.7±0.6** | **98.6±0.8** | **98.9±0.5** | **98.3±0.6** |
| MRL ×6 | 98.4±0.7 | 97.8±0.8 | 98.7±0.7 | 97.7±0.7 |
| nthuDDD2 ×2 | 95.4±1.0 | 94.5±1.0 | 98.0±0.7 | dữ liệu bảng bị cắt trong bản text extraction |
| UTA-RLDD ×6 | **97.6±0.7** | **96.6±0.8** | **98.4±0.5** | **96.0±0.5** |

Bài báo cho thấy augmentation không tăng đơn điệu: một số dataset đạt tối ưu ở ×4, trong khi UTA-RLDD hưởng lợi rõ ở ×6.

### 1.7. Cross-dataset generalization

Bảng leave-one-dataset / zero-shot transfer cho thấy Light-VTD thường đứng đầu nhóm baseline Transformer, nhưng không phải mọi cặp train→test đều đạt ≥93%:

| Train | Test | Light-VTD Accuracy |
|---|---|---:|
| MRL ×2 | nthuDDD2 | 92.3±1.0 |
| MRL ×2 | UTA-RLDD | 91.2±1.1 |
| MRL ×4 | nthuDDD2 | **93.7±0.8** |
| MRL ×4 | UTA-RLDD | 92.5±0.9 |
| MRL ×6 | nthuDDD2 | 93.0±0.9 |
| MRL ×6 | UTA-RLDD | 91.9±1.0 |
| nthuDDD2 ×2 | MRL | **94.0±0.9** |
| nthuDDD2 ×2 | UTA-RLDD | 91.7±1.0 |
| nthuDDD2 ×4 | MRL | 93.2±1.0 |
| nthuDDD2 ×4 | UTA-RLDD | 90.9±1.1 |
| nthuDDD2 ×6 | MRL | 92.6±1.1 |
| nthuDDD2 ×6 | UTA-RLDD | 90.3±1.2 |
| UTA-RLDD ×2 | MRL | **94.5±0.8** |
| UTA-RLDD ×2 | nthuDDD2 | 92.4±0.9 |

> **Quan trọng:** abstract/discussion tuyên bố cross-dataset accuracy "ít nhất 93%", nhưng bảng kết quả chi tiết có nhiều trường hợp khoảng **90.3-92.6%**. Khi dùng số liệu trong luận văn/paper, nên trích bảng chi tiết thay vì lặp lại claim tổng quát trong abstract.

### 1.8. Robustness

Cross-domain MRL → nthuDDD2 dưới các perturbation:

| Model | Motion blur MCC / PR-AUC | Occlusion MCC / PR-AUC | Low-light MCC / PR-AUC | Avg. MCC drop |
|---|---:|---:|---:|---:|
| Light-VTD | **0.917 / 0.958** | **0.906 / 0.957** | **0.914 / 0.961** | **3.9%** |
| MPViT | 0.861 / 0.931 | 0.887 / 0.944 | 0.842 / 0.920 | 7.4% |
| MViT v2 | 0.901 / 0.952 | 0.873 / 0.939 | 0.856 / 0.929 | 6.8% |
| TokenLearner | 0.866 / 0.934 | 0.848 / 0.923 | 0.892 / 0.947 | 7.1% |
| RegionViT | 0.896 / 0.950 | 0.854 / 0.927 | 0.871 / 0.937 | 6.8% |

### 1.9. Ablation

| Variant | Macro-F1 | MCC | Params | GFLOPs | Latency |
|---|---:|---:|---:|---:|---:|
| **Full Light-VTD** | **97.8±0.5** | **0.970±0.006** | **8.6 M** | **1.2** | **12.4±0.5 ms** |
| Plain patch embedding thay conv stem | 96.8±0.6 | 0.960±0.008 | 7.9 M | 1.1 | 11.8±0.4 ms |
| Global MHSA thay Swin windows | 97.2±0.7 | 0.964±0.009 | 8.7 M | **3.8** | **27.1±0.8 ms** |
| Bỏ shifted windows | 97.1±0.6 | 0.963±0.007 | 8.6 M | 1.1 | 12.1±0.5 ms |
| **Bỏ multi-path token fusion** | **96.6±0.8** | **0.957±0.010** | 8.2 M | 1.1 | 11.9±0.4 ms |
| Bỏ augmentation curriculum | 97.0±0.7 | 0.962±0.009 | 8.6 M | 1.2 | 12.4±0.5 ms |

Kết luận ablation quan trọng:

- multi-path token fusion là thành phần làm mất hiệu năng nhiều nhất khi loại bỏ;
- conv stem có ích thực sự cho chi tiết quanh mắt;
- global MHSA không cải thiện accuracy nhưng làm FLOPs tăng khoảng 3.2× và latency >2×;
- curriculum augmentation chủ yếu tăng độ bền, không làm thay đổi inference cost.

### 1.10. Hiệu quả triển khai

| Model | GFLOPs | Peak GPU memory | Latency batch=1 | Power |
|---|---:|---:|---:|---:|
| **Light-VTD** | **1.2** | **1.1 GB** | **12.4±0.5 ms** | 115±7 W |
| MPViT | 2.9 | 1.6 GB | 35.6±1.4 ms | 145±8 W |
| MViT v2 | 9.1 | 4.2 GB | 14.8±0.6 ms | 95±6 W |
| TokenLearner ViT | 3.7 | 3.8 GB | 27.8±1.2 ms | 85±5 W |
| RegionViT | 8.6 | 2.0 GB | 22.5±1.0 ms | 175±10 W |

Web interface + Grad-CAM được báo cáo khoảng **~102-120 ms**; Raspberry Pi 4 đạt khoảng **7.9 FPS / <120 ms** theo phần mô tả deployment.

### 1.11. Ưu điểm

- Kiến trúc tương đối cân bằng giữa **local detail + global context + multi-scale fusion**.
- Có kiểm tra **cross-dataset**, robustness, ablation và deployment chứ không chỉ báo Accuracy.
- Windowed attention hiệu quả hơn global MHSA rõ rệt về FLOPs và latency.
- CLAHE và augmentation curriculum có bằng chứng định lượng về tác dụng dưới illumination/occlusion/blur.
- Có explainability với Grad-CAM và đánh giá định lượng IoU.
- Phù hợp làm **visual encoder frame-level** cho hệ thống lớn hơn.

### 1.12. Nhược điểm

- **Không có temporal modeling**: không trực tiếp mô hình hóa blink duration, PERCLOS theo thời gian, head nod progression hay microsleep.
- Chỉ dùng RGB; không tích hợp tín hiệu khác.
- Dataset vẫn thiên về môi trường tương đối kiểm soát, số subject của UTA-RLDD nhỏ.
- Chưa đánh giá đầy đủ glare ban đêm, mưa, bóng tối cực đoan, sensor noise mạnh.
- Grad-CAM chỉ là post-hoc explanation; không cung cấp uncertainty-aware explanation.
- Chưa lượng hóa đầy đủ thermal throttling, camera buffering, concurrent I/O hay battery/low-voltage deployment.
- Có một số **mâu thuẫn nội bộ trong báo cáo số liệu** giữa abstract, bảng dataset và bảng cross-dataset; cần ưu tiên bảng chi tiết khi trích dẫn.

---

## 2. Subject-Independent Temporal Framework - EAR + LR / RF / LSTM / BiLSTM

**Nguồn:** `sensors-26-06193.pdf`, Sensors 2026, 26, 6193.

### 2.1. Mục tiêu chính

Bài này không cố tối đa Accuracy bằng một backbone ảnh lớn. Trọng tâm là kiểm tra:

1. temporal context có thực sự giúp phát hiện buồn ngủ không;
2. mô hình phức tạp như LSTM/BiLSTM có thực sự vượt mô hình cổ điển không;
3. kết quả còn giữ được khi **test driver hoàn toàn chưa từng xuất hiện trong training** hay không.

Đây là điểm rất quan trọng vì random frame split có thể gây **identity leakage**.

### 2.2. Dữ liệu

NTHU-DDD subset:

| Thuộc tính | Giá trị |
|---|---:|
| Drivers | 4: 001, 002, 005, 006 |
| Video streams | 27 |
| Tổng frames | 65,929 |
| Drowsy | 36,023 (54.64%) |
| Non-drowsy | 29,906 (45.36%) |
| Validation | Leave-One-Driver-Out (LODO) |

Phân bố theo người:

| Driver | Non-drowsy | Drowsy | Tổng |
|---|---:|---:|---:|
| 001 | 9,336 | 9,578 | 18,914 |
| 002 | 7,829 | 10,595 | 18,424 |
| 005 | 8,843 | 13,087 | 21,930 |
| 006 | 3,898 | 2,763 | 6,661 |

### 2.3. Feature engineering

MediaPipe Face Mesh được dùng để lấy landmark mắt.

#### Eye Aspect Ratio - EAR

```math
EAR = \frac{||p_2-p_6|| + ||p_3-p_5||}{2||p_1-p_4||}
```

Mỗi frame được ánh xạ thành vector 4 chiều:

```math
x_t = [EAR_t,\; EAR_{RM}(t),\; \Delta EAR_t,\; EC_t]
```

Trong đó:

- `EAR`: độ mở mắt tức thời;
- `Rolling Mean EAR`: trung bình trượt 5 frame;
- `EAR Delta`: thay đổi frame-to-frame;
- `Eye Closure Indicator`: 1 nếu `EAR < 0.21`, ngược lại 0.

Sequence lengths: **T = 10, 20, 30 frames**.

### 2.4. Các mô hình

#### Logistic Regression

Input sequence `T×4` được flatten thành vector `4T`. Đây là linear baseline, rất nhẹ.

#### Random Forest

Cũng dùng input `4T`, với:

- 300 trees;
- `class_weight = balanced`.

RF có thể học các quan hệ phi tuyến và threshold interaction giữa EAR, delta, rolling mean và closure flag.

#### LSTM

```text
[T, 4]
  ↓
LSTM, 64 hidden units
  ↓
Dropout 0.30
  ↓
Dense 32 + ReLU
  ↓
Dropout 0.20
  ↓
Sigmoid
```

#### BiLSTM

Pipeline tương tự LSTM nhưng recurrent representation là bidirectional.

### 2.5. Training

| Tham số | Giá trị |
|---|---:|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | val loss, patience=7, restore best |
| Internal validation | stream-disjoint, ~15% development data |
| Scaling | MinMaxScaler fit chỉ trên internal train |
| LR class weight | balanced |
| RF class weight | balanced |
| LSTM/BiLSTM | unweighted BCE |
| Seeds cho stability | 42, 123, 2026 |

Điểm tốt trong protocol là **overlapping windows từ cùng một stream không được phép xuất hiện đồng thời trong internal train và validation**.

### 2.6. Kết quả đầy đủ theo sequence length

Giá trị là `mean ± std` trên bốn driver LODO.

| T | Model | Accuracy | Precision Drowsy | Recall Drowsy | F1 Drowsy | Macro-F1 | Weighted-F1 | AUROC | AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | BiLSTM | 0.535±0.030 | 0.566±0.078 | 0.647±0.241 | 0.580±0.054 | 0.521±0.033 | 0.521±0.027 | 0.609±0.141 | 0.626±0.038 |
| 10 | LSTM | 0.535±0.036 | 0.563±0.089 | 0.671±0.219 | 0.592±0.044 | 0.520±0.043 | 0.519±0.050 | 0.628±0.132 | 0.630±0.040 |
| 10 | **Logistic Regression** | **0.576±0.072** | 0.608±0.061 | 0.592±0.235 | 0.579±0.096 | **0.570±0.075** | **0.571±0.069** | 0.633±0.122 | 0.629±0.036 |
| 10 | Random Forest | 0.537±0.017 | 0.558±0.061 | 0.674±0.177 | 0.596±0.030 | 0.525±0.010 | 0.527±0.008 | 0.595±0.102 | 0.623±0.052 |
| 20 | BiLSTM | 0.527±0.035 | 0.567±0.067 | 0.608±0.272 | 0.556±0.077 | 0.514±0.030 | 0.511±0.022 | 0.626±0.128 | 0.611±0.024 |
| 20 | LSTM | 0.569±0.086 | **0.614±0.054** | 0.571±0.283 | 0.560±0.126 | 0.557±0.091 | 0.559±0.084 | 0.628±0.129 | 0.623±0.027 |
| 20 | Logistic Regression | 0.574±0.075 | 0.609±0.065 | 0.589±0.241 | 0.577±0.100 | 0.567±0.078 | 0.569±0.071 | 0.641±0.130 | 0.641±0.044 |
| 20 | Random Forest | 0.549±0.012 | 0.571±0.066 | 0.688±0.175 | 0.609±0.026 | 0.537±0.007 | 0.539±0.012 | 0.614±0.106 | 0.646±0.059 |
| 30 | BiLSTM | 0.504±0.028 | 0.556±0.091 | 0.623±0.263 | 0.554±0.050 | 0.482±0.062 | 0.475±0.079 | 0.645±0.139 | 0.652±0.056 |
| 30 | LSTM | 0.502±0.039 | 0.557±0.096 | 0.624±0.261 | 0.554±0.044 | 0.475±0.084 | 0.467±0.103 | 0.643±0.136 | 0.646±0.050 |
| 30 | Logistic Regression | 0.574±0.076 | **0.612±0.071** | 0.591±0.247 | 0.578±0.102 | 0.567±0.078 | 0.568±0.072 | **0.647±0.135** | 0.651±0.050 |
| 30 | **Random Forest** | 0.556±0.012 | 0.577±0.069 | **0.701±0.173** | **0.618±0.025** | 0.542±0.010 | 0.544±0.019 | 0.626±0.107 | **0.663±0.062** |

### 2.7. Kết luận từ bảng chính

- **Accuracy cao nhất:** Logistic Regression, T=10, `0.576±0.072`.
- **Recall drowsy cao nhất:** Random Forest, T=30, `0.701±0.173`.
- **F1 drowsy cao nhất:** Random Forest, T=30, `0.618±0.025`.
- **AUPRC cao nhất:** Random Forest, T=30, `0.663±0.062`.
- LSTM/BiLSTM **không vượt** mô hình cổ điển một cách nhất quán.

Điều này không chứng minh LSTM kém hơn RF nói chung; nó cho thấy với **chỉ 4 engineered features và 4 subjects**, capacity của recurrent network không được chuyển hóa thành generalization tốt hơn.

### 2.8. Temporal context quan trọng hơn engineered features

Controlled RF baseline:

| Cấu hình | Accuracy | F1 Drowsy | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| B1: 1 frame, raw EAR | 0.514±0.012 | 0.531±0.013 | 0.532±0.030 | 0.547±0.061 |
| B2: 1 frame, 4 engineered features | 0.529±0.017 | 0.573±0.019 | 0.561±0.059 | 0.576±0.039 |
| B3: 30 frames, raw EAR | 0.553±0.012 | **0.618±0.021** | 0.624±0.108 | **0.663±0.064** |
| B4: 30 frames, 4 engineered features | **0.556±0.012** | **0.618±0.025** | **0.626±0.107** | **0.663±0.062** |

Cơ chế rút ra rất rõ:

- từ B1 → B3, chỉ thêm temporal context đã tăng F1 `0.531 → 0.618` và AUPRC `0.547 → 0.663`;
- từ B3 → B4, thêm engineered features gần như không còn cải thiện F1/AUPRC.

=> **Thông tin thời gian là nguồn cải thiện chính; feature engineering chỉ hữu ích rõ khi temporal context còn thiếu.**

### 2.9. Subject-wise RF @ T=30

| Driver test | Accuracy | Always-drowsy baseline | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001 | 0.564 | 0.513 | 0.559 | 0.704 | 0.623 | 0.589 | 0.592 |
| 002 | 0.538 | 0.582 | 0.614 | 0.557 | 0.584 | 0.540 | 0.632 |
| 005 | 0.560 | 0.603 | 0.645 | 0.599 | 0.622 | 0.594 | 0.696 |
| 006 | 0.561 | 0.420 | 0.488 | **0.943** | **0.644** | **0.783** | **0.731** |

Inter-driver variance rất lớn: recall thay đổi từ **0.557 đến 0.943**. Đây là bằng chứng trực tiếp rằng EAR behavior mang tính cá nhân cao.

### 2.10. Confusion matrix - RF @ T=30

Gộp dự đoán từ các LODO fold:

- True Drowsy: **23,092**
- Drowsy → Non-drowsy: **12,931**
- True Non-drowsy: **13,064**
- Non-drowsy → Drowsy: **16,059**

Tức số false positive và false negative vẫn lớn; mô hình chưa đạt mức deployment-ready.

### 2.11. Threshold sensitivity

RF @ T=30 gần như không nhạy với EAR threshold trong khoảng 0.19-0.23:

| EAR threshold | Accuracy | F1 Drowsy | AUROC | AUPRC |
|---:|---:|---:|---:|---:|
| 0.19 | 0.557±0.013 | 0.619±0.026 | 0.626±0.109 | 0.663±0.063 |
| 0.20 | 0.556±0.013 | 0.617±0.027 | 0.627±0.107 | 0.663±0.063 |
| 0.21 | 0.556±0.012 | 0.618±0.025 | 0.626±0.107 | 0.663±0.062 |
| 0.22 | 0.556±0.011 | 0.618±0.025 | 0.625±0.106 | 0.662±0.062 |
| 0.23 | 0.557±0.012 | 0.618±0.026 | 0.625±0.106 | 0.662±0.062 |

### 2.12. Ưu điểm

- **LODO subject-independent**: đánh giá thực tế hơn random frame split.
- Protocol kiểm soát leakage khá chặt: subject-disjoint test, stream-disjoint internal validation, scaler fit chỉ trên train.
- Ablation B1-B4 tách được tác dụng của temporal context khỏi feature engineering.
- Chỉ ra rõ rằng **model complexity không tự động đồng nghĩa với generalization**.
- RF/LR rất rẻ, dễ chạy real-time trên thiết bị hạn chế.
- Phù hợp làm **behavioral temporal branch** song song với CNN/landmark/deep feature branch.

### 2.13. Nhược điểm

- Chỉ **4 driver độc lập** - đây là bottleneck lớn nhất. Nhiều overlapping windows không làm tăng số subject độc lập.
- Chỉ 4 feature quanh mắt; bỏ qua miệng/yawn, head pose, facial tension, deep visual representation.
- Accuracy tuyệt đối thấp (~50-58%) dưới LODO.
- Fixed EAR threshold vẫn có thể không phù hợp cho từng cá nhân dù sensitivity 0.19-0.23 khá ổn trên tập này.
- Chưa external validation trên dataset độc lập.
- LSTM/BiLSTM có nhiều tham số hơn nhưng dữ liệu subject-level quá ít để học invariant temporal representation.

---

## 3. GDAN / GDAU - Guided Dual-Attention Networks

**Nguồn:** `technologies-14-00544-v2.pdf`, Technologies 2026, 14, 544.

### 3.1. Mục tiêu

Bài báo không chỉ đề xuất GDAN mà thực hiện benchmark 13 biến thể attention theo bốn chiều:

1. classification performance;
2. calibration;
3. cross-dataset generalization;
4. FPGA edge deployment.

Đầu vào là **eye image grayscale 64×64**, bài toán nhị phân Awake / Drowsy.

### 3.2. CNN backbone

```text
64×64×1
  ↓
ConvBlock: Conv3×3 → BN → ReLU → MaxPool2×2, 32 channels
  ↓ 32×32×32
ConvBlock, 64 channels
  ↓ 16×16×64
ConvBlock, 128 channels
  ↓ 8×8×128
GDAU
  ↓
Flatten = 8192
  ↓
FC 256 → Dropout → FC 2
  ↓
Softmax
```

Baseline/channel variants dùng FC512; GDAN mặc định dùng FC256.

### 3.3. GDAU

GDAU gồm hai nhánh áp dụng tuần tự.

#### Position-aware spatial attention

Từ feature map `F ∈ R^{C×H×W}`:

```math
M_{avg} = \frac{1}{C}\sum_c F_c,
\qquad
M_{max} = \max_c F_c
```

Sau đó concat với learned positional encoding `P`:

```math
A_s = \sigma(Conv_{7×7}([M_{avg}; M_{max}; P]))
```

```math
F_s = F \odot A_s
```

#### SE channel attention

```math
z = GAP(F)
```

```math
A_c = \sigma(W_2 ReLU(W_1 z)), \qquad r=16
```

```math
F_c = F \odot A_c
```

Default order: **Spatial → Channel**.

### 3.4. Parameter count

- Baseline 32/64/128 + FC512: **4.29 M**.
- GDAN + FC256: **2.19 M**.
- GDAU bản thân chỉ thêm khoảng **2,261 parameters**.

Điểm rất quan trọng: phần giảm từ 4.29 M → 2.19 M **chủ yếu do FC256 thay FC512**, không phải do attention tự làm model nhỏ đi.

### 3.5. Dữ liệu nguồn

Multi-Gaze Eye Dataset được tổ chức lại từ MRL:

| Nhóm | Số ảnh |
|---|---:|
| Forward / Awake | 3,457 |
| Left / Awake | 3,498 |
| Right / Awake | 8,122 |
| Close / Drowsy | 88,726 |
| **Tổng Awake** | **15,077** |
| **Tổng Drowsy** | **88,726** |
| **Tổng** | **103,803** |

Tỉ lệ mất cân bằng khoảng **5.9:1**.

Protocol:

1. split toàn bộ dữ liệu trước: train 70%, val 15%, test 15%;
2. chỉ **undersample training set** về 1:1;
3. validation/test giữ phân bố mất cân bằng tự nhiên.

Test set khoảng:

- 13,309 Drowsy;
- 2,262 Awake;
- tổng `n ≈ 15,571`.

### 3.6. Preprocessing và training

Preprocessing:

- grayscale;
- resize 64×64;
- normalize về `[-1, 1]`.

Augmentation train:

- horizontal flip p=0.5;
- rotation ±10°;
- brightness ±20%;
- contrast ±20%.

Hyperparameters:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| LR | 0.001 |
| Weight decay | 1e-4 |
| Batch | 64 |
| Epochs | 20 |
| Scheduler | ReduceLROnPlateau |
| Scheduler patience | 5 |
| LR factor | 0.5 |
| Early stopping patience | 10 |
| Loss | Cross-entropy |
| Dropout | 0.5 |
| Seeds | 0-4 |

### 3.7. In-domain performance

Trên test set mất cân bằng tự nhiên:

| Model | Accuracy | Balanced Acc. | F1 | ROC-AUC | Params |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 81.78±0.97% | 85.86±0.87% | 0.841±0.008 | 0.921±0.004 | 4.29 M |
| Channel Attention | **83.86±0.76%** | 84.99±3.05% | 0.856±0.006 | 0.920±0.012 | 4.29 M |
| **GDAN** | 82.83±0.55% | 85.99±1.39% | 0.849±0.004 | 0.922±0.006 | **2.19 M** |
| CBAM-L | 82.85±0.79% | **86.77±1.84%** | 0.849±0.006 | **0.927±0.008** | fair-backbone comparison |
| SE-L | 83.08±1.58% | 85.81±2.23% | 0.851±0.013 | 0.922±0.010 | fair-backbone comparison |
| ECA-L | 82.78±1.01% | 86.17±1.54% | 0.849±0.008 | 0.924±0.007 | fair-backbone comparison |
| CoordAtt-L | 83.78±0.96% | **86.86±0.83%** | **0.857±0.007** | **0.927±0.004** | fair-backbone comparison |

Kết luận của chính bài báo: **không có attention mechanism nào thống trị mọi metric**; chênh lệch practical giữa các top model chỉ khoảng ≤1.1 điểm % Accuracy.

### 3.8. Per-class metrics

| Class | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Awake | Baseline | 44.0% | **92%** | 59.4% |
| Awake | Channel | 47.0% | 87% | **60.8%** |
| Awake | GDAN | 45.5% | 90% | 60.5% |
| Drowsy | Baseline | **98.3%** | 80% | 88.3% |
| Drowsy | Channel | 97.4% | **83%** | **89.8%** |
| Drowsy | GDAN | 98.1% | 82% | 89.0% |

Điểm đáng chú ý: raw Accuracy cao không phản ánh đầy đủ hệ thống vì test set mất cân bằng mạnh; Balanced Accuracy, FAR, MR và per-class recall quan trọng hơn.

### 3.9. Attention ablation

Tất cả dùng cùng backbone 32/64/128 + FC256:

| Variant | Pos. Spatial | SE Channel | Accuracy | Balanced Acc. | F1 | Δ Accuracy vs baseline |
|---|---|---|---:|---:|---:|---:|
| Baseline-FC256 | ✗ | ✗ | 80.46±0.73% | 85.88±0.62% | 0.830±0.006 | - |
| Spatial-P only | ✓ | ✗ | 81.76±0.72% | 85.98±0.89% | 0.841±0.006 | +1.30 pp |
| SE channel only | ✗ | ✓ | **83.08±1.58%** | 85.81±2.23% | **0.851±0.013** | **+2.62 pp** |
| GDAN both | ✓ | ✓ | 82.83±0.55% | **85.99±1.39%** | 0.849±0.004 | +2.37 pp |

Cơ chế rút ra:

- channel attention một mình tăng raw Accuracy nhiều hơn full GDAN;
- balanced accuracy của cả bốn cấu hình gần như **không đổi (~85.8-86.0%)**;
- lợi ích raw Accuracy phần lớn đến từ thay đổi prediction trên lớp majority hơn là cải thiện đồng đều hai lớp.

> **Lưu ý về bài báo:** abstract dùng cách diễn đạt "+2.37% balanced accuracy over baseline", nhưng Table 6 cho thấy +2.37 pp là **raw Accuracy** (`80.46 → 82.83`); Balanced Accuracy chỉ `85.88 → 85.99` (+0.11 pp). Nên trích bảng khi cần số liệu chính xác.

### 3.10. Attention ordering

| Ordering | Accuracy | Balanced Acc. | ROC-AUC |
|---|---:|---:|---:|
| Spatial → Channel | 82.83±0.55% | 85.99±1.39% | 0.922±0.006 |
| Channel → Spatial | 83.33±0.35% | 86.65±1.47% | 0.925±0.007 |

Chênh lệch **không có ý nghĩa thống kê**. Thứ tự attention không phải yếu tố quyết định trong bài toán này.

### 3.11. Safety-critical metrics và calibration

Sau temperature scaling:

| Model | Accuracy | Bal. Acc. | ECE ↓ | Brier ↓ | FAR ↓ | MR ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline FC512 | 81.78% | 85.86% | 0.0422 | 0.1117 | 0.0839 | 0.1990 |
| GDAN FC256 | 82.83% | 85.99% | 0.0368 | 0.1087 | 0.0955 | 0.1847 |
| GDAN FC512 | 83.34% | 87.32% | 0.0407 | 0.1046 | **0.0706** | 0.1830 |
| CBAM-L FC512 | 83.79% | **87.54%** | 0.0372 | **0.1038** | 0.0718 | 0.1774 |
| SE-L FC512 | **83.84%** | 83.86% | **0.0339** | 0.1087 | 0.1611 | **0.1617** |
| CoordAtt-L FC512 | 83.60% | 85.78% | 0.0416 | 0.1071 | 0.1114 | 0.1730 |

Không model nào tốt nhất trên mọi metric safety.

#### Calibration methods

ECE, càng thấp càng tốt:

| Model | None | Temperature | Platt | Vector | Isotonic |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0636 | 0.0432 | 0.0196 | 0.0339 | **0.0085** |
| GDAN FC256 | 0.0583 | 0.0354 | 0.0209 | 0.0289 | **0.0088** |
| GDAN FC512 | 0.0665 | 0.0401 | 0.0162 | 0.0332 | **0.0072** |
| CBAM-L | 0.0620 | 0.0355 | 0.0163 | 0.0319 | **0.0075** |
| SE-L | 0.0471 | 0.0331 | 0.0171 | 0.0309 | **0.0083** |
| CoordAtt-L | 0.0607 | 0.0398 | 0.0142 | 0.0271 | **0.0093** |

Brier score:

| Model | None | Temperature | Platt | Vector | Isotonic |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1183 | 0.1110 | 0.0848 | 0.1011 | 0.0775 |
| GDAN FC256 | 0.1151 | 0.1081 | 0.0836 | 0.0998 | 0.0770 |
| GDAN FC512 | 0.1131 | 0.1040 | 0.0794 | 0.0980 | 0.0747 |
| CBAM-L | 0.1118 | 0.1023 | **0.0747** | 0.1005 | **0.0744** |
| SE-L | 0.1124 | 0.1080 | 0.0777 | 0.1020 | 0.0771 |
| CoordAtt-L | 0.1138 | 0.1056 | 0.0776 | 0.0925 | 0.0772 |

Isotonic regression giảm ECE mạnh nhất, nhưng phức tạp hơn temperature scaling.

### 3.12. Cross-dataset generalization

Train trên Multi-Gaze, zero-shot sang YawDD và NTHU-DDD:

| Model | Source Accuracy | YawDD | NTHU-DDD | Avg. drop |
|---|---:|---:|---:|---:|
| Baseline | 81.78% | 50.13% | **56.68%** | 34.6% |
| Channel | 83.86% | 51.10% | 54.47% | 35.8% |
| **GDAN** | 82.83% | 51.32% | 51.95% | 36.7% |
| CBAM-L | 82.85% | 49.22% | 48.88% | 39.3% |
| SE-L | 83.08% | 50.17% | 49.85% | 37.1% |
| ECA-L | 82.78% | 50.87% | 56.37% | 34.7% |
| CoordAtt-L | 83.78% | 50.77% | 50.02% | 38.4% |

=> Tất cả gần random chance. Attention không giải quyết domain shift.

Bài báo phân tích embedding:

- within-source class separation: **7.18**;
- within-target class separation: **1.74**;
- same-class cross-domain shift: **4.96**.

Tức **domain shift lớn hơn chính khoảng cách giữa hai class trong target domain**.

### 3.13. Few-shot adaptation

NTHU-DDD, số sample mỗi class:

| Model | Zero-shot | 5-shot | 10-shot | 25-shot | 50-shot |
|---|---:|---:|---:|---:|---:|
| Baseline FC512 | 52.90±1.10% | 52.50±0.90% | 52.55±0.85% | 52.50±0.90% | 52.45±0.85% |
| GDAN FC256 | 51.85±3.45% | 51.25±3.45% | 51.15±3.95% | 50.95±4.15% | 51.05±3.65% |
| CBAM-L FC512 | 51.95±2.45% | 51.30±1.30% | 50.90±0.80% | 50.85±0.85% | 51.30±1.00% |

5-50 samples/class gần như không giúp. Đây là domain mismatch mức modality, không chỉ parameter mismatch.

### 3.14. FPGA / INT8 deployment

Xilinx Kria KV260, INT8 Vitis AI:

| Model | Params | INT8 size | Accuracy Δ | DPU latency | FPS | CPU latency | Power Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 4.29 M | 4.1 MB | +1.0% | 0.755 ms | 1324 | 9.08 ms | 2.00 W |
| Channel Attn. | 4.29 M | 4.1 MB | 0.0% | 1.016 ms | 984 | 9.09 ms | 1.56 W |
| GDAN-DPU | 2.19 M | 2.1 MB | 0.0% | 1.116 ms | 896 | 8.35 ms | **1.04 W** |
| GDAN-DN | 2.16 M | 2.1 MB | -0.1% | **0.931 ms** | **1075** | - | 1.23 W |
| CBAM | 533 K | 0.5 MB | 0.0% | không compile DPU | - | 2.59 ms | - |
| SE-Net | 533 K | 0.5 MB | -0.5% | **0.481 ms** | **2077** | **2.21 ms** | **0.69 W** |
| ECA-Net | 533 K | 0.5 MB | 0.0% | không compile DPU | - | 2.20 ms | - |

Các operation gây vấn đề:

- CBAM: `adaptive_max_pool2d` không được hỗ trợ;
- ECA: `conv1d` không được hỗ trợ;
- GDAN gốc có `reduction_max`, phải thiết kế GDAN-DPU/GDAN-DN.

GDAN-DN thay spatial attention bằng `Conv2d(128,1,7)` trực tiếp trên feature map, giảm boundary crossing và nhanh hơn GDAN-DPU khoảng **17%**.

### 3.15. Ưu điểm

- Benchmark attention khá có kiểm soát, có fair-backbone comparison.
- Không chỉ nhìn Accuracy; có Balanced Accuracy, FAR, MR, calibration, per-class metrics.
- Rất mạnh về **hardware-aware deployment** và INT8/FPGA.
- Chỉ ra thẳng rằng attention mechanism không phải bottleneck chính.
- Phân tích domain shift ở cả image statistics và embedding space.
- Tách được ảnh hưởng của FC size khỏi attention bằng ablation.
- Có hướng thiết kế DPU-native cụ thể, hữu ích cho hệ thống edge.

### 3.16. Nhược điểm

- Bài toán chủ yếu là **eye-state image classification**, chưa phải mô hình hóa fatigue progression theo thời gian.
- Cross-dataset generalization thất bại hoàn toàn (~48-57%).
- Few-shot 5-50 sample/class không khắc phục được domain shift.
- Position-aware spatial attention mang lợi ích nhỏ; scalar encoding gần ngang 8×8 positional map, tức positional encoding hoạt động gần như learned bias.
- Channel attention một mình đã tương đương hoặc tốt hơn full GDAU ở nhiều metric.
- Toolchain FPGA giới hạn operation; kiến trúc tốt trên PyTorch chưa chắc compile được DPU.
- Calibration tốt cần post-hoc calibration; confidence raw chưa đủ đáng tin cho safety-critical thresholding.

---

# 4. So sánh ba hướng tiếp cận

## 4.1. So sánh kiến trúc

| Khía cạnh | Light-VTD | EAR Temporal Framework | GDAN |
|---|---|---|---|
| Input | RGB face ROI 224×224 | Landmark-derived 4-D feature sequence | Grayscale eye crop 64×64 |
| Spatial representation | CNN stem + hierarchical Swin | Không học trực tiếp từ pixel | 3-block CNN |
| Temporal modeling | **Không** | **Có**, bằng window T=10/20/30; LSTM/BiLSTM hoặc flatten cho ML | **Không** |
| Attention | Swin + multi-scale token fusion | Không | Spatial + SE channel attention |
| Main strength | multi-scale visual robustness | temporal behavior + subject-independent protocol | compact attention + calibration + FPGA |
| Main weakness | frame-level | feature quá nghèo + chỉ 4 drivers | domain shift cực mạnh |
| Edge focus | Có | Mô hình nhẹ tự nhiên | **Rất mạnh** |
| Cross-subject/domain realism | Cross-dataset | **LODO unseen driver** | Cross-dataset, nhưng thất bại |

## 4.2. Không nên so sánh Accuracy trực tiếp

Các con số như `98.7% Light-VTD`, `57.6% Logistic Regression LODO`, và `82.83% GDAN` không phản ánh trực tiếp model nào "mạnh hơn" vì:

- Light-VTD chủ yếu đánh giá image classification trong từng dataset và cross-dataset;
- temporal framework test trên **unseen driver**, một protocol khó hơn nhiều;
- GDAN test trên source dataset mất cân bằng mạnh và còn có cross-modality domain shift khi sang NTHU/YawDD.

Vì vậy, **evaluation protocol là một biến của bài toán**, không phải chi tiết phụ.

## 4.3. Insight quan trọng nhất cho thiết kế hệ thống phát hiện buồn ngủ

Ba bài ghép lại cho một kiến trúc hệ thống khá rõ:

```text
Video frame
   │
   ├── Visual branch
   │     CNN / Light-VTD-like encoder
   │     → mắt, miệng, head pose, facial fatigue representation
   │
   ├── Landmark / behavioural branch
   │     EAR, PERCLOS, MAR, blink duration, head dynamics
   │
   └── Temporal aggregation
         LSTM / GRU / Temporal Transformer / RF trên temporal features
               ↓
       calibrated decision / risk score
               ↓
       personalized threshold / alert logic
```

Cơ sở từ ba bài:

1. **Light-VTD:** visual encoder cần giữ local cue + multi-scale context, và robustness phải được kiểm tra dưới blur/occlusion/low-light.
2. **Temporal framework:** temporal context đóng góp nhiều hơn việc chỉ thêm vài engineered feature tại từng frame.
3. **GDAN:** attention tinh vi không tự giải quyết domain shift; calibration và deployment constraint phải được xem như phần của kiến trúc, không phải việc làm sau cùng.

## 4.4. Nếu mục tiêu là Recall buồn ngủ cao

- Không chọn model chỉ theo Accuracy.
- Ưu tiên **Recall drowsy, F1 drowsy, PR-AUC/AUPRC, MCC/Balanced Accuracy**, và theo dõi FAR/MR.
- Random Forest T=30 trong paper temporal là ví dụ rõ: Accuracy chỉ 0.556 nhưng Recall drowsy 0.701 và AUPRC 0.663 - tốt hơn các cấu hình khác theo mục tiêu phát hiện lớp nguy hiểm.
- GDAN paper cũng cho thấy Accuracy cao hơn có thể đi cùng Balanced Accuracy hoặc FAR/MR kém hơn.

## 4.5. Bài học về generalization

- **Identity leakage** có thể làm kết quả quá lạc quan → nên split theo subject/driver.
- **Domain shift** có thể lớn hơn class separation → architecture tweak không cứu được dữ liệu lệch miền.
- Cross-dataset cần đảm bảo **cùng loại đầu vào** trước khi đánh giá: eye crop ↔ eye crop, face ↔ face; nếu eye crop train nhưng full-face test thì đó gần như là modality shift.
- Đánh giá thực tế nên có ít nhất:
  - subject-independent split;
  - cross-dataset split;
  - low-light / blur / occlusion;
  - per-class recall/F1;
  - calibration;
  - latency + memory + power nếu hướng edge.

---

# 5. Kết luận ngắn

- **Light-VTD** mạnh nhất về **visual representation + robustness + hiệu quả Transformer**, nhưng thiếu temporal modeling.
- **EAR Temporal Framework** có absolute performance thấp, nhưng là bài quan trọng nhất về **evaluation realism** và chứng minh **temporal context thực sự mang thông tin** dưới unseen-driver split.
- **GDAN** cho thấy **attention không phải thuốc chữa bách bệnh**: lợi ích giữa các attention nhỏ, domain shift mới là bottleneck lớn; điểm mạnh nhất của paper là calibration và FPGA deployment.
- Với hệ thống phát hiện buồn ngủ hoàn chỉnh, hướng hợp lý không phải chọn một trong ba, mà là **visual encoder mạnh + temporal behavior modeling + calibrated decision layer**, đánh giá bằng subject-independent/cross-domain protocol.

---

# 6. Nguồn

1. Al Rafy et al., **Domain-robust vision transformer with hierarchical swin encoding for explainable low-latency driver drowsiness detection**, Scientific Reports, 2026. File: `s41598-026-41847-y.pdf`.
2. Ajayi et al., **A Subject-Independent Temporal Framework for Behavioural Eye-Based Driver Drowsiness Detection**, Sensors, 2026. File: `sensors-26-06193.pdf`.
3. Hussein et al., **Guided Dual-Attention Networks for Compact Driver Drowsiness Detection: Performance, Calibration, and Edge Deployment**, Technologies, 2026. File: `technologies-14-00544-v2.pdf`.
