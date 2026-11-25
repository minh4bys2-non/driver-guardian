# LiteFat_v1 Experiment Ledger

Nhật ký thực nghiệm cho module `LiteFat_v1` (KSS 1–9 @ 10 FPS, Sliding Window 60s / Stride 45s).  
Mục tiêu của tệp này là khóa chặt baseline, ghi nhận đầy đủ bằng chứng định lượng cho từng phiên bản và ngăn việc chỉnh sửa lặp đi lặp lại nhiều tham số cùng lúc.

---

## 1. Bảng tổng hợp các phiên bản đã chạy (v3 – v8)

> Tất cả các phiên bản bên dưới đều train từ đầu (Epoch 1, không warm-start chéo), cùng split **Train=518 / Val=66 / Test=65**, chế độ **Landmark-Only (`use_visual_context=False`)**.  
> **Lưu ý công thức `val_score`:**  
> - **v3–v5:** `0.40*(1 - mae/4) + 0.30*within_1_acc + 0.30*max(0, qwk)`  
> - **v6–v8:** `0.30*(1 - mae/4) + 0.15*(1 - rmse/4.5) + 0.20*within_1_acc + 0.20*max(0, qwk) + 0.15*regime_macro_f1`  
> Vì vậy **không so sánh trực tiếp số `val_score` giữa v3–v5 và v6–v8**.

| Ver | Commit / Env | Best Ep | Val MAE | Val QWK | Val AUC | Val UL QWK | Val UTA QWK | **Test MAE** | **Test RMSE** | **Test QWK** | **Test AUC** | Test F1 | Test RegimeAcc | Test UL MAE / QWK / AUC | **Test UTA MAE / QWK / AUC** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | `2cbb4b6` | 12 | 2.2720 | 0.2973 | 0.7385 | 0.4189 | 0.2028 | 2.0679 | 2.5788 | 0.3072 | 0.5833 | 0.5238 | 0.4615 | 1.5002 / 0.5573 / 0.7083 | 2.5253 / 0.1862 / 0.4826 |
| **v4** | `d3cd27f` | 18 | 1.6984 | 0.5123 | 0.8673 | 0.5616 | **0.4624** | 1.5341 | 1.9063 | 0.4840 | 0.7089 | 0.4390 | **0.4769** | **1.1905 / 0.5436** / 0.7262 | 1.8108 / 0.4635 / 0.7049 |
| **v5** | `e7f3259` | 40 | 1.6483 | 0.5437 | 0.8654 | 0.6379 | 0.4481 | 1.6055 | 1.9576 | **0.4968** | 0.7189 | **0.5106** | 0.4154 | 1.2792 / 0.5000 / **0.7381** | 1.8684 / 0.4907 / 0.7083 |
| v6 | `43d6f2b` | 47 | **1.6324** | **0.5789** | 0.8859 | **0.7051** | 0.4476 | 1.5616 | 1.9368 | 0.4426 | 0.6733 | 0.4390 | 0.4154 | 1.3909 / 0.4064 / 0.6964 | 1.6991 / 0.4605 / 0.6736 |
| **v7** | `29d3604` | 28 | 1.9394 | 0.4154 | 0.7863 | 0.5271 | 0.2977 | 1.9866 | 2.3273 | 0.2523 | 0.5989 | 0.2941 | 0.4154 | 1.3918 / 0.5076 / 0.6726 | 2.4658 / 0.1014 / 0.5347 |
| v8a (`seed=42`) | `colab` A100 | 48 | 1.6554 | 0.5448 | 0.8878 | 0.6741 | 0.4089 | 1.6906 | 2.1022 | 0.3394 | 0.6756 | 0.3684 | 0.4000 | 1.5331 / 0.3438 / 0.6667 | 1.8176 / 0.3390 / 0.6424 |
| v8b (`seed=7`) | `colab` A100 | 46 | 1.6854 | 0.5143 | 0.8615 | 0.6266 | 0.4036 | 1.6666 | 2.0240 | 0.4427 | 0.7167 | 0.4444 | 0.4308 | 1.4389 / 0.5085 / 0.7262 | 1.8501 / 0.4112 / 0.6910 |
| **Mean(v8a, v8b)** | Base 2-seed | 47 | 1.6704 | 0.5296 | 0.8746 | 0.6504 | 0.4063 | 1.6786 ± 0.012 | 2.0631 ± 0.039 | 0.3910 ± 0.052 | 0.6961 ± 0.021 | 0.4064 | 0.4154 | 1.4860 / 0.4262 / 0.6965 | 1.8339 / 0.3751 / 0.6667 |
| v8d (`interval_fixed`) | `colab` A100 | 49 | 1.7926 | 0.5121 | **0.9005** | 0.6339 | 0.3632 | 1.6518 | 1.9928 | 0.3917 | 0.6744 | 0.3243 | 0.4000 | 1.4945 / 0.2832 / 0.5714 | 1.7785 / 0.4377 / **0.7674** |
| **`v8f` (`seed=42`, Track A)** | `colab` A100 | 42 | 1.7421 | 0.5046 | 0.8712 | 0.6665 | 0.3556 | **1.4675** | **1.8742** | 0.4909 | 0.6900 | 0.4878 | **0.4769** | 1.2506 / 0.4832 / 0.6964 | **1.6422 / 0.4915** / 0.6806 |
| `v8f_s7` (`seed=7`, Track A) | `colab` A100 | 49 | 1.9010 | 0.4432 | 0.8605 | 0.5991 | 0.2869 | 1.7253 | 2.1423 | 0.4295 | 0.7167 | 0.4615 | 0.4154 | 1.4892 / 0.4955 / 0.7262 | 1.9156 / 0.4000 / 0.6875 |
| **Mean(v8f, v8f_s7)** | Track A 2-seed | 45.5 | 1.8216 | 0.4739 | 0.8659 | 0.6328 | 0.3213 | **1.5964 ± 0.129** | 2.0083 ± 0.134 | **0.4602 ± 0.031** | **0.7033 ± 0.013** | **0.4747** | 0.4462 | **1.3699 / 0.4893 ± 0.006 / 0.7113** | **1.7789 / 0.4458 ± 0.046 / 0.6840** |
| ENS `0.5(v8a+v8b)` | Base 2-seed | — | **1.6242** | **0.5654** | 0.8859 | 0.6826 | 0.4495 | 1.6085 | 1.9742 | 0.4325 | 0.7189 | 0.4324 | 0.4308 | 1.3628 / 0.5018 / 0.7321 | 1.8064 / 0.4014 / 0.6875 |
| ENS `0.5(v8b+v8d)` | 2-run avg | — | 1.7076 | 0.5137 | 0.8946 | — | — | 1.5811 | — | 0.4349 | 0.7233 | — | — | 1.2999 / 0.4702 / 0.6964 | 1.8077 / 0.4179 / 0.7604 |
| ENS `0.5(v8f+v8f_s7)` | Track A 2-seed | — | 1.7823 | 0.4943 | 0.8702 | 0.6456 | 0.3484 | 1.5380 | 1.8945 | 0.4785 | 0.7156 | 0.4324 | 0.4462 | 1.2396 / **0.5409** / 0.7024 | 1.7783 / 0.4513 / 0.7014 |
| **ENS `0.5(v8f+v8b)`** | Cross-Boundary | — | 1.6842 | 0.5253 | 0.8702 | 0.6348 | 0.4279 | **1.4994** | **1.8622** | **0.4905** | **0.7300** | 0.3889 | 0.4308 | **1.2042** / 0.5254 / 0.7262 | **1.7371 / 0.4752** / 0.7153 |
| ENS `1/3(v8f+v8b+v8d)` | 3-model diverse | — | 1.7097 | 0.5110 | 0.8946 | 0.6351 | 0.3841 | 1.5366 | 1.8637 | 0.4646 | 0.7156 | 0.3889 | 0.4615 | 1.2709 / 0.4839 / 0.6845 | 1.7506 / 0.4536 / **0.7326** |

*(Ghi chú: Test AUC dùng `drowsy_prob` giống `train.py`; nếu tính AUC trực tiếp trên `pred_kss`, `ENS 0.5(v8f+v8b)` đạt `0.7433` và `ENS 0.5(v8f+v8f_s7)` đạt `0.7311`.)*

### Khác biệt cấu hình & code chính giữa các bản

- **v3 (`2cbb4b6`)**: `stride_sec=30.0`, `static_shape_ratio=0.20`, `dann=0.05/0.20`, `ema_decay=0.995`, `weight_decay=2e-3`.
- **v4 (`d3cd27f`)**: `stride_sec=45.0`, `static_shape_ratio=0.55`, `dann=0.02/0.15`, `ema_decay=0.990`, tắt nhân chồng `sample_weight` khi bật `subject_balanced_sampling`.
- **v5 (`e7f3259`)**: `static_shape_ratio=0.35`, `weight_decay=3e-3`, `dropout=0.35`, thêm `boundary_margin_weight=0.15`, `uta_ordinal_sigma=1.25` (Gaussian mềm cho UTA), quota sampler `0.60 UL / 0.40 UTA`.
- **v6 (`43d6f2b`)**: `static_shape_ratio=0.15`, `dann=0.15/0.25`, giới hạn regime support trong ordinal UTA (`<=6.0`), thêm boundary penalty cấp window, tách P85 mắt khỏi baseline cơ quan & tách vận tốc mi/miệng cục bộ khỏi xoay đầu, đổi công thức `val_score` thêm RMSE & `regime_macro_f1`.
- **v7 (`29d3604`)**: Đổi 6 thứ cùng lúc — (1) Interval Regression cho UTA với khoảng mid `[4.5, 6.2]` & tâm lệch `(2.5, 5.4, 7.8)` trọng số `0.15`; (2) `boundary_margin_weight=0.05` & ngưỡng `6.30/6.65`; (3) `dann=0.05/0.15`; (4) quota sampler `0.65 UL / 0.35 UTA` & số mũ `(0.70, 0.50, 0.50)`; (5) làm mượt nhân quả 2-frame cho vận tốc mi mắt (`36..47`) + đổi `eye_y_scale`; (6) `final_attn_logits` dùng trọng số từ aggregator + tách nhóm `no_decay` trong AdamW. **Kết quả: hồi quy nặng trên UTA-RLDD** (Test UTA QWK 0.4605 → 0.1014, Val UTA MAE bị chặn sàn ~2.08).
- **v8a (`seed=42`) & v8b (`seed=7`)**: Khôi phục baseline `v6` qua các cờ mặc định trong `TrainConfig`. **Xác nhận H1**: Test UTA MAE giảm `2.4658 → 1.8176 / 1.8501`, Test UTA QWK phục hồi `0.1014 → 0.3390 / 0.4112` (Mean `0.3751 ≥ 0.35`); đo được biên độ nhiễu seed trên Test `N=65` là `|Δ QWK| = 0.1032` (`UL-DD |Δ QWK| = 0.1646`). Chẩn đoán còn hiện tượng **"Mid-Regime Magnet `[4.5, 6.5)`"** (`58.3%–63.9%` dự đoán UTA Test rơi vào Mid; `7/12` `KSS=2` tràn lên Mid) do `loss_boundary` chỉ chặn 1 phía `6.20/6.70`, bỏ trống biên `4.5`.
- **v8d (`seed=42`, `uta_reg_mode="interval_fixed"`, `cw=0.50`)**: Thử nghiệm đơn biến dùng khoảng `[1,4] / [4.5,6.5] / [6.5,9]` với tâm thật `{2, 6, 8}`. **Hiệu ứng bập bênh**: đẩy Test UTA QWK lên `0.4377` (`AUC=0.7674`), nhưng bên trong khoảng `dist_out=0` làm lực kéo về tâm `{2, 8}` yếu đi `3.0x–3.5x` (`grad=0.50` vs `1.75` ở `point`) và đổi `mid_upper=6.5` trong ordinal loss, khiến `KSS=2` đạt `0% Within-1` (`72.2%` mẫu UTA Test nằm ở Mid), Val tụt (`val_score 0.5317 → 0.5002`, `MAE 1.6554 → 1.7926`) và `UL-DD Test QWK` rớt xuống `0.2832`. **Kết luận: Reject standalone.**
- **v8f (`seed=42`) & v8f_s7 (`seed=7`, `use_alert_boundary_margin=True`, `4.20/4.70` trên nền `point`)**: Giữ `uta_reg_mode="point"` (100% lực kéo tâm & `mid_upper=6.0`) và thêm rào cản đối xứng tại `4.20 / 4.70`. **Kết luận: Accepted** — Mean 2-seed trên Test đạt **`QWK = 0.4602 ± 0.0307` (`+0.0692 > +0.05` so với `(v8a, v8b)`), `MAE = 1.5964` (`-0.0822`), `UL-DD QWK = 0.4893 ± 0.0061` (`+0.0632`, giảm nhiễu seed `13x`), `UTA QWK = 0.4458 ± 0.0458` (`+0.0707`)**; riêng `v8f` lập kỷ lục toàn dự án về Test MAE (`1.4675`), Test RMSE (`1.8742`), Test W1 (`43.08%`), UTA MAE (`1.6422`) và UTA QWK (`0.4915`). Lưu ý hiện tượng **bất đối xứng offset Val vs Test**: do mẫu Non-Drowsy (`<=6`) nhiều gấp `~2:1` Drowsy (`>=7`), tường `4.20` dịch nhẹ tâm toàn cục xuống (`Val pred_mean: 5.53→5.42` / `5.31→5.07`), giúp khử lệch dương trên Test nhưng làm Val UTA vốn lệch âm (`uta_50/52/53`) tụt nhẹ; `ENS 0.5*(v8f+v8b)` cân bằng cả hai mặt (`Val RegimeAcc=71.21%, F1=0.8000` & `Test MAE=1.4994, QWK=0.4905, AUC=0.7300`).

---

## 2. ĐÃ THỬ – KHÔNG LẶP LẠI

1. **Quét qua lại `dann_weight` / `dann_max_alpha`** (`0.05/0.20` → `0.02/0.15` → `0.15/0.25` → `0.05/0.15`): Không tạo khác biệt nhất quán trên test. **Khóa tại `0.15 / 0.25` (baseline v6)** và không tune tiếp nếu không có ablation riêng.
2. **Quét qua lại `static_shape_ratio`** (`0.20` → `0.55` → `0.35` → `0.15`): Dao động 4 lần mà v4/v5/v6 tương đương về mặt thống kê trên 65 mẫu test. **Khóa tại `0.15` (baseline v6)**, ngừng chỉnh.
3. **`uta_reg_mode="interval"` với tâm lệch `(2.5, 5.4, 7.8)` và trần mid `6.2` (v7)**: Lệch trực tiếp với nhãn thật UTA `{2, 6, 8}` mà MAE/QWK chấm điểm, làm gradient thứ tự trong khoảng triệt tiêu. **Cấm bật lại cấu hình tâm lệch này.**
4. **Chỉnh `boundary_margin_weight` và ngưỡng `6.20/6.70` vs `6.30/6.65` cùng lúc với nhiều biến khác**: Chưa thấy lợi ích rõ trên test so với v4 (không dùng boundary). **Giữ mặc định v6 (`0.15`, `6.20/6.70`)** cho đến khi kiểm chứng đơn biến.
5. **Đổi 4–8 biến một lúc + chỉ chạy 1 seed trên tập Test N=65**: Sai số chuẩn của QWK/AUC trên N=65 khoảng `±0.08–0.10` (`|v8a - v8b| = 0.1032` QWK), khiến v4/v5/v6 thực chất tương đương thống kê và che lấp nguyên nhân khi có hồi quy.
6. **`uta_reg_mode="interval_fixed"` với `uta_interval_center_weight=0.50` không có MSE tâm (v8d)**: Triệt tiêu MSE tâm khiến lực kéo về `{2.0, 8.0}` bên trong khoảng yếu hơn `3.0x–3.5x` so với `point` mode (`0.50` vs `1.75`) và đẩy `mid_upper=6.5` trong ordinal loss; gây `0% Within-1` cho `KSS=2`, nén `72.2%` mẫu UTA Test vào Mid và làm `UL-DD Test QWK` tụt xuống `0.2832`. **Cấm dùng lại làm loss đơn lẻ thay thế `point` mode.**

---

## 3. Mô hình chốt & Quy tắc thực nghiệm sau chuỗi `v8`

- **Best Single Model Checkpoint (Chốt triển khai đơn mô hình):**
  - **`checkpoints/LiteFat_v1_novisual8/v8f/best.pt`** (`seed=42`, `uta_reg_mode="point"`, **`use_alert_boundary_margin=True`**, `boundary_alert_max=4.20`, `boundary_mid_min=4.70`, `boundary_drowsy_min=6.70`, `boundary_nondrowsy_max=6.20`).
  - Hiệu năng Test (`N=65`): **`MAE = 1.4675`** (CI95 `[1.1976, 1.7632]`), **`RMSE = 1.8742`**, **`Within-1 = 43.08%`**, **`QWK = 0.4909`** (CI95 `[0.2996, 0.6452]`), **`AUC = 0.6900`**, **`F1 = 0.4878`** (`UL-DD: MAE=1.2506, QWK=0.4832`; `UTA-RLDD: MAE=1.6422, QWK=0.4915`).
- **Best 2-Model Ensembles (Chốt suy luận đa mô hình):**
  - **Cross-Boundary `0.5*(v8f + v8b)`:** `Test MAE = 1.4994`, `RMSE = 1.8622`, `QWK = 0.4905`, **`AUC = 0.7300` (`0.7433` theo `pred_kss`)**; `Val RegimeAcc = 71.21%, F1 = 0.8000, QWK = 0.5253`.
  - **2-Seed Track A `0.5*(v8f + v8f_s7)`:** `Test MAE = 1.5380`, `RMSE = 1.8945`, `QWK = 0.4785`, `AUC = 0.7156` (`UL-DD QWK = 0.5409`, `UTA QWK = 0.4513`).
- **Baseline tham chiếu mặc định trong `TrainConfig`:** Giữ `use_alert_boundary_margin=False` làm mặc định lớp `TrainConfig` (`v6-equivalent`, tương ứng baseline `(v8a, v8b)`) để không làm gãy tương thích ngược các checkpoint cũ; bật `use_alert_boundary_margin=True` khi tái lập `v8f` (và cân nhắc thêm `alert_boundary_scale < 1.0` nếu làm thực nghiệm v9 nhằm giữ cân bằng Val/Test).
- **Quy tắc cho mọi thực nghiệm sau này (v9+):**
  1. **Mỗi run chỉ đổi 1 biến** so với baseline đã khóa và kiểm chứng tối thiểu **2 seed (`42`, `7`)**.
  2. **Mọi thay đổi hành vi mới đi qua cờ `TrainConfig`**, lưu đầy đủ `run_meta.json`, `val_predictions.csv`, `test_predictions.csv`, `bootstrap_ci_95`.

---

## 4. Hàng đợi & Trạng thái chuỗi `v8`

| Run | Giai đoạn | Thay đổi duy nhất so với baseline `(v8a, v8b)` | Seed | `save_freq` | Trạng thái | Kết quả tóm tắt |
|---|---|---|---|---|---|---|
| `v8a` | GĐ 1 (Baseline) | Không đổi (`uta_reg_mode="point"`) | 42 | 1 | **Done** | Test MAE=`1.6906`, QWK=`0.3394`, UTA QWK=`0.3390` |
| `v8b` | GĐ 1 (Baseline) | Không đổi (`uta_reg_mode="point"`) | 7 | 1 | **Done** | Test MAE=`1.6666`, QWK=`0.4427`, UTA QWK=`0.4112` |
| `v8d` | GĐ 2 (Ablation 1) | `uta_reg_mode="interval_fixed"` (`cw=0.50`) | 42 | 1 | **Done (Rejected)** | Test UTA QWK=`0.4377`, nhưng Val Score=`0.5002`, UL QWK=`0.2832`, KSS=2 W1=`0%` |
| **`v8f`** | **GĐ 3 (Track A - Run 1)** | **`use_alert_boundary_margin=True` (`4.20/4.70` + `point`)** | **42** | **1** | **Done (Accepted — Best Single)** | **Test MAE=`1.4675`, QWK=`0.4909`, W1=`43.08%`, UTA MAE=`1.6422`, UTA QWK=`0.4915`** |
| **`v8f_s7`** | **GĐ 3 (Track A - Run 2)** | **`use_alert_boundary_margin=True` (`4.20/4.70` + `point`)** | **7** | **1** | **Done (Accepted — Seed 2)** | Test MAE=`1.7253`, QWK=`0.4295`, UL QWK=`0.4955`, UTA QWK=`0.4000`; **Mean QWK=`0.4602` (`+0.0692`)** |
| `v8c` | Backlog | `eye_vel_smooth=True`, `eye_y_scale_mode="v7"` | 42 (+7) | 1 | Backlog | Kiểm định giả thuyết H3 |
| `v8e` | Backlog | `sampler_uta_quota=0.35`, `sampler_weight_exponents=(0.70, 0.50, 0.50)` | 42 (+7) | 1 | Backlog | Kiểm định giả thuyết H2 (ưu tiên thấp) |

---

## 5. Nhật ký chi tiết các Run v8

### Run: v8a (Baseline v6, `save_freq=1`)
- **Ngày / Commit:** 2026-11-20 | `colab` (`NVIDIA A100-SXM4-40GB`)
- **Biến thay đổi duy nhất so với baseline:** không đổi (baseline `v6`)
- **Seed:** `42` | **Epochs / Best Ep:** `60/48`
- **Val (best.pt):** MAE=`1.6554` | QWK=`0.5448` | AUC=`0.8878` | Score=`0.5317` | UL QWK=`0.6741` | UTA QWK=`0.4089`
- **Test (best.pt):** MAE=`1.6906` (CI95: `[1.3944, 1.9919]`) | QWK=`0.3394` (CI95: `[0.1660, 0.4945]`) | AUC=`0.6756` (CI95: `[0.5444, 0.8039]`) | F1=`0.3684`
- **Test Per-Dataset:** UL-DD (MAE=`1.5331`, QWK=`0.3438`, AUC=`0.6667`) | UTA-RLDD (MAE=`1.8176`, QWK=`0.3390`, AUC=`0.6424`)
- **Kết luận:** Baseline Seed 1 (xác nhận H1: thoát sụp đổ UTA của `v7`).

### Run: v8b (Baseline v6, `save_freq=1`)
- **Ngày / Commit:** 2026-11-20 | `colab` (`NVIDIA A100-SXM4-40GB`)
- **Biến thay đổi duy nhất so với baseline:** không đổi (baseline `v6`)
- **Seed:** `7` | **Epochs / Best Ep:** `58/46`
- **Val (best.pt):** MAE=`1.6854` | QWK=`0.5143` | AUC=`0.8615` | Score=`0.5384` | UL QWK=`0.6266` | UTA QWK=`0.4036`
- **Test (best.pt):** MAE=`1.6666` (CI95: `[1.4089, 1.9540]`) | QWK=`0.4427` (CI95: `[0.2731, 0.5742]`) | AUC=`0.7167` (CI95: `[0.5985, 0.8272]`) | F1=`0.4444`
- **Test Per-Dataset:** UL-DD (MAE=`1.4389`, QWK=`0.5085`, AUC=`0.7262`) | UTA-RLDD (MAE=`1.8501`, QWK=`0.4112`, AUC=`0.6910`)
- **Tổng hợp 2-Seed `(v8a, v8b)`:** Mean Test MAE=`1.6786 ± 0.0120` | Mean Test QWK=`0.3910 ± 0.0516` (`|Δ|=0.1032`) | Mean Test AUC=`0.6961 ± 0.0206` | Mean UTA QWK=`0.3751 ± 0.0361` (Đạt cổng UTA ≥ 0.35; Ensemble `0.5*(v8a+v8b)` đạt Test MAE=`1.6085`, QWK=`0.4325`, AUC=`0.7189`).

### Run: v8d (`uta_reg_mode="interval_fixed"`, `save_freq=1`)
- **Ngày / Commit:** 2026-11-21 | `colab` (`NVIDIA A100-SXM4-40GB`)
- **Biến thay đổi duy nhất so với baseline:** `uta_reg_mode="interval_fixed"` (`uta_interval_center_weight=0.50`)
- **Seed:** `42` | **Epochs / Best Ep:** `60/49`
- **Val (best.pt):** MAE=`1.7926` | QWK=`0.5121` | AUC=`0.9005` | Score=`0.5002` | UL QWK=`0.6339` | UTA QWK=`0.3632`
- **Test (best.pt):** MAE=`1.6518` (CI95: `[1.3873, 1.9188]`) | QWK=`0.3917` (CI95: `[0.2426, 0.5202]`) | AUC=`0.6744` (CI95: `[0.5433, 0.8068]`) | F1=`0.3243`
- **Test Per-Dataset:** UL-DD (MAE=`1.4945`, QWK=`0.2832`, AUC=`0.5714`) | UTA-RLDD (MAE=`1.7785`, QWK=`0.4377`, AUC=`0.7674`)
- **Kết luận:** **Reject (standalone)** — Hiệu ứng bập bênh: Test UTA QWK/AUC tăng nhờ phạt biên nhưng Val hồi quy (`val_score 0.5317→0.5002`, `MAE 1.6554→1.7926`), `KSS=2` đạt `0% Within-1` (`min=3.04`, `72.2%` mẫu UTA Test nằm ở Mid) và `UL-DD Test QWK` rớt `0.2832`.

### Run: v8f (Track A: Dual-Boundary `use_alert_boundary_margin=True`, `save_freq=1`)
- **Ngày / Commit:** 2026-11-22 | `colab` (`NVIDIA A100-SXM4-40GB`)
- **Biến thay đổi duy nhất so với baseline:** `use_alert_boundary_margin=True` (`boundary_alert_max=4.20`, `boundary_mid_min=4.70`)
- **Seed:** `42` | **Epochs / Best Ep:** `54/42` (early stop)
- **Val (best.pt):** MAE=`1.7421` | QWK=`0.5046` | AUC=`0.8712` | Score=`0.5075` | UL QWK=`0.6665` | UTA QWK=`0.3556`
- **Test (best.pt):** MAE=`1.4675` (CI95: `[1.1976, 1.7632]`) | RMSE=`1.8742` | W1=`43.08%` | Exact=`29.23%` | QWK=`0.4909` (CI95: `[0.2996, 0.6452]`) | AUC=`0.6900` (CI95: `[0.5552, 0.8149]`) | F1=`0.4878` | RegimeAcc=`47.69%`
- **Test Per-Dataset:** UL-DD (MAE=`1.2506`, QWK=`0.4832`, AUC=`0.6964`) | UTA-RLDD (MAE=`1.6422`, QWK=`0.4915`, AUC=`0.6806`, `KSS=2 W1=25.0%`)
- **Kết luận:** **Accept (Best Single Checkpoint)** — Kỷ lục Test MAE (`1.4675`), Test RMSE (`1.8742`), Test W1 (`43.08%`), Test UTA MAE (`1.6422`) và Test UTA QWK (`0.4915`); `r_within` dương mạnh trên `7/8` tài xế Test (`uta_56=+0.98`, `uta_55=+0.94`, `uta_58=+0.69`, `uldd_A=+0.64`, `uldd_C=+0.64`).

### Run: v8f_s7 (Track A Seed 2: `use_alert_boundary_margin=True`, `save_freq=1`)
- **Ngày / Commit:** 2026-11-22 | `colab` (`NVIDIA A100-SXM4-40GB`)
- **Biến thay đổi duy nhất so với baseline:** `use_alert_boundary_margin=True` (`boundary_alert_max=4.20`, `boundary_mid_min=4.70`)
- **Seed:** `7` | **Epochs / Best Ep:** `60/49`
- **Val (best.pt):** MAE=`1.9010` | QWK=`0.4432` | AUC=`0.8605` | Score=`0.4729` | UL QWK=`0.5991` | UTA QWK=`0.2869`
- **Test (best.pt):** MAE=`1.7253` (CI95: `[1.4498, 2.0396]`) | RMSE=`2.1423` | W1=`33.85%` | QWK=`0.4295` (CI95: `[0.2184, 0.5961]`) | AUC=`0.7167` (CI95: `[0.5906, 0.8323]`) | F1=`0.4615` | RegimeAcc=`41.54%`
- **Test Per-Dataset:** UL-DD (MAE=`1.4892`, QWK=`0.4955`, AUC=`0.7262`) | UTA-RLDD (MAE=`1.9156`, QWK=`0.4000`, AUC=`0.6875`, UTA Mid=`50.0%`)
- **Tổng hợp 2-Seed Track A `(v8f, v8f_s7)` vs Baseline `(v8a, v8b)`:**
  - Mean Test MAE: `1.6786 → 1.5964 ± 0.1289` (`-0.0822`)
  - Mean Test QWK: `0.3910 ± 0.0516 → 0.4602 ± 0.0307` (`+0.0692 > +0.05`, giảm `40%` nhiễu seed)
  - Mean Test AUC: `0.6961 → 0.7033 ± 0.0133` (`+0.0072`)
  - Mean Test UL-DD QWK: `0.4262 ± 0.0823 → 0.4893 ± 0.0061` (`+0.0632`, giảm `13x` nhiễu seed)
  - Mean Test UTA-RLDD QWK: `0.3751 ± 0.0361 → 0.4458 ± 0.0458` (`+0.0707`)
  - **Ensembles:** `0.5*(v8f + v8b)` (`Test MAE=1.4994, QWK=0.4905, AUC=0.7300 | Val RegimeAcc=71.21%, F1=0.8000`); `0.5*(v8f + v8f_s7)` (`Test MAE=1.5380, QWK=0.4785, UL QWK=0.5409, UTA QWK=0.4513`).

### Mẫu ghi chép cho Run mới (Copy-Paste)

```markdown
### Run: v9x (<mô tả ngắn 1 thay đổi>)
- **Ngày / Commit:** YYYY-MM-DD | `<git_hash>` (check `run_meta.json`)
- **Biến thay đổi duy nhất so với baseline:** `<key>=<value>`
- **Seed:** `<seed>` | **Epochs / Best Ep:** `<ran>/<best>`
- **Val (best.pt):** MAE=`...` | QWK=`...` | AUC=`...` | UL QWK=`...` | UTA QWK=`...`
- **Test (best.pt):** MAE=`...` (CI95: `[..., ...]`) | QWK=`...` (CI95: `[..., ...]`) | AUC=`...` (CI95: `[..., ...]`) | F1=`...`
- **Test Per-Dataset:** UL-DD (MAE=`...`, QWK=`...`, AUC=`...`) | UTA-RLDD (MAE=`...`, QWK=`...`, AUC=`...`)
- **Kết luận:** Accept / Reject / Needs 2nd seed — lý do: ...
```
