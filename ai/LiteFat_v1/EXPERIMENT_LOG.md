# LiteFat_v1 Experiment Ledger

Nhật ký thực nghiệm cho module `LiteFat_v1` (KSS 1–9 @ 10 FPS, Sliding Window 60s / Stride 45s).  
Mục tiêu của tệp này là khóa chặt baseline, ghi nhận đầy đủ bằng chứng định lượng cho từng phiên bản và ngăn việc chỉnh sửa lặp đi lặp lại nhiều tham số cùng lúc.

---

## 1. Bảng tổng hợp các phiên bản đã chạy (v3 – v7)

> Tất cả các phiên bản bên dưới đều train từ đầu (Epoch 1, không warm-start chéo), cùng split **Train=518 / Val=66 / Test=65**, chế độ **Landmark-Only (`use_visual_context=False`)**.  
> **Lưu ý công thức `val_score`:**  
> - **v3–v5:** `0.40*(1 - mae/4) + 0.30*within_1_acc + 0.30*max(0, qwk)`  
> - **v6–v7:** `0.30*(1 - mae/4) + 0.15*(1 - rmse/4.5) + 0.20*within_1_acc + 0.20*max(0, qwk) + 0.15*regime_macro_f1`  
> Vì vậy **không so sánh trực tiếp số `val_score` giữa v3–v5 và v6–v7**.

| Ver | Commit | Best Ep | Val MAE | Val QWK | Val AUC | Val UL QWK | Val UTA QWK | **Test MAE** | **Test RMSE** | **Test QWK** | **Test AUC** | Test F1 | Test RegimeAcc | Test UL MAE / QWK / AUC | **Test UTA MAE / QWK / AUC** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | `2cbb4b6` | 12 | 2.2720 | 0.2973 | 0.7385 | 0.4189 | 0.2028 | 2.0679 | 2.5788 | 0.3072 | 0.5833 | 0.5238 | 0.4615 | 1.5002 / 0.5573 / 0.7083 | 2.5253 / 0.1862 / 0.4826 |
| **v4** | `d3cd27f` | 18 | 1.6984 | 0.5123 | 0.8673 | 0.5616 | **0.4624** | **1.5341** | **1.9063** | 0.4840 | 0.7089 | 0.4390 | **0.4769** | **1.1905 / 0.5436** / 0.7262 | 1.8108 / 0.4635 / 0.7049 |
| **v5** | `e7f3259` | 40 | 1.6483 | 0.5437 | 0.8654 | 0.6379 | 0.4481 | 1.6055 | 1.9576 | **0.4968** | **0.7189** | **0.5106** | 0.4154 | 1.2792 / 0.5000 / **0.7381** | 1.8684 / **0.4907 / 0.7083** |
| v6 | `43d6f2b` | 47 | **1.6324** | **0.5789** | **0.8859** | **0.7051** | 0.4476 | 1.5616 | 1.9368 | 0.4426 | 0.6733 | 0.4390 | 0.4154 | 1.3909 / 0.4064 / 0.6964 | **1.6991** / 0.4605 / 0.6736 |
| **v7** | `29d3604` | 28 | 1.9394 | 0.4154 | 0.7863 | 0.5271 | 0.2977 | 1.9866 | 2.3273 | **0.2523** | **0.5989** | **0.2941** | 0.4154 | 1.3918 / 0.5076 / 0.6726 | **2.4658 / 0.1014 / 0.5347** |

### Khác biệt cấu hình & code chính giữa các bản

- **v3 (`2cbb4b6`)**: `stride_sec=30.0`, `static_shape_ratio=0.20`, `dann=0.05/0.20`, `ema_decay=0.995`, `weight_decay=2e-3`.
- **v4 (`d3cd27f`)**: `stride_sec=45.0`, `static_shape_ratio=0.55`, `dann=0.02/0.15`, `ema_decay=0.990`, tắt nhân chồng `sample_weight` khi bật `subject_balanced_sampling`.
- **v5 (`e7f3259`)**: `static_shape_ratio=0.35`, `weight_decay=3e-3`, `dropout=0.35`, thêm `boundary_margin_weight=0.15`, `uta_ordinal_sigma=1.25` (Gaussian mềm cho UTA), quota sampler `0.60 UL / 0.40 UTA`.
- **v6 (`43d6f2b`)**: `static_shape_ratio=0.15`, `dann=0.15/0.25`, giới hạn regime support trong ordinal UTA (`<=6.0`), thêm boundary penalty cấp window, tách P85 mắt khỏi baseline cơ quan & tách vận tốc mi/miệng cục bộ khỏi xoay đầu, đổi công thức `val_score` thêm RMSE & `regime_macro_f1`.
- **v7 (`29d3604`)**: Đổi 6 thứ cùng lúc — (1) Interval Regression cho UTA với khoảng mid `[4.5, 6.2]` & tâm lệch `(2.5, 5.4, 7.8)` trọng số `0.15`; (2) `boundary_margin_weight=0.05` & ngưỡng `6.30/6.65`; (3) `dann=0.05/0.15`; (4) quota sampler `0.65 UL / 0.35 UTA` & số mũ `(0.70, 0.50, 0.50)`; (5) làm mượt nhân quả 2-frame cho vận tốc mi mắt (`36..47`) + đổi `eye_y_scale`; (6) `final_attn_logits` dùng trọng số từ aggregator + tách nhóm `no_decay` trong AdamW. **Kết quả: hồi quy nặng trên UTA-RLDD** (Test UTA QWK 0.4605 → 0.1014, Val UTA MAE bị chặn sàn ~2.08).

---

## 2. ĐÃ THỬ – KHÔNG LẶP LẠI

1. **Quét qua lại `dann_weight` / `dann_max_alpha`** (`0.05/0.20` → `0.02/0.15` → `0.15/0.25` → `0.05/0.15`): Không tạo khác biệt nhất quán trên test. **Khóa tại `0.15 / 0.25` (baseline v6)** và không tune tiếp nếu không có ablation riêng.
2. **Quét qua lại `static_shape_ratio`** (`0.20` → `0.55` → `0.35` → `0.15`): Dao động 4 lần mà v4/v5/v6 tương đương về mặt thống kê trên 65 mẫu test. **Khóa tại `0.15` (baseline v6)**, ngừng chỉnh.
3. **`uta_reg_mode="interval"` với tâm lệch `(2.5, 5.4, 7.8)` và trần mid `6.2` (v7)**: Lệch trực tiếp với nhãn thật UTA `{2, 6, 8}` mà MAE/QWK chấm điểm, làm gradient thứ tự trong khoảng triệt tiêu. **Cấm bật lại cấu hình tâm lệch này.**
4. **Chỉnh `boundary_margin_weight` và ngưỡng `6.20/6.70` vs `6.30/6.65` cùng lúc với nhiều biến khác**: Chưa thấy lợi ích rõ trên test so với v4 (không dùng boundary). **Giữ mặc định v6 (`0.15`, `6.20/6.70`)** cho đến khi kiểm chứng đơn biến.
5. **Đổi 4–8 biến một lúc + chỉ chạy 1 seed trên tập Test N=65**: Sai số chuẩn của QWK/AUC trên N=65 khoảng `±0.08–0.10`, khiến v4/v5/v6 thực chất tương đương thống kê và che lấp nguyên nhân khi có hồi quy.

---

## 3. Baseline hiện hành & Quy tắc thực nghiệm (v8+)

- **Baseline hiện hành (khóa):** `v6-equivalent` — toàn bộ cờ ablation trong `TrainConfig` để ở mặc định:
  - `uta_reg_mode="point"`
  - `boundary_drowsy_min=6.70`, `boundary_nondrowsy_max=6.20`, `boundary_margin_weight=0.15`
  - `dann_weight=0.15`, `dann_max_alpha=0.25`, `static_shape_ratio=0.15`
  - `sampler_uta_quota=0.40`, `sampler_weight_exponents=(1.0, 1.0, 0.5)`
  - `eye_vel_smooth=False`, `eye_y_scale_mode="v6"`, `window_attn_weighted_logits=False`
- **Quy tắc bắt buộc cho mọi run từ v8 trở đi:**
  1. **Mỗi run chỉ đổi 1 biến** (hoặc 1 nhóm biến gắn liền không tách được) so với baseline đã khóa.
  2. **Kiểm chứng tối thiểu 2 seed** (ví dụ `seed=42` và `seed=7`) trước khi kết luận cải tiến hay hồi quy.
  3. **Tiêu chuẩn chấp nhận thay đổi:** Trung bình ≥ 2 seed cải thiện **> 0.05 QWK** (hoặc vượt ngoài Bootstrap 95% CI) trên **cả Val lẫn Test**, không làm sụt per-dataset UTA/UL-DD.
  4. **Mọi thay đổi hành vi mới phải đi qua cờ trong `TrainConfig`**, mặc định giữ nguyên baseline.
  5. Kiểm tra `run_meta.json`, `val_predictions.csv`, `test_predictions.csv`, `bootstrap_ci_95` trong `test_metrics.json` sau mỗi run.

---

## 4. Hàng đợi thực nghiệm v8

> **Phạm vi triển khai hiện tại (2 run, `save_freq=1` lưu đầy đủ checkpoint):** ưu tiên chạy `v8a` (`seed=42`) và `v8b` (`seed=7`) bằng `train_1.ipynb` để xác nhận tái hiện baseline v6 và đo biên độ nhiễu 2 seed. Các run `v8c–v8e` được giữ ở giai đoạn sau (tùy chọn mở rộng khi có thêm ngân sách).

| Run | Giai đoạn | Thay đổi duy nhất so với baseline v6 | Seed | `save_freq` | Trạng thái | Kết quả tóm tắt |
|---|---|---|---|---|---|---|
| `v8a` | **Hiện tại (Run 1/2)** | Không đổi (baseline v6 qua cờ) | 42 | 1 | Planned | Tái hiện v6 (~0.44+ test QWK, ~0.46 test UTA QWK) |
| `v8b` | **Hiện tại (Run 2/2)** | Không đổi (baseline v6 qua cờ) | 7 | 1 | Planned | Đo biên độ nhiễu giữa 2 seed độc lập |
| `v8c` | Mở rộng (sau) | `eye_vel_smooth=True`, `eye_y_scale_mode="v7"` | 42 (+7) | 1 | Backlog | Kiểm định giả thuyết H3 |
| `v8d` | Mở rộng (sau) | `uta_reg_mode="interval_fixed"` (tâm=2/6/8, mid=[4.5,6.5], cw=0.5) | 42 (+7) | 1 | Backlog | Kiểm định giả thuyết H1 đã sửa tâm |
| `v8e` | Mở rộng (sau) | `sampler_uta_quota=0.35`, `sampler_weight_exponents=(0.70, 0.50, 0.50)` | 42 (+7) | 1 | Backlog | Kiểm định giả thuyết H2 (ưu tiên thấp) |

---

## 5. Mẫu ghi chép cho Run mới (Copy-Paste)

```markdown
### Run: v8x (<mô tả ngắn 1 thay đổi>)
- **Ngày / Commit:** YYYY-MM-DD | `<git_hash>` (check `run_meta.json`)
- **Biến thay đổi duy nhất so với baseline:** `<key>=<value>`
- **Seed:** `<seed>` | **Epochs / Best Ep:** `<ran>/<best>`
- **Val (best.pt):** MAE=`...` | QWK=`...` | AUC=`...` | UL QWK=`...` | UTA QWK=`...`
- **Test (best.pt):** MAE=`...` (CI95: `[..., ...]`) | QWK=`...` (CI95: `[..., ...]`) | AUC=`...` (CI95: `[..., ...]`) | F1=`...`
- **Test Per-Dataset:** UL-DD (MAE=`...`, QWK=`...`, AUC=`...`) | UTA-RLDD (MAE=`...`, QWK=`...`, AUC=`...`)
- **Kết luận:** Accept / Reject / Needs 2nd seed — lý do: ...
```
