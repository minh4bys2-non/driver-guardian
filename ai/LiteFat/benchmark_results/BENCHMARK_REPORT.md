# Báo Cáo Đo Đạc & So Sánh Mô Hình LiteFat v6

- **Ngày thực hiện**: 09/11/2026 11:02:33
- **Tập Validation**: 1,978 mẫu (cân bằng subject UL-DD & SUST)
- **Tập Test**: 1,018 mẫu chuẩn (giữ cố định 100% qua tất cả các thử nghiệm)

---

## 1. Bảng So Sánh Hiệu Năng Trên Tập Test (1,018 Mẫu)

| Độ đo kiểm thử | Stage 1 @0.50 | Stage 1 @Opt (0.42) | Stage 2 @0.50 | Stage 2 @Opt (0.58) | Bước nhảy vọt |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accuracy** | 66.21% | 61.49% | 66.90% | **67.39%** | Đỉnh cao mới ở Stage 2 |
| **Precision** | 66.60% | 57.65% | 64.11% | **66.28%** | Tăng độ tin cậy cảnh báo |
| **Recall (Độ nhạy)** | 62.32% | 80.76% | **73.75%** | 68.14% | Nhận diện buồn ngủ kịp thời |
| **Specificity (Độ đặc hiệu)** | 69.94% | 42.97% | 60.31% | **66.67%** | Giảm báo động giả khi tỉnh táo |
| **F1-Score** | 0.6439 | 0.6728 | **0.6859** | 0.6719 | Cân bằng F1 tốt nhất |
| **AUC-ROC Toàn bộ** | 0.6869 | 0.6869 | **0.7310** | **0.7310** | Khả năng phân tách nhị phân vượt trội |
| **Macro-AUC** | 0.7238 | 0.7238 | **0.8059** | **0.8059** | Cân bằng liên dataset |
| **Average Precision (AP)** | 0.6659 | 0.6659 | **0.6970** | **0.6970** | Độ chính xác trung bình PR |

---

## 2. Bảng So Sánh Hiệu Năng Trên Tập Validation (1,978 Mẫu)

| Độ đo Validation | Stage 1 @0.50 | Stage 1 @Opt (0.42) | Stage 2 @0.50 | Stage 2 @Opt (0.58) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 70.73% | 69.67% | 58.75% | 60.62% |
| **Precision** | 75.76% | 65.58% | 54.77% | 56.72% |
| **Recall** | 55.89% | 75.27% | 72.48% | 70.02% |
| **Specificity** | 84.00% | 64.66% | 46.46% | 52.20% |
| **F1-Score** | 0.6433 | 0.7009 | 0.6240 | 0.6267 |
| **AUC-ROC** | 0.7708 | 0.7708 | 0.6598 | 0.6598 |

---

## 3. Phân Rã Chi Tiết Theo Từng Bộ Dữ Liệu Con (Test Set)

| Bộ dữ liệu con | Số mẫu | Stage 1 AUC | Stage 1 Acc (@Opt) | Stage 2 AUC | Stage 2 Acc (@Opt) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SUST** | 299 | 0.6128 | 59.53% | **0.8379** | **71.91%** | Bối cảnh ảnh mặt tăng sức mạnh vượt bậc |
| **UL-DD** | 664 | 0.7125 | 61.60% | **0.6568** | **64.46%** | Đặc trưng hình thái triệt tiêu shortcut tốt hơn |
| **VBDDD** | 55 | 0.8462 | 70.91% | **0.9231** | **78.18%** | Bối cảnh ảnh mặt tăng sức mạnh vượt bậc |

---

## 4. Danh Mục Biểu Đồ Trực Quan Hóa Đã Xuất

1. [`1_roc_curves.png`](./1_roc_curves.png): Đường cong ROC trên tập Val, Test và phân rã SUST, UL-DD, VBDDD.
2. [`2_pr_curves.png`](./2_pr_curves.png): Đường cong Precision-Recall và chỉ số Average Precision (AP).
3. [`3_confusion_matrices.png`](./3_confusion_matrices.png): Lưới ma trận nhầm lẫn 2x4 chuẩn hóa so sánh trực tiếp cả 2 tập và 2 ngưỡng.
4. [`4_dataset_subject_comparison.png`](./4_dataset_subject_comparison.png): Biểu đồ cột so sánh AUC & Acc theo từng dataset và từng subject kiểm thử.
5. [`5_threshold_sensitivity.png`](./5_threshold_sensitivity.png): Đường cong độ nhạy F1, Acc, Rec, Prec theo dải ngưỡng [0.0 .. 1.0].
6. [`6_metrics_comparison_summary.png`](./6_metrics_comparison_summary.png): Biểu đồ cột nhóm so sánh 6 chỉ số cốt lõi.
7. [`7_summary_dashboard.png`](./7_summary_dashboard.png): Dashboard tổng hợp hợp nhất 4 khung hình đẹp nhất phục vụ trình chiếu.