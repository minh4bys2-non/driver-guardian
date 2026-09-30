# Cấu trúc cây thư mục lưu trữ trên đĩa (`E:\LSTM\data_processed\`)
Toàn bộ video clip đầu ra sẽ được tổ chức theo cấu trúc phân tầng chuẩn Machine Learning:

```text
E:\LSTM\data_processed\
├── dataset_merged_split.csv               # Bảng tra cứu toàn bộ ~4,882 mẫu
├── dataset_merged_split.json              # File cấu trúc nạp cho PyTorch Dataset
│
├── train\                                 # TẬP HUẤN LUYỆN (80% MẪU ~3,908 CLIPS)
│   ├── 0_alert\                           # Nhãn 0: Lái xe tỉnh táo / bình thường
│   │   ├── sust_n_1.mp4                   (Clip chuẩn 10s từ SUST - Hardlink/Copy)
│   │   ├── vbddd_sub0_normal_driving_1.avi (Clip ~12s từ VBDDD - Hardlink/Copy)
│   │   ├── uta_sub01_f1p1_label0_w00_15.0s.mp4  # [Clip cắt từ UTA-RLDD]
│   │   ├── uta_sub01_f1p1_label0_w01_12.5s.mp4  # [Clip cắt từ UTA-RLDD]
│   │   └── ...
│   └── 1_drowsy\                          # Nhãn 1: Lái xe buồn ngủ / ngủ gật
│       ├── sust_d_1.mp4
│       ├── vbddd_sub0_normal_drowsiness_1.avi
│       ├── uta_sub01_f1p1_label10_w00_17.5s.mp4 # [Clip cắt từ UTA-RLDD]
│       └── ...
│
└── val\                                   # TẬP KIỂM ĐỊNH (20% MẪU ~974 CLIPS)
    ├── 0_alert\                           # Nhãn 0: Kiểm định tỉnh táo
    │   ├── sust_n_5.mp4
    │   ├── vbddd_sub29_normal_driving_1.avi
    │   ├── uta_sub05_f1p1_label0_w00_10.0s.mp4  # [Đối tượng sole uta_sub05]
    │   └── ...
    └── 1_drowsy\                          # Nhãn 1: Kiểm định buồn ngủ
        ├── sust_d_5.mp4
        ├── vbddd_sub29_normal_drowsiness_1.avi
        ├── uta_sub05_f1p1_label10_w00_20.0s.mp4 # [Đối tượng sole uta_sub05]
        └── ...
```