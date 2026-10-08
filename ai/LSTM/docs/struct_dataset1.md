## Cấu trúc dataset 
```
E:\LSTM\data_processed\ (hoặc E:\LSTM\uldd_processed\)
├── dataset_merged_split.csv               # Bảng tra cứu toàn bộ clip sau khi cắt
├── dataset_merged_split.json              # File JSON nạp cho PyTorch Dataset
│
├── train\                                 # TẬP HUẤN LUYỆN (~70% mẫu, ~1.615 clips)
│   ├── 0_alert\                           # Nhãn 0: Tỉnh táo (KSS <= 4.5)
│   │   ├── uldd_subE_s01_kss1.0_w00_25.0s.mp4
│   │   ├── uldd_subE_s01_kss1.0_w01_25.0s.mp4
│   │   ├── uldd_subE_s01_kss1.0_w02_25.0s.mp4
│   │   └── ...
│   └── 1_drowsy\                          # Nhãn 1: Buồn ngủ (KSS >= 7.0)
│       ├── uldd_subE_s08_kss7.0_w00_25.0s.mp4
│       ├── uldd_subE_s08_kss7.0_w01_25.0s.mp4
│       └── ...
│
├── val\                                   # TẬP KIỂM ĐỊNH (~15% mẫu, ~352 clips)
│   ├── 0_alert\                           # Nhãn 0: Kiểm định tỉnh táo
│   │   ├── uldd_subA_s01_kss4.0_w00_25.0s.mp4
│   │   ├── uldd_subH_s02_kss3.0_w00_25.0s.mp4
│   │   └── ...
│   └── 1_drowsy\                          # Nhãn 1: Kiểm định buồn ngủ
│       ├── uldd_subA_s08_kss7.0_w00_25.0s.mp4
│       ├── uldd_subH_s09_kss8.0_w00_25.0s.mp4
│       └── ...
│
└── test\                                  # TẬP KIỂM THỬ ĐỘC LẬP (~15% mẫu, ~352 clips)
    ├── 0_alert\                           # Nhãn 0: Kiểm thử độc lập tỉnh táo
    │   ├── uldd_subD_s01_kss2.0_w00_25.0s.mp4
    │   ├── uldd_subQ_s02_kss3.0_w00_25.0s.mp4
    │   └── ...
    └── 1_drowsy\                          # Nhãn 1: Kiểm thử độc lập buồn ngủ
        ├── uldd_subD_s06_kss8.0_w00_25.0s.mp4
        ├── uldd_subQ_s08_kss7.0_w00_25.0s.mp4
        └── ...
```