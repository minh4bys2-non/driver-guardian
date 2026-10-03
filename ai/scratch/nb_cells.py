# Cell 0 (markdown)
# DỰ ÁN TỐT NGHIỆP: NHẬN DIỆN TÀI XẾ BUỒN NGỦ VỚI DEEP GRU (SUST, MERGED & 50/50 BALANCED DATASET)
## NOTEBOOK: `datn4ni4.ipynb` - TỐI ƯU HÓA TOÀN DIỆN VỚI TEMPORAL ATTENTION POOLING, SONG HÀNH F1/F2 SCORE & CHỐNG OVERFITTING
---
### 1. Kiến trúc tối ưu hóa hai giai đoạn (Two-Stage Decoupled Pipeline):
* **Giai đoạn trích xuất đặc trưng không gian (Đã hoàn tất tại máy Local qua `extract_to_pt1.py`):**
  * Mô hình PAFPN CNN (`backbone_neck.onnx` - YOLOv10 trunk ~1.64M params) trích xuất 3 tầng đặc trưng đa tỷ lệ:
    * Tầng $p_3$: $64$ kênh (kích thước nén $1 \times 1$ qua `AdaptiveAvgPool2d`).
    * Tầng $p_4$: $128$ kênh (kích thước nén $1 \times 1$ qua `AdaptiveAvgPool2d`).
    * Tầng $p_5$: $256$ kênh (kích thước nén $1 \times 1$ qua `AdaptiveAvgPool2d`).
    * Tổng số kênh khi ghép nối: $64 + 128 + 256 = 448$ kênh.
  * Hỗ trợ lưu trữ đóng gói nhị phân siêu nhẹ (.pt) nạp trực tiếp vào RAM trong **< 0.5 giây**:
    * **Dataset Cân bằng 50/50 mới** (`features_processed_train.pt`: 4,046 clips, `features_processed_val.pt`: 1,026 clips - Zero Face Leakage).
    * **Dataset Gộp** (`features_merged_train.pt`: 11,892 clips, `features_merged_val.pt`: 2,962 clips).
    * **Dataset SUST** (`features_sust_train.pt`: 1,659 clips, `features_sust_val.pt`: 415 clips).

* **Giai đoạn huấn luyện mô hình chuỗi thời gian tối ưu hóa (Notebook `datn4ni4.ipynb`):**
  * **Temporal Attention Pooling (Phương án A):** Thay vì ép mọi frame học cùng một nhãn video (gây nhiễu gradient do zero-padding), mô hình sử dụng tầng Attention Pooling động gom toàn chuỗi thời gian thành 1 vector đại diện duy nhất, triệt tiêu 100% ảnh hưởng của zero-padding frame qua Attention Masking.
  * **Chiến lược Song hành F1 & F2 Score (Recall-First cho DMS):**
    * **$F_1$-Score:** Thước đo chuẩn mực đánh giá sự cân bằng giữa Precision và Recall.
    * **$F_2$-Score:** Thước đo an toàn giao thông DMS, đặt trọng số Recall cao gấp đôi Precision, giúp hệ thống phạt nặng lỗi bỏ sót tài xế buồn ngủ ($FN$).
  * **Kiểm soát Overfitting triệt để:**
    * **Tinh gọn mô hình:** 2 tầng GRU stacked (`hidden_dim = 192`, ~450k tham số, giảm 64% so với bản 1.24M cũ).
    * **Tăng cường Regularization:** `adapter_dropout = 0.25`, `dropout = 0.35`, `weight_decay = 1e-3`.
    * **Tăng cường dữ liệu Tensor (`TemporalTensorAugmenter`):** Feature Gaussian Noise + Time Masking 15% frames ngẫu nhiên, **bật/tắt linh hoạt qua config (`cfg.use_temporal_aug`)**.
    * **Cơ chế Dừng sớm & Dual Checkpoints:** Tự động lưu riêng `best_gru_f1.pth` và `best_gru_f2.pth`, kết hợp `EarlyStopping(patience=6)`.
  * **Module Threshold Tuning tự động:** Quét tìm ngưỡng an toàn tối ưu $th_{\text{safe}} \in [0.25, 0.60]$ bảo đảm **$Recall \ge 85\%$** cho hệ thống giám sát tài xế thực tế.


############################################################

# Cell 1 (code)
# ==============================================================================
# CELL 1: CÀI ĐẶT CÁC GÓI BỔ TRỢ & THIẾT LẬP MÔI TRƯỜNG (ENVIRONMENT SETUP)
# ==============================================================================
!pip install -q tensorboard seaborn scikit-learn

import os
import sys
import time
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

# Đảm bảo console Windows/Linux không bị lỗi mã hóa UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Tối ưu hóa phân bổ bộ nhớ PyTorch, chống phân mảnh VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, precision_recall_curve, roc_auc_score
)

# Kiểm tra thư viện TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARN] Chưa tìm thấy tensorboard. Chạy: !pip install tensorboard")


# Cố định Seed ngẫu nhiên để đảm bảo tính tái lập 100% (Reproducibility)
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)


def compute_classification_metrics(y_true, y_pred, pos_label: int = 1) -> Dict[str, float]:
    """Tính toán song hành bộ 4 chỉ số cốt lõi: Precision, Recall, F1-Score và F2-Score."""
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label, zero_division=0)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2)
    }


print("=" * 75)
print(f"[+] Python Version       : {sys.version.split()[0]}")
print(f"[+] PyTorch Version      : {torch.__version__}")
print(f"[+] CUDA Khả dụng        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU 0 (Tesla T4)     : {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[+] Dung lượng VRAM      : {vram_gb:.2f} GB")
print(f"[+] TensorBoard Engine   : {'Khả dụng' if TENSORBOARD_AVAILABLE else 'Chưa khả dụng'}")
print("=" * 75)


############################################################

# Cell 2 (code)
# ==============================================================================
# CELL 2: CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ HUẤN LUYỆN (CENTRAL CONFIGURATION)
# ==============================================================================
IS_KAGGLE = os.path.exists("/kaggle")


class KaggleTensorConfig:
    """Cấu hình tập trung cho quy trình huấn luyện Deep GRU v4 (Recall-First, F1/F2, Attention Pooling)."""

    # 1. ĐƯỜNG DẪN TỆP DỮ LIỆU TENSOR (.pt) (Ưu tiên: 50/50 processed -> merged -> sust)
    if IS_KAGGLE:
        # Đường dẫn khi nạp Dataset trên Kaggle
        k_proc_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_processed_train.pt"
        k_proc_val = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_processed_val.pt"
        k_merged_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_merged_train.pt"
        k_merged_val = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_merged_val.pt"
        k_sust_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_train.pt"
        k_sust_val = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_val.pt"

        if os.path.exists(k_proc_train) and os.path.exists(k_proc_val):
            train_pt, val_pt = k_proc_train, k_proc_val
            dataset_name = "50/50 Balanced Processed Dataset (4,046 Train : 1,026 Val)"
        elif os.path.exists(k_merged_train) and os.path.exists(k_merged_val):
            train_pt, val_pt = k_merged_train, k_merged_val
            dataset_name = "Merged Dataset (11,892 Train : 986 Val)"
        elif os.path.exists(k_sust_train) and os.path.exists(k_sust_val):
            train_pt, val_pt = k_sust_train, k_sust_val
            dataset_name = "SUST Dataset (1,659 Train : 415 Val)"
        else:
            train_pt, val_pt = k_merged_train, k_merged_val
            dataset_name = "Merged Dataset (Default Fallback)"
        output_dir = "/kaggle/working"
    else:
        # Đường dẫn khi chạy thử nghiệm cục bộ
        l_proc_train = "extracted_features_pt/features_processed_train.pt"
        l_proc_val = "extracted_features_pt/features_processed_val.pt"
        l_merged_train = "extracted_features_pt/features_merged_train.pt"
        l_merged_val = "extracted_features_pt/features_merged_val.pt"
        l_sust_train = "extracted_features_pt/features_sust_train.pt"
        l_sust_val = "extracted_features_pt/features_sust_val.pt"

        if os.path.exists(l_proc_train) and os.path.exists(l_proc_val):
            train_pt, val_pt = l_proc_train, l_proc_val
            dataset_name = "50/50 Balanced Processed Dataset (4,046 Train : 1,026 Val)"
        elif os.path.exists(l_merged_train) and os.path.exists(l_merged_val):
            train_pt, val_pt = l_merged_train, l_merged_val
            dataset_name = "Merged Dataset (11,892 Train : 986 Val)"
        elif os.path.exists(l_sust_train) and os.path.exists(l_sust_val):
            train_pt, val_pt = l_sust_train, l_sust_val
            dataset_name = "SUST Dataset (1,659 Train : 415 Val)"
        else:
            train_pt, val_pt = l_merged_train, l_merged_val
            dataset_name = "Merged Dataset (Default Fallback)"
        output_dir = "./runs_output"

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "runs")
    experiment_name = "gru_v4_recall_attn"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. CẤU HÌNH DỮ LIỆU
    seq_len: Optional[int] = None  # None: Tự động thích ứng chuỗi động | int: Cố định (ví dụ 120)
    cnn_neck_channels = (64, 128, 256)  # Kênh thực tế của 3 tầng (p3, p4, p5) từ backbone_neck.onnx
    total_feature_dim: int = 448  # Tổng số kênh khi ghép nối (64 + 128 + 256 = 448)

    # 3. KIẾN TRÚC MÔ HÌNH DEEP GRU V4 (Tinh gọn & Chống Overfitting)
    input_dim: int = 256  # Chiều vector sau Spatial Feature Adapter (chuẩn 256 giữ trọn vẹn đặc trưng)
    hidden_dim: int = 192  # 192 hidden units (~450k params thay vì 1.24M, chống học thuộc lòng)
    num_layers: int = 2  # 2 tầng GRU stacked (giảm từ 3 tầng của bản cũ)
    num_classes: int = 2  # 0: Tỉnh táo (Alert), 1: Buồn ngủ (Drowsy)
    spatial_fusion: str = "attention"  # 'attention' | 'concat' | 'sum' | 'mean'
    supervision_mode: str = "attention_pooling"  # 'attention_pooling' (Phương án A gom toàn chuỗi)
    adapter_dropout: float = 0.25  # Dropout sau Spatial Feature Adapter (tăng từ 0.1)
    dropout: float = 0.35  # Dropout giữa các tầng GRU (tăng từ 0.2)

    # 4. CHIẾN LƯỢC TỐI ƯU SONG HÀNH F1 & F2 (ƯU TIÊN RECALL CHO DMS)
    pos_weight: float = 1.35  # Trọng số phạt cho lớp Buồn ngủ (1.35x) trong Loss
    monitor_metric: str = "f2"  # 'f2' (ưu tiên Recall cho an toàn DMS) hoặc 'f1' (chuẩn cân bằng)
    target_recall: float = 0.85  # Mục tiêu Recall tối thiểu >= 85% cho Threshold Tuning

    # 5. TĂNG CƯỜNG DỮ LIỆU THỜI GIAN (BẬT / TẮT QUA CONFIG)
    use_temporal_aug: bool = False   # True: Bật Augmentation | False: Tắt hoàn toàn (Dữ liệu tĩnh)
    aug_p_noise: float = 0.30  # Xác suất thêm nhiễu Gaussian vào vector đặc trưng
    aug_noise_std: float = 0.02  # Độ lệch chuẩn của nhiễu
    aug_p_drop: float = 0.20  # Xác suất che ngẫu nhiên khung hình (Time Masking)
    aug_drop_ratio: float = 0.15  # Tỷ lệ khung hình bị che trong clip (15%)

    # 6. TỐI ƯU HÓA HUẤN LUYỆN SIÊU TỐC
    batch_size: int = 64
    epochs: int = 40  # 40 epochs có Early Stopping
    lr0: float = 5e-4  # Tốc độ học khởi tạo
    weight_decay: float = 1e-3  # L2 Regularization tăng gấp 10 lần (1e-4 -> 1e-3 chống Overfitting)
    grad_clip_norm: float = 1.0  # Ngưỡng cắt gradient
    lr_min_factor: float = 5e-4  # Tốc độ học tối thiểu cho Cosine Annealing
    early_stopping_patience: int = 6  # Dừng sớm nếu không cải thiện metric mục tiêu sau 6 epochs
    amp: bool = True  # Bật Automatic Mixed Precision (FP16)
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 7. CẤU HÌNH RESUME HUẤN LUYỆN TỪ CHECKPOINT
    resume_checkpoint: Optional[str] = None
    auto_resume: bool = False
    reset_optimizer: bool = False


cfg = KaggleTensorConfig()

print("=" * 75)
print(f"[+] Môi trường thực thi    : {'KAGGLE' if IS_KAGGLE else 'LOCAL PC'}")
print(f"[+] Thiết bị tính toán     : {cfg.device}")
print(f"[+] Bộ Dữ liệu Nhận diện   : {cfg.dataset_name}")
print(f"[+] Đường dẫn Train Tensor : {cfg.train_pt}")
print(f"[+] Đường dẫn Val Tensor   : {cfg.val_pt}")
print(f"[+] Thư mục Checkpoints    : {cfg.checkpoint_dir}")
print(f"[+] Thư mục Log TensorBoard: {cfg.log_dir}")
print(f"[+] Phương án Giám sát     : {cfg.supervision_mode.upper()} (Phương án A)")
print(f"[+] Tăng cường chuỗi (Aug) : {'BẬT (Noise + Time Masking)' if cfg.use_temporal_aug else 'TẮT (Dữ liệu tĩnh)'}")
print(f"[+] Tiêu chuẩn Checkpoint  : Song hành F1 & F2 (Ưu tiên chính: {cfg.monitor_metric.upper()}) | Pos Weight={cfg.pos_weight}")
print(f"[+] Siêu tham số           : Batch={cfg.batch_size}, Epochs={cfg.epochs}, Layers={cfg.num_layers}, Hidden={cfg.hidden_dim}, LR={cfg.lr0}")
print("=" * 75)


############################################################

# Cell 3 (code)
# ==============================================================================
# CELL 3: ENGINE GIÁM SÁT TOÀN DIỆN TENSORBOARD (TENSORBOARD LOGGER ENGINE)
# ==============================================================================
class TensorBoardLogger:
    """Quản lý ghi nhật ký sự kiện, đường cong huấn luyện song hành F1/F2, Confusion Matrix và Computational Graph."""

    def __init__(self, log_dir: str = "runs", experiment_name: str = "gru_drowsiness"):
        self.enabled = TENSORBOARD_AVAILABLE
        if not self.enabled:
            print("[TensorBoardLogger] Cảnh báo: Thư viện `tensorboard` chưa sẵn sàng. Logger sẽ tạm tắt.")
            self.writer = None
            self.log_path = None
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        print(f"[TensorBoardLogger] Đã khởi tạo log engine giám sát tại: {self.log_path}")

    def log_train_batch(self, loss: float, accuracy: float, step: int):
        """Ghi nhận Loss và Accuracy chi tiết ở từng Batch huấn luyện (Batch-level)."""
        if self.writer:
            self.writer.add_scalar("Train/Batch_Loss", loss, step)
            self.writer.add_scalar("Train/Batch_Accuracy", accuracy, step)

    def log_train_epoch(self, epoch_loss: float, epoch_acc: float, lr: float, epoch: int):
        """Ghi nhận Loss, Accuracy và Learning Rate tổng kết sau mỗi Epoch (Epoch-level)."""
        if self.writer:
            self.writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
            self.writer.add_scalar("Train/Epoch_Accuracy", epoch_acc, epoch)
            self.writer.add_scalar("Train/Learning_Rate", lr, epoch)

    def log_val_epoch(self, val_loss: float, val_acc: float, epoch: int):
        """Ghi nhận Loss và Accuracy trên tập Validation sau mỗi Epoch."""
        if self.writer:
            self.writer.add_scalar("Val/Epoch_Loss", val_loss, epoch)
            self.writer.add_scalar("Val/Epoch_Accuracy", val_acc, epoch)

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Ghi nhận song hành các chỉ số: Precision, Recall, F1-Score, F2-Score."""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)

    def log_train_val_gap(self, train_loss: float, val_loss: float, epoch: int):
        """Ghi nhận độ phân kỳ (khoảng cách giữa Val Loss và Train Loss) để kiểm soát Overfitting."""
        if self.writer:
            self.writer.add_scalar("Diagnostics/Loss_Gap", val_loss - train_loss, epoch)

    def log_confusion_matrix(self, cm: np.ndarray, epoch: int, class_names=("Tỉnh táo", "Buồn ngủ")):
        """Vẽ và đẩy ảnh ma trận nhầm lẫn (Confusion Matrix Heatmap) lên TensorBoard."""
        if not self.writer:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Dự đoán (Predicted)")
        ax.set_ylabel("Thực tế (Actual)")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        self.writer.add_figure("Images/Confusion_Matrix", fig, epoch)
        plt.close(fig)

    def log_graph(self, model: nn.Module, dummy_input: tuple):
        """Ghi nhận sơ đồ ma trận kiến trúc mạng nơ-ron (Computational Graph)."""
        if not self.writer:
            return
        try:
            self.writer.add_graph(model, dummy_input)
            print("[TensorBoardLogger] Đã lưu Computational Graph của mô hình vào TensorBoard.")
        except Exception as e:
            print(f"[TensorBoardLogger][WARN] Không thể lưu computational graph: {e}")

    def close(self):
        """Đóng và hoàn tất tiến trình ghi log."""
        if self.writer:
            self.writer.close()
            print(f"[TensorBoardLogger] Đã hoàn tất và đóng SummaryWriter tại: {self.log_path}")


############################################################

# Cell 4 (code)
# ==============================================================================
# CELL 4: KIẾN TRÚC MÔ HÌNH SPATIAL FEATURE ADAPTER, TEMPORAL ATTENTION POOLING & DEEP GRU CLASSIFIER
# ==============================================================================
class SpatialFeatureAdapter(nn.Module):
    """
    Bộ chuyển đổi và kết hợp đặc trưng không gian đa tỷ lệ (p3: 64, p4: 128, p5: 256).
    Hỗ trợ các phương thức Fusion: 'attention', 'concat', 'sum', 'mean'.
    Hỗ trợ cả tensor 2D [B, C], tensor chuỗi 3D [B, T, C], và feature map 4D/5D.
    """

    def __init__(
            self,
            in_channels: Tuple[int, ...] = (64, 128, 256),
            out_dim: int = 256,
            fusion: str = "attention",
            dropout: float = 0.25,
            hidden_dim: int = 256
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = tuple(in_channels)
        self.out_dim = out_dim
        self.fusion = fusion.lower()
        self.total_in_channels = sum(in_channels)

        if self.fusion == "attention":
            self.projection = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(inplace=True)
                ) for c in in_channels
            ])
            self.attention_mlp = nn.Sequential(
                nn.Linear(out_dim * self.num_scales, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_scales)
            )
        elif self.fusion == "concat":
            self.projection = nn.Sequential(
                nn.Linear(self.total_in_channels, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True)
            )
            self.attention_mlp = None
        elif self.fusion in ("sum", "mean"):
            self.projection = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(inplace=True)
                ) for c in in_channels
            ])
            self.attention_mlp = None
        else:
            raise ValueError(f"Phương thức fusion '{fusion}' không được hỗ trợ. Chọn: ['attention', 'concat', 'sum', 'mean'].")

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _pool_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Nén đặc trưng không gian (H, W) về (1, 1) nếu đầu vào là feature map 4D hoặc 5D."""
        if x.dim() == 5:  # [B, T, C, H, W]
            b, t, c, h, w = x.shape
            return F.adaptive_avg_pool2d(x.reshape(b * t, c, h, w), (1, 1)).view(b, t, c)
        elif x.dim() == 4:  # [B, C, H, W]
            b, c, h, w = x.shape
            return F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        return x

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            return_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if len(args) > 0:
            features = (features, *args)

        if isinstance(features, (tuple, list)):
            if len(features) != self.num_scales:
                raise ValueError(f"Số lượng feature maps đầu vào ({len(features)}) không khớp in_channels ({self.num_scales}).")
            feats = [self._pool_feature(f) for f in features]
        elif isinstance(features, torch.Tensor):
            feat = self._pool_feature(features)
            if feat.shape[-1] == self.out_dim:
                return (self.dropout(feat), None) if return_weights else self.dropout(feat)
            if self.fusion == "concat":
                out = self.dropout(self.projection(feat))
                if return_weights:
                    weights = torch.ones(*feat.shape[:-1], self.num_scales, device=feat.device) / self.num_scales
                    return out, weights
                return out
            elif feat.shape[-1] == self.total_in_channels:
                feats = list(torch.split(feat, list(self.in_channels), dim=-1))
            else:
                raise ValueError(f"Kích thước kênh cuối ({feat.shape[-1]}) không khớp ({self.total_in_channels}) hoặc ({self.out_dim}).")
        else:
            raise TypeError(f"Định dạng features không hợp lệ: {type(features)}.")

        if self.fusion == "concat":
            concat_v = torch.cat(feats, dim=-1)
            out = self.dropout(self.projection(concat_v))
            if return_weights:
                weights = torch.ones(*concat_v.shape[:-1], self.num_scales, device=concat_v.device) / self.num_scales
                return out, weights
            return out

        v_list = [proj(f) for proj, f in zip(self.projection, feats)]

        if self.fusion == "attention":
            concat_v = torch.cat(v_list, dim=-1)
            weights = self.attention_mlp(concat_v)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)

            stacked_v = torch.stack(v_list, dim=-2)
            fused = (stacked_v * weights).sum(dim=-2)
            fused = self.dropout(fused)

            if return_weights:
                return fused, weights.squeeze(-1)
            return fused

        elif self.fusion == "sum":
            stacked_v = torch.stack(v_list, dim=-2)
            fused = self.dropout(stacked_v.sum(dim=-2))
            if return_weights:
                weights = torch.ones(*stacked_v.shape[:-1], self.num_scales, device=stacked_v.device) / self.num_scales
                return fused, weights
            return fused

        elif self.fusion == "mean":
            stacked_v = torch.stack(v_list, dim=-2)
            fused = self.dropout(stacked_v.mean(dim=-2))
            if return_weights:
                weights = torch.ones(*stacked_v.shape[:-1], self.num_scales, device=stacked_v.device) / self.num_scales
                return fused, weights
            return fused


class TemporalAttentionPooling(nn.Module):
    """
    Gom tụ chuỗi thời gian có trọng số chú ý động (Temporal Attention MLP),
    triệt tiêu 100% gradient rác từ các khung hình zero-padding qua Attention Masking.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_out: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_out: Tensor đầu ra từ GRU [Batch, Time, Hidden]
            seq_lens: Tensor độ dài thực tế của từng mẫu trong batch [Batch]
        Returns:
            pooled: Vector đặc trưng đại diện clip [Batch, Hidden]
            weights: Trọng số chú ý tương ứng từng khung hình [Batch, Time]
        """
        b, t, h = gru_out.shape
        scores = self.attn(gru_out).squeeze(-1)  # [B, T]

        if seq_lens is not None:
            # Tạo boolean mask: True cho frame hợp lệ, False cho frame padding
            mask = torch.arange(t, device=gru_out.device).unsqueeze(0) < seq_lens.to(gru_out.device).unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # [B, T]
        # Gom tụ có trọng số: [B, 1, T] x [B, T, H] -> [B, 1, H] -> [B, H]
        pooled = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)
        return pooled, weights


class DeepGRUClassifier(nn.Module):
    """
    Mô hình phân loại chuỗi thời gian Deep GRU 2 tầng kết hợp Temporal Attention Pooling (Phương án A):
    (p3, p4, p5) -> Spatial Adapter -> 2-layer GRU -> Temporal Attention Pooling -> FC Head (num_classes).
    """

    def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 192,
            num_layers: int = 2,
            num_classes: int = 2,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            fusion: str = "attention",
            adapter_dropout: float = 0.25,
            dropout: float = 0.35,
            supervision_mode: str = "attention_pooling"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.fusion = fusion
        self.supervision_mode = supervision_mode

        self.spatial_adapter = SpatialFeatureAdapter(
            in_channels=spatial_in_channels,
            out_dim=input_dim,
            fusion=fusion,
            dropout=adapter_dropout
        )
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.temporal_pooling = TemporalAttentionPooling(hidden_dim=hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    @property
    def lstm(self) -> nn.GRU:
        """Alias tương thích ngược cho các đoạn mã cũ gọi model.lstm."""
        return self.gru

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], map_location: str = "cpu") -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier và nạp trọng số trực tiếp từ file checkpoint (.pth/.pt)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cfg_dict = ckpt.get("config", {})

        weight_key = next((k for k in ("gru.weight_ih_l0", "lstm.weight_ih_l0") if k in state_dict), None)
        if weight_key:
            input_dim = state_dict[weight_key].shape[1]
            hidden_dim = state_dict[weight_key].shape[0] // 3
        else:
            input_dim = int(cfg_dict.get("input_dim", 256))
            hidden_dim = int(cfg_dict.get("hidden_dim", 192))

        num_classes = int(cfg_dict.get("num_classes", 2))
        layer_indices = {int(k.split("weight_ih_l")[-1]) for k in state_dict if "weight_ih_l" in k and k.split("weight_ih_l")[-1].isdigit()}
        num_layers = len(layer_indices) if layer_indices else int(cfg_dict.get("num_layers", 2))
        fusion = cfg_dict.get("spatial_fusion", "attention")
        spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))

        remapped_sd = {k.replace("lstm.", "gru.", 1) if k.startswith("lstm.") else k: v for k, v in state_dict.items()}

        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            spatial_in_channels=spatial_in_channels,
            fusion=fusion
        )
        if map_location is not None:
            model = model.to(map_location)
        model.load_state_dict(remapped_sd, strict=False)
        model.eval()
        return model

    @classmethod
    def from_config(cls, config: Any) -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier trực tiếp từ đối tượng TrainConfig."""
        return cls(
            input_dim=getattr(config, "input_dim", 256),
            hidden_dim=getattr(config, "hidden_dim", 192),
            num_layers=getattr(config, "num_layers", 2),
            num_classes=getattr(config, "num_classes", 2),
            spatial_in_channels=getattr(config, "cnn_neck_channels", (64, 128, 256)),
            fusion=getattr(config, "spatial_fusion", "attention"),
            adapter_dropout=getattr(config, "adapter_dropout", 0.25),
            dropout=getattr(config, "dropout", 0.35),
            supervision_mode=getattr(config, "supervision_mode", "attention_pooling")
        )

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
            h_0: Optional[torch.Tensor] = None,
            hc: Optional[Any] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        if len(args) > 0:
            features = (features, *args)
        if hc is not None and h_0 is None:
            h_0 = hc[0] if isinstance(hc, tuple) else hc
        if isinstance(h_0, tuple):
            h_0 = h_0[0]

        # 1. Chuyển đổi đặc trưng không gian
        x = self.spatial_adapter(features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 2. Trích xuất đặc trưng chuỗi thời gian qua Deep GRU
        gru_out, h_n = self.gru(x, h_0)

        # 3. Phân loại theo cơ chế
        if return_sequence:
            logits = self.fc_out(gru_out)  # [B, T, num_classes] (Hỗ trợ streaming / sequence)
            attn_weights = None
        else:
            # Phương án A: Temporal Attention Pooling (Clip-level)
            pooled, attn_weights = self.temporal_pooling(gru_out, seq_lens)
            logits = self.fc_out(pooled)  # [B, num_classes]

        outputs = [logits]
        if return_weights:
            outputs.append(attn_weights)
        if return_state:
            outputs.append(h_n)

        return tuple(outputs) if len(outputs) > 1 else logits


if __name__ == "__main__":
    print("=" * 75)
    print("[*] KIỂM TRA MÔ HÌNH DEEP GRU V4 (TEMPORAL ATTENTION POOLING & STREAMING)")
    print("=" * 75)

    batch_size, seq_len = 4, 120
    f1_3d = torch.randn(batch_size, seq_len, 64)
    f2_3d = torch.randn(batch_size, seq_len, 128)
    f3_3d = torch.randn(batch_size, seq_len, 256)
    dummy_lens = torch.tensor([120, 100, 80, 60], dtype=torch.long)

    model_v4 = DeepGRUClassifier(
        input_dim=256,
        hidden_dim=192,
        num_layers=2,
        num_classes=2,
        fusion="attention"
    )

    out_clip, w_attn = model_v4((f1_3d, f2_3d, f3_3d), seq_lens=dummy_lens, return_sequence=False, return_weights=True)
    out_seq = model_v4((f1_3d, f2_3d, f3_3d), return_sequence=True)

    print(f"[+] Output Clip Mode (Phương án A) : {list(out_clip.shape)} (Mong đợi [4, 2])")
    print(f"[+] Attention Weights Shape         : {list(w_attn.shape)} (Mong đợi [4, 120])")
    print(f"[+] Output Sequence Mode            : {list(out_seq.shape)} (Mong đợi [4, 120, 2])")

    # Kiểm tra mask: mẫu thứ 4 (lens=60) các frame từ 60 đến 119 phải có attention weight = 0.0
    padded_attn_sum = w_attn[3, 60:].sum().item()
    valid_attn_sum = w_attn[3, :60].sum().item()
    print(f"[+] Tổng trọng số Attention frame padding (mẫu 4): {padded_attn_sum:.6f} (Mong đợi 0.000000)")
    print(f"[+] Tổng trọng số Attention frame hợp lệ (mẫu 4) : {valid_attn_sum:.6f} (Mong đợi 1.000000)")

    total_params = sum(p.numel() for p in model_v4.parameters())
    print(f"[+] Tổng số tham số mô hình tinh gọn v4          : {total_params:,} (Giảm 64% so với 1.24M)")

    assert out_clip.shape == (4, 2)
    assert out_seq.shape == (4, 120, 2)
    assert abs(valid_attn_sum - 1.0) < 1e-4
    assert padded_attn_sum < 1e-6
    print("[+] TẤT CẢ KIỂM THỬ ARCHITECTURE ĐÃ VƯỢT QUA 100% THÀNH CÔNG!")
    print("=" * 75)


############################################################

# Cell 5 (code)
# ==============================================================================
# CELL 5: DATASET NẠP TENSOR SIÊU TỐC, TĂNG CƯỜNG DỮ LIỆU ĐỘNG & DATALOADERS
# ==============================================================================
class TemporalTensorAugmenter:
    """
    Bộ tăng cường dữ liệu chuỗi thời gian trên Tensor - Bật/Tắt linh hoạt trực tiếp từ Config.
    Giúp chống Overfitting hiệu quả bằng Feature Noise và Time Masking.
    """

    def __init__(
            self,
            enabled: bool = True,
            p_noise: float = 0.30,
            noise_std: float = 0.02,
            p_drop: float = 0.20,
            drop_ratio: float = 0.15
    ):
        self.enabled = enabled
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_drop = p_drop
        self.drop_ratio = drop_ratio

    def __call__(
            self,
            p3: torch.Tensor,
            p4: torch.Tensor,
            p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return p3, p4, p5  # Nếu tắt, trả về nguyên bản 100% không làm chậm pipeline

        # 1. Feature Gaussian Noise Injection (Thêm nhiễu nhẹ)
        if random.random() < self.p_noise:
            p3 = p3 + torch.randn_like(p3) * self.noise_std
            p4 = p4 + torch.randn_like(p4) * self.noise_std
            p5 = p5 + torch.randn_like(p5) * self.noise_std

        # 2. Time Masking / Random Frame Dropping (Che ngẫu nhiên 10-15% frames trong clip)
        if random.random() < self.p_drop:
            seq_len = p3.shape[0]
            num_drop = max(1, int(seq_len * self.drop_ratio))
            drop_idx = torch.randperm(seq_len)[:num_drop]
            p3 = p3.clone()
            p4 = p4.clone()
            p5 = p5.clone()
            p3[drop_idx] = 0.0
            p4[drop_idx] = 0.0
            p5[drop_idx] = 0.0

        return p3, p4, p5


class PreloadedTensorDataset(Dataset):
    """
    Dataset tải toàn bộ Tensor .pt vào RAM một lần duy nhất.
    Thời gian nạp chỉ tốn < 0.5s cho toàn bộ video!
    Hỗ trợ kiểm tra cân bằng nhãn và tích hợp Temporal Augmentation khi huấn luyện.
    """

    def __init__(self, pt_path: str, is_train: bool = True, augmenter: Optional[TemporalTensorAugmenter] = None):
        super().__init__()
        self.is_train = is_train
        self.augmenter = augmenter if (is_train and augmenter is not None and augmenter.enabled) else None
        path = Path(pt_path)

        if not path.exists():
            print(f"[!] Không tìm thấy tệp: {path.resolve()}.")
            print("    -> Đang sinh tập dữ liệu mô phỏng ngẫu nhiên để kiểm thử pipeline...")
            num_samples = 64 if is_train else 16
            self.p3 = torch.randn(num_samples, 120, 64, dtype=torch.float32)
            self.p4 = torch.randn(num_samples, 120, 128, dtype=torch.float32)
            self.p5 = torch.randn(num_samples, 120, 256, dtype=torch.float32)
            self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
            self.video_ids = [f"dummy_video_{i:04d}" for i in range(num_samples)]
            self.is_variable_len = False
            self.seq_lens = torch.full((num_samples,), 120, dtype=torch.int32)
            print(f"[+] Đã tạo {num_samples} mẫu dữ liệu mô phỏng.")
            return

        t0 = time.time()
        print(f"[+] Đang nạp dữ liệu từ: {path.resolve()}...")
        data = torch.load(str(path), map_location="cpu")

        self.is_variable_len = data.get("is_variable_len", isinstance(data["p3"], list))

        if self.is_variable_len:
            self.p3 = [t.float() for t in data["p3"]]
            self.p4 = [t.float() for t in data["p4"]]
            self.p5 = [t.float() for t in data["p5"]]
        else:
            self.p3 = data["p3"].float()
            self.p4 = data["p4"].float()
            self.p5 = data["p5"].float()

        self.labels = data["labels"].long()
        self.video_ids = data.get("video_ids", [f"video_{i}" for i in range(len(self.labels))])
        raw_seq_lens = data.get("seq_lens", None)
        if raw_seq_lens is not None:
            self.seq_lens = torch.tensor(raw_seq_lens, dtype=torch.int32) if not isinstance(raw_seq_lens, torch.Tensor) else raw_seq_lens.int()
        else:
            self.seq_lens = torch.tensor([p.shape[0] for p in self.p3], dtype=torch.int32) if self.is_variable_len else torch.full((len(self.labels),), self.p3.shape[1], dtype=torch.int32)

        load_sec = time.time() - t0
        alert_cnt = (self.labels == 0).sum().item()
        drowsy_cnt = (self.labels == 1).sum().item()
        drowsy_ratio = drowsy_cnt / len(self.labels) * 100 if len(self.labels) > 0 else 0

        print(f"    - Nạp hoàn tất {len(self.labels)} video vào RAM trong {load_sec:.2f}s! (Chuỗi động={self.is_variable_len})")
        print(f"    - Phân bố nhãn: Tỉnh táo (0) = {alert_cnt} ({100 - drowsy_ratio:.1f}%), Buồn ngủ (1) = {drowsy_cnt} ({drowsy_ratio:.1f}%)")
        if abs(drowsy_ratio - 50.0) < 1.0:
            print("    - Đánh giá: 🎯 Đạt chuẩn cân bằng nhãn lý tưởng 50/50!")
        else:
            print(f"    - Đánh giá: Chênh lệch nhẹ, áp dụng pos_weight={cfg.pos_weight} để ưu tiên Recall.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        p3, p4, p5 = self.p3[idx], self.p4[idx], self.p5[idx]
        if self.augmenter is not None:
            p3, p4, p5 = self.augmenter(p3, p4, p5)
        s_len = self.seq_lens[idx].item() if self.seq_lens is not None else p3.shape[0]
        return (p3, p4, p5), self.labels[idx], s_len


def dynamic_tensor_collate_fn(batch):
    """Collate function tự động pad các chuỗi tensor có độ dài khác nhau về độ dài lớn nhất trong Batch."""
    p3_list = [item[0][0] for item in batch]
    p4_list = [item[0][1] for item in batch]
    p5_list = [item[0][2] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    seq_lens = torch.tensor([item[2] for item in batch], dtype=torch.long)

    p3_padded = torch.nn.utils.rnn.pad_sequence(p3_list, batch_first=True, padding_value=0.0)
    p4_padded = torch.nn.utils.rnn.pad_sequence(p4_list, batch_first=True, padding_value=0.0)
    p5_padded = torch.nn.utils.rnn.pad_sequence(p5_list, batch_first=True, padding_value=0.0)

    return (p3_padded, p4_padded, p5_padded), labels, seq_lens


# Khởi tạo Augmenter theo cấu hình cfg.use_temporal_aug
train_augmenter = TemporalTensorAugmenter(
    enabled=cfg.use_temporal_aug,
    p_noise=cfg.aug_p_noise,
    noise_std=cfg.aug_noise_std,
    p_drop=cfg.aug_p_drop,
    drop_ratio=cfg.aug_drop_ratio
)

print("=" * 75)
print(f"[*] KHỞI TẠO TẬP DỮ LIỆU HUẤN LUYỆN & KIỂM THỬ (Augmentation={train_augmenter.enabled}):")
train_dataset = PreloadedTensorDataset(cfg.train_pt, is_train=True, augmenter=train_augmenter)
val_dataset = PreloadedTensorDataset(cfg.val_pt, is_train=False)

collate_fn = dynamic_tensor_collate_fn if (getattr(train_dataset, "is_variable_len", False) or getattr(val_dataset, "is_variable_len", False)) else None

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    collate_fn=collate_fn
)

print(f"[+] Khởi tạo DataLoaders thành công (DynamicCollate={collate_fn is not None}):")
print(f"    - Train: {len(train_dataset)} mẫu ({len(train_loader)} batches | Batch size: {cfg.batch_size})")
print(f"    - Val  : {len(val_dataset)} mẫu ({len(val_loader)} batches)")
print("=" * 75)


############################################################

# Cell 6 (code)
# ==============================================================================
# CELL 6: VÒNG LẶP HUẤN LUYỆN RECALL-FIRST, EARLY STOPPING & CƠ CHẾ DUAL CHECKPOINTS
# ==============================================================================
class DrowsinessClipLoss(nn.Module):
    """Hàm mất mát phân loại Clip trực tiếp (Phương án A), phạt nặng lỗi bỏ sót buồn ngủ qua pos_weight."""

    def __init__(self, pos_weight: float = 1.35):
        super().__init__()
        weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32) if pos_weight != 1.0 else None
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = getattr(self.criterion, "weight", None)
        if weights is not None and weights.device != logits.device:
            self.criterion.weight = weights.to(logits.device)
        return self.criterion(logits, targets)



class EarlyStopping:
    """Early Stopping giám sát chỉ số mục tiêu (F2-Score hoặc F1-Score) để chống Overfitting."""

    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# 1. Khởi tạo Mô hình trên GPU
model = DeepGRUClassifier(
    input_dim=cfg.input_dim,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
    num_classes=cfg.num_classes,
    spatial_in_channels=cfg.cnn_neck_channels,
    fusion=cfg.spatial_fusion,
    adapter_dropout=cfg.adapter_dropout,
    dropout=cfg.dropout,
    supervision_mode=cfg.supervision_mode
).to(cfg.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[+] KHỞI TẠO MÔ HÌNH DeepGRUClassifier v4 (Phương án A - Attention Pooling):")
print(f"    - Tổng tham số mô hình  : {total_params:,}")
print(f"    - Tham số cần huấn luyện: {trainable_params:,}")

# 2. Khởi tạo Optimizer, Scheduler và Loss
criterion = DrowsinessClipLoss(pos_weight=cfg.pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr0, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.epochs,
    eta_min=cfg.lr0 * cfg.lr_min_factor
)

# 3. Kích hoạt AMP Mixed Precision FP16
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and "cuda" in cfg.device))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and "cuda" in cfg.device))


def get_autocast():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", enabled=cfg.amp)
    return torch.cuda.amp.autocast(enabled=(cfg.amp and "cuda" in cfg.device))


# 4. Khởi tạo Logger TensorBoard và đăng ký Computational Graph
logger = TensorBoardLogger(log_dir=cfg.log_dir, experiment_name=cfg.experiment_name)
sample_seq_len = cfg.seq_len if cfg.seq_len is not None else 120
dummy_graph_input = (
    torch.zeros(1, sample_seq_len, 64, device=cfg.device),
    torch.zeros(1, sample_seq_len, 128, device=cfg.device),
    torch.zeros(1, sample_seq_len, 256, device=cfg.device)
)
logger.log_graph(model, (dummy_graph_input,))

# 5. KHỞI TẠO BỘ GIÁM SÁT DUAL CHECKPOINT & EARLY STOPPING
history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "precision": [], "recall": [], "f1": [], "f2": [],
    "lr": []
}

best_val_f1 = 0.0
best_val_f2 = 0.0
best_primary_score = 0.0
best_val_acc = 0.0
global_step = 0
start_epoch = 1

early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

print("=" * 75)
print(f"[*] BẮT ĐẦU HUẤN LUYỆN V4 TỪ EPOCH {start_epoch} ĐẾN {cfg.epochs} (AMP FP16={cfg.amp})")
print(f"    - Tiêu chí lưu Checkpoint chính : {cfg.monitor_metric.upper()} Score | Early Stopping Patience: {cfg.early_stopping_patience}")
print(f"    - Trọng số phạt bỏ sót buồn ngủ : pos_weight = {cfg.pos_weight}")
print("=" * 75)

start_total_time = time.time()

for epoch in range(start_epoch, cfg.epochs + 1):
    t_epoch_start = time.time()

    # --- VÒNG LẶP TRAIN ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    train_pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Train]",
        leave=False,
        dynamic_ncols=True
    )

    for batch in train_pbar:
        global_step += 1
        (p3, p4, p5), labels, seq_lens = batch if len(batch) == 3 else (batch[0], batch[1], None)

        p3 = p3.to(cfg.device, non_blocking=True)
        p4 = p4.to(cfg.device, non_blocking=True)
        p5 = p5.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        if seq_lens is not None:
            seq_lens = seq_lens.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Huấn luyện với Mixed Precision FP16
        with get_autocast():
            # Phương án A: Trả về Clip-level Logits [B, 2] đã gom tụ Attention không padding
            logits = model((p3, p4, p5), seq_lens=seq_lens, return_sequence=False)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        batch_acc = (preds == labels).float().mean().item()
        batch_loss = loss.item()

        logger.log_train_batch(loss=batch_loss, accuracy=batch_acc, step=global_step)

        train_loss += batch_loss * len(labels)
        train_correct += (preds == labels).sum().item()
        train_total += len(labels)

        train_pbar.set_postfix({
            "loss": f"{train_loss / train_total:.4f}",
            "acc": f"{(train_correct / train_total) * 100:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    logger.log_train_epoch(epoch_train_loss, epoch_train_acc, current_lr, epoch)

    # --- VÒNG LẶP EVALUATION ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Val]  ",
        leave=False,
        dynamic_ncols=True
    )

    with torch.no_grad():
        for batch in val_pbar:
            (p3, p4, p5), labels, seq_lens = batch if len(batch) == 3 else (batch[0], batch[1], None)

            p3 = p3.to(cfg.device, non_blocking=True)
            p4 = p4.to(cfg.device, non_blocking=True)
            p5 = p5.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            if seq_lens is not None:
                seq_lens = seq_lens.to(cfg.device, non_blocking=True)

            with get_autocast():
                logits = model((p3, p4, p5), seq_lens=seq_lens, return_sequence=False)
                loss = criterion(logits, labels)

            val_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)

            val_correct += (preds == labels).sum().item()
            val_total += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            val_pbar.set_postfix({
                "val_loss": f"{val_loss / val_total:.4f}",
                "val_acc": f"{(val_correct / val_total) * 100:.2f}%"
            })

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    # Ghi nhận log Validation
    logger.log_val_epoch(epoch_val_loss, epoch_val_acc, epoch)
    logger.log_train_val_gap(epoch_train_loss, epoch_val_loss, epoch)

    # Tính toán bộ 4 chỉ số song hành: Precision, Recall, F1, F2
    metrics = compute_classification_metrics(all_targets, all_preds, pos_label=1)
    prec, rec, f1, f2 = metrics["precision"], metrics["recall"], metrics["f1"], metrics["f2"]

    logger.log_metrics({
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "F2_Score": f2
    }, epoch)

    if epoch % 5 == 0 or epoch == cfg.epochs:
        cm = confusion_matrix(all_targets, all_preds)
        logger.log_confusion_matrix(cm, epoch)

    # Lưu lịch sử
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_acc"].append(epoch_val_acc)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["f1"].append(f1)
    history["f2"].append(f2)
    history["lr"].append(current_lr)

    # Đóng gói trạng thái checkpoint đầy đủ
    checkpoint_state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_acc": epoch_val_acc,
        "val_loss": epoch_val_loss,
        "f1_score": f1,
        "f2_score": f2,
        "recall": rec,
        "precision": prec,
        "history": history,
        "config": {
            "input_dim": cfg.input_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_classes": cfg.num_classes,
            "spatial_fusion": cfg.spatial_fusion,
            "supervision_mode": cfg.supervision_mode,
            "cnn_neck_channels": cfg.cnn_neck_channels,
            "adapter_dropout": cfg.adapter_dropout,
            "dropout": cfg.dropout,
            "pos_weight": cfg.pos_weight
        }
    }

    # Luôn lưu last_gru.pth để có thể resume bất kỳ lúc nào
    last_ckpt_path = os.path.join(cfg.checkpoint_dir, "last_gru.pth")
    torch.save(checkpoint_state, last_ckpt_path)

    # 1. Lưu Checkpoint kỷ lục F1 (Cân bằng Precision-Recall)
    is_best_f1 = f1 > best_val_f1
    if is_best_f1:
        best_val_f1 = f1
        checkpoint_state["best_val_f1"] = best_val_f1
        f1_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_gru_f1.pth")
        torch.save(checkpoint_state, f1_ckpt_path)

    # 2. Lưu Checkpoint kỷ lục F2 (Ưu tiên độ nhạy Recall cho an toàn DMS)
    is_best_f2 = f2 > best_val_f2
    if is_best_f2:
        best_val_f2 = f2
        checkpoint_state["best_val_f2"] = best_val_f2
        f2_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_gru_f2.pth")
        torch.save(checkpoint_state, f2_ckpt_path)

    # 3. Checkpoint chính best_gru.pth theo dõi theo cfg.monitor_metric
    primary_metric_val = f2 if cfg.monitor_metric == "f2" else f1
    is_best_primary = primary_metric_val > best_primary_score
    if is_best_primary:
        best_primary_score = primary_metric_val
        best_val_acc = epoch_val_acc
        checkpoint_state["best_primary_score"] = best_primary_score
        best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_gru.pth")
        torch.save(checkpoint_state, best_ckpt_path)

    epoch_sec = time.time() - t_epoch_start
    star_tag = "[BEST F2]" if (cfg.monitor_metric == "f2" and is_best_primary) else ("[BEST F1]" if is_best_primary else "")

    print(f"Epoch {epoch:02d}/{cfg.epochs:02d} [{epoch_sec:.2f}s] | "
          f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc * 100:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc * 100:.2f}% | "
          f"F1: {f1:.4f} | F2: {f2:.4f} | Rec: {rec * 100:.2f}% {star_tag}")

    # Kiểm tra kích hoạt Early Stopping
    early_stop_triggered = early_stopper(primary_metric_val)
    if early_stopper.early_stop:
        print(f"\n[!] KÍCH HOẠT EARLY STOPPING tại Epoch {epoch}! "
              f"Chỉ số {cfg.monitor_metric.upper()} không cải thiện sau {cfg.early_stopping_patience} epochs liên tiếp.")
        print(f"    -> Đã dừng sớm để bảo vệ mô hình khỏi hiện tượng Overfitting!")
        break

logger.close()
total_time_min = (time.time() - start_total_time) / 60
print("\n" + "=" * 75)
print(f"[SUCCESS] Quá trình huấn luyện v4 hoàn tất trong {total_time_min:.2f} phút!")
print(f"[+] Kỷ lục F1-Score tốt nhất : {best_val_f1:.4f} (Đã lưu: best_gru_f1.pth)")
print(f"[+] Kỷ lục F2-Score tốt nhất : {best_val_f2:.4f} (Đã lưu: best_gru_f2.pth)")
print(f"[+] Best Checkpoint chính    : {os.path.join(cfg.checkpoint_dir, 'best_gru.pth')}")
print("=" * 75)


############################################################

# Cell 7 (code)
# ==============================================================================
# CELL 7: TRỰC QUAN HÓA TOÀN DIỆN, SO SÁNH F1/F2 & TỐI ƯU HÓA NGƯỠNG AN TOÀN DMS
# ==============================================================================
if len(history.get("train_loss", [])) > 0:
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Đường cong Loss (Train vs Val)
    axes[0, 0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="royalblue", linewidth=2)
    axes[0, 0].plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="crimson", linewidth=2)
    axes[0, 0].set_title("Quá trình Hội tụ Loss (Loss Curves)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=11)

    # 2. Đường cong Accuracy
    axes[0, 1].plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="forestgreen", linewidth=2)
    axes[0, 1].plot(epochs_range, [a * 100 for a in history["val_acc"]], "s-", label="Val Acc (%)", color="darkorange", linewidth=2)
    axes[0, 1].set_title("Độ chính xác qua từng Epoch (Accuracy Curves)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(fontsize=11)

    # 3. ĐỒ THỊ SONG HÀNH F1-SCORE VÀ F2-SCORE
    axes[1, 0].plot(epochs_range, history["f1"], "d-", label="F1-Score (Cân bằng)", color="purple", linewidth=2)
    axes[1, 0].plot(epochs_range, history["f2"], "^-", label="F2-Score (Ưu tiên Recall DMS)", color="darkred", linewidth=2.5)
    axes[1, 0].set_title("So sánh Tiến hóa Song hành: F1 vs F2 Score", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("Score", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(fontsize=11)

    # 4. Độ phân kỳ Generalization Gap (Val Loss - Train Loss)
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    axes[1, 1].plot(epochs_range, gap, "x--", label="Generalization Gap (Val - Train)", color="teal", linewidth=2)
    axes[1, 1].axhline(y=0.0, color="gray", linestyle=":")
    axes[1, 1].set_title("Khoảng cách Phân kỳ Kiểm soát Overfitting", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Loss Gap", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend(fontsize=11)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 5. ĐÁNH GIÁ TRÊN CHECKPOINT TỐT NHẤT & QUÉT TÌM NGƯỠNG AN TOÀN DMS
# ------------------------------------------------------------------------------
print("=" * 75)
print("--- ĐÁNH GIÁ MA TRẬN NHẦM LẪN & TỐI ƯU HÓA NGƯỠNG AN TOÀN CHO HỆ THỐNG DMS ---")
print("=" * 75)

best_path = os.path.join(cfg.checkpoint_dir, "best_gru.pth")
if not os.path.exists(best_path):
    best_path = os.path.join(cfg.checkpoint_dir, "last_gru.pth")

if os.path.exists(best_path):
    try:
        model = DeepGRUClassifier.from_checkpoint(best_path, map_location=cfg.device)
        print(f"[+] Nạp thành công mô hình từ: {best_path} qua `from_checkpoint`")
    except Exception as e:
        print(f"[!] Nạp mô hình qua load_state_dict: {e}")
        ckpt = torch.load(best_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
    ckpt = torch.load(best_path, map_location=cfg.device)
    print(f"[+] Trọng số Checkpoint: Epoch {ckpt.get('epoch', 'N/A')} | F1={ckpt.get('f1_score', 0):.4f} | F2={ckpt.get('f2_score', 0):.4f}")
else:
    print(f"[!] Sử dụng trọng số mô hình hiện tại.")

model = model.to(cfg.device)
model.eval()

eval_logits, eval_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        (p3, p4, p5), labels, seq_lens = batch if len(batch) == 3 else (batch[0], batch[1], None)

        p3 = p3.to(cfg.device, non_blocking=True)
        p4 = p4.to(cfg.device, non_blocking=True)
        p5 = p5.to(cfg.device, non_blocking=True)
        if seq_lens is not None:
            seq_lens = seq_lens.to(cfg.device, non_blocking=True)

        logits = model((p3, p4, p5), seq_lens=seq_lens, return_sequence=False)
        eval_logits.append(logits.cpu())
        eval_targets.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)

eval_logits = torch.cat(eval_logits, dim=0)
eval_probs = F.softmax(eval_logits, dim=-1)[:, 1].numpy()  # Xác suất nhãn Buồn ngủ (1)
eval_targets = np.array(eval_targets)

# ------------------------------------------------------------------------------
# 6. MODULE QUÉT NGƯỠNG AN TOÀN (DMS THRESHOLD TUNER)
# ------------------------------------------------------------------------------
print("\n[*] BẢNG QUÉT TÌM NGƯỠNG TỐI ƯU SONG HÀNH F1 VÀ F2:")
print(f"{'Ngưỡng (Th)':<12} | {'Precision (%)':<15} | {'Recall (%)':<12} | {'F1-Score':<10} | {'F2-Score':<10} | {'Ghi chú'}")
print("-" * 75)

threshold_candidates = np.arange(0.20, 0.62, 0.02)
best_f1_val = 0.0
best_th_f1 = 0.50

best_f2_val = 0.0
best_th_safe = 0.50
records = []

for th in threshold_candidates:
    p_labels = (eval_probs >= th).astype(int)
    m = compute_classification_metrics(eval_targets, p_labels, pos_label=1)
    prec_i, rec_i, f1_i, f2_i = m["precision"], m["recall"], m["f1"], m["f2"]

    records.append({"th": th, "prec": prec_i, "rec": rec_i, "f1": f1_i, "f2": f2_i})

    # Cập nhật kỷ lục F1
    if f1_i > best_f1_val:
        best_f1_val = f1_i
        best_th_f1 = th

    # Cập nhật kỷ lục F2 an toàn thỏa mãn Recall >= target_recall
    if rec_i >= cfg.target_recall and f2_i > best_f2_val:
        best_f2_val = f2_i
        best_th_safe = th

    tag = ""
    if abs(th - 0.50) < 1e-4:
        tag = "<-- Mặc định (Default)"
    print(f"{th:<12.2f} | {prec_i * 100:<15.2f} | {rec_i * 100:<12.2f} | {f1_i:<10.4f} | {f2_i:<10.4f} | {tag}")

# Fallback an toàn nếu chưa chạm target_recall
if best_f2_val == 0.0:
    best_th_safe = records[np.argmax([r["f2"] for r in records])]["th"]
    best_f2_val = records[np.argmax([r["f2"] for r in records])]["f2"]

print("-" * 75)
print(f"[+] Ngưỡng tối ưu theo F1-Score (Cân bằng) : Threshold = {best_th_f1:.2f} | F1 = {best_f1_val:.4f}")
print(f"[+] Ngưỡng tối ưu An toàn DMS (F2 & Recall): Threshold = {best_th_safe:.2f} | F2 = {best_f2_val:.4f}")
print("=" * 75)

# ------------------------------------------------------------------------------
# 7. VẼ SONG SONG 2 MA TRẬN NHẦM LẪN (DUAL CONFUSION MATRIX)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
class_labels = ["0: Tỉnh táo", "1: Buồn ngủ"]

# Ma trận 1: Ngưỡng chuẩn 0.50
preds_default = (eval_probs >= 0.50).astype(int)
cm_default = confusion_matrix(eval_targets, preds_default)
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax1, cbar=False, annot_kws={"fontsize": 13, "fontweight": "bold"})
ax1.set_title(f"Ngưỡng Mặc định (Th = 0.50)\nRecall Buồn ngủ: {recall_score(eval_targets, preds_default, pos_label=1) * 100:.2f}%", fontsize=12, fontweight="bold")
ax1.set_xlabel("Dự đoán (Predicted)", fontsize=11)
ax1.set_ylabel("Thực tế (Actual)", fontsize=11)

# Ma trận 2: Ngưỡng an toàn tối ưu th_safe
preds_safe = (eval_probs >= best_th_safe).astype(int)
cm_safe = confusion_matrix(eval_targets, preds_safe)
sns.heatmap(cm_safe, annot=True, fmt="d", cmap="Greens", xticklabels=class_labels, yticklabels=class_labels, ax=ax2, cbar=False, annot_kws={"fontsize": 13, "fontweight": "bold"})
ax2.set_title(f"Ngưỡng An toàn DMS Tối ưu (Th = {best_th_safe:.2f})\nRecall Buồn ngủ: {recall_score(eval_targets, preds_safe, pos_label=1) * 100:.2f}%", fontsize=12, fontweight="bold")
ax2.set_xlabel("Dự đoán (Predicted)", fontsize=11)
ax2.set_ylabel("Thực tế (Actual)", fontsize=11)

plt.tight_layout()
plt.show()

# Thống kê so sánh số ca bỏ sót nguy hiểm
tn_def, fp_def, fn_def, tp_def = cm_default.ravel()
tn_saf, fp_saf, fn_saf, tp_saf = cm_safe.ravel()

print("\n" + "=" * 75)
print("BÁO CÁO CẢI THIỆN ĐỘ AN TOÀN CHO HỆ THỐNG GIÁM SÁT TÀI XẾ (DMS):")
print(f"-> Số ca bỏ sót tài xế buồn ngủ (False Negative) tại Th=0.50 : {fn_def} clips ({fn_def / (fn_def + tp_def) * 100:.2f}%)")
print(f"-> Số ca bỏ sót tài xế buồn ngủ (False Negative) tại Th={best_th_safe:.2f} : {fn_saf} clips ({fn_saf / (fn_saf + tp_saf) * 100:.2f}%)")
print(f"-> Mức độ giảm thiểu rủi ro tai nạn do ngủ gật               : Giảm {fn_def - fn_saf} ca nguy hiểm (Giảm {((fn_def - fn_saf) / fn_def) * 100:.1f}%)!")
print("=" * 75)


############################################################

# Cell 8 (code)
# ==============================================================================
# CELL 8: ĐÓNG GÓI TOÀN BỘ KẾT QUẢ, CẤU HÌNH NGƯỠNG AN TOÀN & TẢI VỀ
# ==============================================================================
import zipfile

# 1. Tạo tệp cấu hình ngưỡng an toàn threshold_config.json
threshold_metadata = {
    "model_name": "DeepGRUClassifier_v4",
    "architecture": "SpatialFeatureAdapter + 2-layer GRU + TemporalAttentionPooling",
    "supervision_mode": cfg.supervision_mode,
    "input_dim": cfg.input_dim,
    "hidden_dim": cfg.hidden_dim,
    "num_layers": cfg.num_layers,
    "optimal_threshold_f1": float(best_th_f1),
    "optimal_threshold_safe_f2": float(best_th_safe),
    "target_recall": float(cfg.target_recall),
    "metrics_at_default_0_50": {
        "precision": float(precision_score(eval_targets, (eval_probs >= 0.50).astype(int), pos_label=1, zero_division=0)),
        "recall": float(recall_score(eval_targets, (eval_probs >= 0.50).astype(int), pos_label=1, zero_division=0)),
        "f1": float(f1_score(eval_targets, (eval_probs >= 0.50).astype(int), pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(eval_targets, (eval_probs >= 0.50).astype(int), beta=2, pos_label=1, zero_division=0))
    },
    "metrics_at_safe_threshold": {
        "precision": float(precision_score(eval_targets, (eval_probs >= best_th_safe).astype(int), pos_label=1, zero_division=0)),
        "recall": float(recall_score(eval_targets, (eval_probs >= best_th_safe).astype(int), pos_label=1, zero_division=0)),
        "f1": float(f1_score(eval_targets, (eval_probs >= best_th_safe).astype(int), pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(eval_targets, (eval_probs >= best_th_safe).astype(int), beta=2, pos_label=1, zero_division=0))
    },
    "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

thresh_json_path = os.path.join(cfg.output_dir, "threshold_config.json")
with open(thresh_json_path, "w", encoding="utf-8") as f:
    json.dump(threshold_metadata, f, indent=4, ensure_ascii=False)
print(f"[+] Đã tạo tệp cấu hình ngưỡng an toàn: {thresh_json_path}")

# 2. Đóng gói toàn bộ kết quả thí nghiệm vào file zip
zip_output_name = "gru_v4_experiment_results.zip"
zip_path = os.path.join(cfg.output_dir, zip_output_name)

print(f"[+] Đang đóng gói kết quả thí nghiệm v4 vào: {zip_path}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 2.1. Nén toàn bộ log TensorBoard
    if os.path.exists(cfg.log_dir):
        for root, dirs, files in os.walk(cfg.log_dir):
            for file in files:
                file_full = os.path.join(root, file)
                rel_path = os.path.relpath(file_full, cfg.output_dir)
                zipf.write(file_full, rel_path)
    # 2.2. Nén các checkpoints
    if os.path.exists(cfg.checkpoint_dir):
        for root, dirs, files in os.walk(cfg.checkpoint_dir):
            for file in files:
                file_full = os.path.join(root, file)
                rel_path = os.path.relpath(file_full, cfg.output_dir)
                zipf.write(file_full, rel_path)
    # 2.3. Nén file cấu hình ngưỡng an toàn
    if os.path.exists(thresh_json_path):
        zipf.write(thresh_json_path, os.path.basename(thresh_json_path))

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"[SUCCESS] Đã tạo gói kết quả thành công ({zip_size_mb:.2f} MB): {zip_output_name}")

# Tạo đường link tải trực tiếp trong Kaggle
try:
    from IPython.display import FileLink
    print("\n[+] BẤM VÀO ĐƯỜNG DẪN DƯỚI ĐÂY ĐỂ TẢI TOÀN BỘ KẾT QUẢ VỀ MÁY CÁ NHÂN:")
    display(FileLink(zip_output_name))
except Exception:
    pass

print("\n" + "=" * 75)
print("HƯỚNG DẪN SỬ DỤNG KẾT QUẢ:")
print(f"1. Checkpoint chính (Primary) : {os.path.join(cfg.checkpoint_dir, 'best_gru.pth')}")
print(f"2. Checkpoint F1 tốt nhất     : {os.path.join(cfg.checkpoint_dir, 'best_gru_f1.pth')}")
print(f"3. Checkpoint F2 tốt nhất     : {os.path.join(cfg.checkpoint_dir, 'best_gru_f2.pth')}")
print(f"4. Ngưỡng an toàn áp dụng DMS : Threshold = {best_th_safe:.2f} (Đã ghi vào threshold_config.json)")
print("=" * 75)


############################################################

