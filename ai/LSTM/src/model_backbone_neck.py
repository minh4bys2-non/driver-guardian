import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo mã hóa UTF-8 cho console stdout/stderr trên Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Tự động cấu hình sys.path tương thích đa nền tảng ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # ai/LSTM

# Tìm thư mục gốc của repository (driver-guardian)
REPO_ROOT = None
for parent in SCRIPT_DIR.parents:
    if (parent / "ai" / "ObjectDetection_2p6M").exists() and (parent / "ai" / "NewVersionObDetect").exists():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = SCRIPT_DIR.parents[2]

for p in [
    PROJECT_ROOT,
    REPO_ROOT,
    REPO_ROOT / "ai" / "ObjectDetection_2p6M",
    REPO_ROOT / "ai" / "NewVersionObDetect",
]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import torch
import torch.nn as nn
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN


class BackboneNeck(nn.Module):
    """
    Mô hình tích hợp Không gian Spatio: Backbone và PAFPN Neck.
    Chuyên trách trích xuất 3 mức đặc trưng kim tự tháp (P3, P4, P5) từ ảnh đầu vào.

    Đầu vào:
        x (torch.Tensor): Tensor hình ảnh khuôn mặt [B, 3, H, W] (ví dụ: [B, 3, 640, 640]).
    Đầu ra:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - p3: [B, 64, H/8, W/8] (stride 8)
            - p4: [B, 128, H/16, W/16] (stride 16)
            - p5: [B, 256, H/32, W/32] (stride 32)
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
        backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
        neck_n: int = 1,
    ):
        """
        Khởi tạo BackboneNeck.

        Hỗ trợ 2 chế độ:
          1. Truyền trực tiếp module backbone và neck đã khởi tạo (tương thích ngược).
          2. Tự động khởi tạo Backbone và PAFPN từ cấu hình kiến trúc (backbone_w, backbone_n, neck_n).
        """
        super().__init__()
        self.backbone_w = tuple(backbone_w)
        self.backbone_n = tuple(backbone_n)
        self.neck_n = int(neck_n)
        c3, c4, c5 = self.backbone_w[2], self.backbone_w[3], self.backbone_w[4]
        self.neck_chs = (c3, c4, c5)

        if backbone is not None and neck is not None:
            self.backbone = backbone
            self.neck = neck
        else:
            self.backbone = Backbone(w=self.backbone_w, n=self.backbone_n)
            self.neck = PAFPN(chs=self.neck_chs, n=self.neck_n)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lan truyền xuôi qua Backbone và PAFPN Neck."""
        return self.neck(*self.backbone(x))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path, Dict[str, Any]],
        map_location: Optional[Union[str, torch.device]] = "cpu",
        eval_mode: bool = True,
        strict: bool = True,
        use_ema: bool = True,
        **kwargs,
    ) -> "BackboneNeck":
        """
        Khởi tạo mô hình BackboneNeck và nạp trọng số trực tiếp từ checkpoint.

        Hỗ trợ cả:
          1. Checkpoint tinh gọn chỉ chứa tham số BackboneNeck (ví dụ best1.pt).
          2. Checkpoint gốc đầy đủ của Landmark/Object Detector (ví dụ best.pt 65MB),
             tự động lọc lấy các trọng số thuộc Backbone và Neck, bỏ qua Head và Optimizer.

        Args:
            checkpoint_path (Union[str, Path, Dict[str, Any]]): Đường dẫn tệp .pt hoặc dict checkpoint.
            map_location (Optional[Union[str, torch.device]]): Thiết bị nạp (mặc định 'cpu').
            eval_mode (bool): Chuyển mô hình sang chế độ eval() sau khi nạp (mặc định True).
            strict (bool): Bắt buộc khớp chính xác toàn bộ tham số của Backbone và Neck (mặc định True).
            use_ema (bool): Ưu tiên nạp trọng số EMA nếu checkpoint có lưu (mặc định True).
            **kwargs: Cho phép ghi đè các tham số kiến trúc (backbone_w, backbone_n, neck_n).

        Returns:
            BackboneNeck: Thể hiện mô hình đã nạp trọng số hoàn chỉnh.
        """
        # 1. Nạp checkpoint từ file hoặc sử dụng dict trực tiếp
        if isinstance(checkpoint_path, (str, Path)):
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"[BackboneNeck] Không tìm thấy checkpoint tại: {path}")
            checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
        elif isinstance(checkpoint_path, dict):
            checkpoint = checkpoint_path
        else:
            raise TypeError(
                f"[BackboneNeck] checkpoint_path phải là str, Path hoặc dict, nhận được {type(checkpoint_path)}"
            )

        # 2. Trích xuất cấu hình kiến trúc (Ưu tiên: kwargs -> checkpoint config -> metadata -> mặc định)
        cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint.get("cfg"), dict) else {}
        meta_arch = (
            checkpoint.get("metadata", {}).get("architecture", {})
            if isinstance(checkpoint.get("metadata"), dict)
            else {}
        )

        backbone_w = kwargs.get(
            "backbone_w",
            checkpoint.get("backbone_w")
            or cfg.get("trunk_backbone_w")
            or meta_arch.get("trunk_backbone_w")
            or (16, 32, 64, 128, 256),
        )
        backbone_n = kwargs.get(
            "backbone_n",
            checkpoint.get("backbone_n")
            or cfg.get("trunk_backbone_n")
            or meta_arch.get("trunk_backbone_n")
            or (1, 2, 2, 1),
        )
        neck_n = kwargs.get(
            "neck_n",
            checkpoint.get("neck_n")
            or cfg.get("trunk_neck_n")
            or meta_arch.get("trunk_neck_n")
            or 1,
        )

        model = cls(
            backbone_w=backbone_w,
            backbone_n=backbone_n,
            neck_n=neck_n,
        )

        # 3. Lấy state_dict thô từ checkpoint theo thứ tự ưu tiên
        candidate_keys = (
            ["ema", "model", "model_state_dict", "state_dict"]
            if use_ema
            else ["model", "model_state_dict", "state_dict", "ema"]
        )
        raw_sd = None
        for key in candidate_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                raw_sd = checkpoint[key]
                break
        if raw_sd is None:
            if all(isinstance(k, str) for k in checkpoint.keys()):
                raw_sd = checkpoint
            else:
                raise KeyError("[BackboneNeck] Không tìm thấy state_dict hợp lệ trong checkpoint.")

        # 4. Lọc các tham số thuộc backbone và neck
        filtered_sd = {
            k: v for k, v in raw_sd.items() if k.startswith(("backbone.", "neck."))
        }

        # Nếu không có key nào có tiền tố (checkpoint thuần lưu trực tiếp), dùng toàn bộ raw_sd
        if not filtered_sd:
            filtered_sd = raw_sd

        model.load_state_dict(filtered_sd, strict=strict)

        if map_location is not None:
            model.to(map_location)

        if eval_mode:
            model.eval()

        return model


def extract_backbone_neck_checkpoint(
    source_ckpt_path: Union[str, Path],
    output_ckpt_path: Union[str, Path],
    use_ema: bool = True,
    strict_validation: bool = True,
) -> Path:
    """
    Trích xuất chỉ các tham số của BackboneNeck từ một checkpoint detector gốc và lưu thành file .pt tinh gọn.

    Args:
        source_ckpt_path: Đường dẫn checkpoint gốc (chứa full detector).
        output_ckpt_path: Đường dẫn tệp đích cần lưu.
        use_ema: Ưu tiên lấy trọng số EMA nếu có (mặc định True).
        strict_validation: Kiểm tra tính hợp lệ bằng cách nạp lại vào BackboneNeck (mặc định True).

    Returns:
        Path: Đường dẫn tệp checkpoint đã trích xuất thành công.
    """
    src = Path(source_ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"[extract_backbone_neck_checkpoint] Không tìm thấy file nguồn: {src}")

    dst = Path(output_ckpt_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang đọc checkpoint nguồn từ: {src} ...")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    # Chọn state_dict nguồn
    if use_ema and "ema" in ckpt and isinstance(ckpt["ema"], dict):
        source_weights = ckpt["ema"]
        weight_source = "ema"
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        source_weights = ckpt["model"]
        weight_source = "model"
    elif isinstance(ckpt, dict):
        source_weights = ckpt
        weight_source = "raw_dict"
    else:
        raise ValueError(f"Không thể đọc state_dict từ checkpoint: {src}")

    # Lọc chỉ lấy các keys thuộc backbone và neck
    extracted_sd = {
        k: v.clone() for k, v in source_weights.items() if k.startswith(("backbone.", "neck."))
    }

    if not extracted_sd:
        raise ValueError(f"Không tìm thấy tham số nào có tiền tố 'backbone.' hoặc 'neck.' trong {src}")

    num_backbone = sum(1 for k in extracted_sd if k.startswith("backbone."))
    num_neck = sum(1 for k in extracted_sd if k.startswith("neck."))
    total_params = sum(v.numel() for v in extracted_sd.values())
    total_bytes = sum(v.numel() * v.element_size() for v in extracted_sd.values())

    print(f"[✓] Đã trích xuất {len(extracted_sd)} tensors từ nguồn '{weight_source}':")
    print(f"    - Backbone tensors: {num_backbone}")
    print(f"    - Neck tensors: {num_neck}")
    print(f"    - Tổng tham số: {total_params:,} parameters ({total_bytes / (1024*1024):.2f} MB raw)")

    # Đóng gói checkpoint tinh gọn có metadata đầy đủ
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg"), dict) else {}
    meta_arch = (
        ckpt.get("metadata", {}).get("architecture", {})
        if isinstance(ckpt.get("metadata"), dict)
        else {}
    )

    backbone_w = (
        ckpt.get("backbone_w")
        or cfg.get("trunk_backbone_w")
        or meta_arch.get("trunk_backbone_w")
        or (16, 32, 64, 128, 256)
    )
    backbone_n = (
        ckpt.get("backbone_n")
        or cfg.get("trunk_backbone_n")
        or meta_arch.get("trunk_backbone_n")
        or (1, 2, 2, 1)
    )
    neck_n = (
        ckpt.get("neck_n")
        or cfg.get("trunk_neck_n")
        or meta_arch.get("trunk_neck_n")
        or 1
    )

    clean_checkpoint = {
        "model": extracted_sd,
        "backbone_w": tuple(backbone_w),
        "backbone_n": tuple(backbone_n),
        "neck_n": int(neck_n),
        "metadata": {
            "task": "backbone_neck_feature_extraction",
            "source_checkpoint": str(src),
            "weight_type": weight_source,
            "total_params": total_params,
            "num_tensors": len(extracted_sd),
            "neck_channels": [64, 128, 256],
            "output_features": ["p3", "p4", "p5"],
        },
    }

    torch.save(clean_checkpoint, str(dst))
    out_size_mb = dst.stat().st_size / (1024 * 1024)
    src_size_mb = src.stat().st_size / (1024 * 1024)
    print(f"[✓] ĐÃ LƯU CHECKPOINT TINH GỌN VÀO: {dst}")
    print(f"    - Kích thước ban đầu: {src_size_mb:.2f} MB")
    print(f"    - Kích thước trích xuất: {out_size_mb:.2f} MB (Tiết kiệm {(1 - out_size_mb / src_size_mb) * 100:.1f}%)")

    # Kiểm tra nạp nghiệm thu
    if strict_validation:
        print("[*] Đang xác thực nạp qua BackboneNeck.from_checkpoint() ...")
        loaded_model = BackboneNeck.from_checkpoint(dst, eval_mode=True, strict=True)
        dummy_x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            p3, p4, p5 = loaded_model(dummy_x)
        print(f"[✓] Xác thực thành công! Output shapes: P3={tuple(p3.shape)}, P4={tuple(p4.shape)}, P5={tuple(p5.shape)}")

    return dst


def main():
    weights_path = r"D:\Project\DATN\driver-guardian\ai\NewVersionObDetect\checkpoints\231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11\da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9\landmark_finetune\best.pt"
    cnn_filepath = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best1.pt"

    print("=" * 80)
    print(" CÔNG CỤ TRÍCH XUẤT CHECKPOINT BACKBONENECK - DRIVER GUARDIAN AI")
    print("=" * 80)

    extract_backbone_neck_checkpoint(
        source_ckpt_path=weights_path,
        output_ckpt_path=cnn_filepath,
        use_ema=True,
        strict_validation=True,
    )


if __name__ == "__main__":
    main()
