#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/img_preprocess.py
Mục đích:
    Module tiền xử lý và tăng cường ảnh/chuỗi khung hình video chụp tài xế
    trong điều kiện thiếu sáng, ban đêm (Low-Light & Night Vision Enhancement).

Kiến trúc:
    1. BaseImageTransform: Interface trừu tượng (Abstract Base Class) định nghĩa
       chuẩn hóa cho tất cả các phép biến đổi ảnh (apply, apply_sequence, enabled flag).
    2. Các phép biến đổi chuyên sâu:
       - AdaptiveGammaCorrection: Hiệu chỉnh Gamma phi tuyến tự động thích nghi.
       - CLAHETransform: Cân bằng Histogram cục bộ giới hạn tương phản trên kênh sáng LAB/YCrCb.
       - BilateralDenoiseTransform: Khử nhiễu cảm biến ISO cao bằng BilateralFilter bảo tồn cạnh viền mắt/miệng.
       - ColorBalanceTransform: Cân bằng trắng Gray-World khử ám vàng do đèn đường cao áp.
       - UnsharpMaskTransform: Làm sắc nét viền chi tiết con ngươi và mí mắt.
       - MultiScaleRetinexTransform: Tăng cường Retinex đa tỷ lệ cho vùng cực tối.
    3. LowLightImagePreprocessor: Lớp điều phối pipeline xử lý tuần tự, hỗ trợ
       thêm/xóa/bật/tắt động, cơ chế Auto Low-Light Gate và 3 Presets dựng sẵn.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger("LowLightPreprocessor")


# ==============================================================================
# HÀM BỔ TRỢ XỬ LÝ KHÔNG GIAN MÀU & ĐỘ SÁNG (COLOR & LUMINANCE UTILS)
# ==============================================================================
def _normalize_color_format(color_format: str) -> str:
    """Chuẩn hóa chuỗi định dạng màu, hỗ trợ bí danh GRB -> RGB."""
    fmt = color_format.strip().upper()
    if fmt == "GRB":
        return "RGB"
    if fmt not in ("RGB", "BGR"):
        raise ValueError(f"Định dạng màu '{color_format}' không được hỗ trợ. Chỉ hỗ trợ 'RGB' hoặc 'BGR'.")
    return fmt


def _compute_mean_luminance(image: np.ndarray, color_format: str = "RGB") -> float:
    """
    Tính độ sáng trung bình (mean luminance) của ảnh (thang đo 0.0 - 255.0).

    Args:
        image: Ảnh numpy uint8 2D [H, W] hoặc 3D [H, W, 3].
        color_format: Định dạng màu ("RGB" hoặc "BGR").

    Returns:
        float: Giá trị độ sáng trung bình.
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 2:
        return float(np.mean(image))

    fmt = _normalize_color_format(color_format)
    # Tính luminance theo trọng số chuẩn ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
    if fmt == "RGB":
        r = image[..., 0].astype(np.float32)
        g = image[..., 1].astype(np.float32)
        b = image[..., 2].astype(np.float32)
    else:  # BGR
        b = image[..., 0].astype(np.float32)
        g = image[..., 1].astype(np.float32)
        r = image[..., 2].astype(np.float32)

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return float(np.mean(lum))


# ==============================================================================
# 1. GIAO DIỆN TRỪU TƯỢNG (ABSTRACT BASE TRANSFORM INTERFACE)
# ==============================================================================
class BaseImageTransform(ABC):
    """
    Interface cơ sở trừu tượng cho tất cả các phép biến đổi tiền xử lý ảnh.

    Tất cả các phép biến đổi tùy biến kế thừa từ lớp này và hiện thực phương thức apply().
    Hỗ trợ bật/tắt động (`enabled`), áp dụng cho frame đơn hoặc chuỗi sequence.
    """

    def __init__(self, name: Optional[str] = None, enabled: bool = True) -> None:
        """
        Khởi tạo biến đổi cơ sở.

        Args:
            name: Tên định danh của biến đổi (mặc định lấy tên class).
            enabled: Cờ kích hoạt biến đổi (True: chạy, False: bypass).
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Áp dụng phép biến đổi lên một ảnh numpy.

        Args:
            image: Ảnh đầu vào [H, W] hoặc [H, W, 3], kiểu uint8.
            **kwargs: Các tham số ngữ cảnh mở rộng (ví dụ color_format).

        Returns:
            np.ndarray: Ảnh sau biến đổi (cùng kích thước và kiểu uint8).
        """
        pass

    def __call__(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Thực thi biến đổi khi gọi instance như callable object."""
        if not self.enabled:
            return image
        return self.apply(image, **kwargs)

    def apply_sequence(self, frames: Sequence[np.ndarray], **kwargs: Any) -> List[np.ndarray]:
        """
        Áp dụng biến đổi tuần tự lên một chuỗi khung hình video.

        Args:
            frames: Danh sách hoặc tuple chứa các khung hình numpy.
            **kwargs: Các tham số bổ sung chuyển tiếp đến apply().

        Returns:
            List[np.ndarray]: Danh sách các khung hình đã qua xử lý.
        """
        if not self.enabled:
            return list(frames)
        return [self.apply(frame, **kwargs) for frame in frames]

    def set_enabled(self, enabled: bool) -> "BaseImageTransform":
        """Bật hoặc tắt phép biến đổi (hỗ trợ method chaining)."""
        self.enabled = bool(enabled)
        return self

    def get_params(self) -> Dict[str, Any]:
        """Trả về từ điển chứa thông tin cấu hình của biến đổi."""
        return {"name": self.name, "enabled": self.enabled, "class": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"


# ==============================================================================
# 2. CÁC PHÉP BIẾN ĐỔI CHUYÊN SÂU CHO ẢNH THIẾU SÁNG & BAN ĐÊM
# ==============================================================================
class AdaptiveGammaCorrection(BaseImageTransform):
    """
    Hiệu chỉnh Gamma Thích ứng (Adaptive Gamma Correction).

    Tự động ước lượng hệ số gamma dựa trên độ sáng trung bình của ảnh,
    giúp kéo sáng các vùng bóng tối phi tuyến tính mà không làm cháy sáng vùng đủ sáng.
    """

    def __init__(
        self,
        gamma: Optional[float] = None,
        auto_gamma: bool = True,
        target_mean: float = 128.0,
        gamma_range: Tuple[float, float] = (0.25, 2.0),
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            gamma: Giá trị gamma cố định nếu auto_gamma=False. (gamma < 1.0 làm sáng ảnh).
            auto_gamma: Nếu True, tự động tính gamma dựa trên độ sáng trung bình.
            target_mean: Giá trị độ sáng mục tiêu (0 - 255, mặc định 128.0).
            gamma_range: Giới hạn (min_gamma, max_gamma) để ngăn chặn biến dạng quá mức.
        """
        super().__init__(name=name, enabled=enabled)
        self.gamma = gamma
        self.auto_gamma = auto_gamma
        self.target_mean = float(np.clip(target_mean, 10.0, 245.0))
        self.gamma_range = gamma_range

    def _calculate_auto_gamma(self, image: np.ndarray, color_format: str) -> float:
        """Tính gamma tự thích nghi: gamma = ln(target/255) / ln(mean/255)."""
        mean_lum = _compute_mean_luminance(image, color_format=color_format)
        if mean_lum <= 1.0:
            return self.gamma_range[0]
        if mean_lum >= 254.0:
            return self.gamma_range[1]

        norm_mean = mean_lum / 255.0
        norm_target = self.target_mean / 255.0
        calculated_gamma = float(np.log(norm_target) / np.log(norm_mean))
        return float(np.clip(calculated_gamma, self.gamma_range[0], self.gamma_range[1]))

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0:
            return image

        if self.auto_gamma or self.gamma is None:
            effective_gamma = self._calculate_auto_gamma(image, color_format=color_format)
        else:
            effective_gamma = float(np.clip(self.gamma, self.gamma_range[0], self.gamma_range[1]))

        # Xây dựng bảng tra cứu LUT (Lookup Table) tăng tốc độ thực thi O(1)
        inv_gamma = effective_gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, lut)

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "gamma": self.gamma,
            "auto_gamma": self.auto_gamma,
            "target_mean": self.target_mean,
            "gamma_range": self.gamma_range,
        })
        return params


class CLAHETransform(BaseImageTransform):
    """
    Cân bằng Histogram Cục bộ Giới hạn Độ tương phản (CLAHE).

    Chuyển đổi sang không gian màu LAB (hoặc YCrCb), áp dụng thuật toán CLAHE
    chỉ trên kênh độ sáng (L) để tăng độ tương phản chi tiết mắt và khuôn mặt
    mà không gây biến dạng sắc tố màu sắc.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        color_space: str = "LAB",
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            clip_limit: Ngưỡng cắt giới hạn độ tương phản nhằm triệt tiêu nhiễu.
            tile_grid_size: Kích thước lưới ô lưới cục bộ (mặc định 8x8).
            color_space: Không gian màu xử lý ("LAB" hoặc "YCrCb").
        """
        super().__init__(name=name, enabled=enabled)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.color_space = color_space.upper()
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0:
            return image

        fmt = _normalize_color_format(color_format)

        # Trường hợp ảnh Grayscale 2D
        if image.ndim == 2:
            return self._clahe.apply(image)

        # Trường hợp ảnh màu 3D
        if self.color_space == "YCRCB":
            code_forward = cv2.COLOR_RGB2YCrCb if fmt == "RGB" else cv2.COLOR_BGR2YCrCb
            code_backward = cv2.COLOR_YCrCb2RGB if fmt == "RGB" else cv2.COLOR_YCrCb2BGR
            ycrcb = cv2.cvtColor(image, code_forward)
            ycrcb[..., 0] = self._clahe.apply(ycrcb[..., 0])
            return cv2.cvtColor(ycrcb, code_backward)

        # Mặc định sử dụng không gian màu LAB
        code_forward = cv2.COLOR_RGB2LAB if fmt == "RGB" else cv2.COLOR_BGR2LAB
        code_backward = cv2.COLOR_LAB2RGB if fmt == "RGB" else cv2.COLOR_LAB2BGR
        lab = cv2.cvtColor(image, code_forward)
        lab[..., 0] = self._clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, code_backward)

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size,
            "color_space": self.color_space,
        })
        return params


class BilateralDenoiseTransform(BaseImageTransform):
    """
    Bộ Lọc Khử Nhiễu Bảo Toàn Cạnh (Edge-Preserving Bilateral Filter).

    Sử dụng cv2.bilateralFilter để làm mịn nhiễu hạt cảm biến sinh ra do khuếch đại
    ISO trong bóng tối, đồng thời bảo tồn tuyệt đối các đường viền sắc nét quan trọng
    như mí mắt, con ngươi và khuôn miệng.
    """

    def __init__(
        self,
        d: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            d: Đường kính vùng lân cận điểm ảnh (pixel neighborhood diameter).
            sigma_color: Độ lệch chuẩn bộ lọc trong không gian màu sắc tố.
            sigma_space: Độ lệch chuẩn bộ lọc trong không gian tọa độ.
        """
        super().__init__(name=name, enabled=enabled)
        self.d = int(d)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0:
            return image

        # cv2.bilateralFilter hoạt động trực tiếp trên mảng 2D hoặc 3D uint8
        return cv2.bilateralFilter(image, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "d": self.d,
            "sigma_color": self.sigma_color,
            "sigma_space": self.sigma_space,
        })
        return params


class ColorBalanceTransform(BaseImageTransform):
    """
    Cân Bằng Trắng & Khử Ám Màu (Gray-World & Percentile Color Balance).

    Triệt tiêu hiện tượng ám vàng cam do đèn đường cao áp hoặc đèn xe cabin,
    đưa màu da mặt và nền về trạng thái phân bố tự nhiên.
    """

    def __init__(
        self,
        percentile_clip: float = 1.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            percentile_clip: Tỉ lệ phần trăm cắt ở 2 đầu histogram (mặc định 1.0%).
        """
        super().__init__(name=name, enabled=enabled)
        self.percentile_clip = float(np.clip(percentile_clip, 0.0, 10.0))

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0 or image.ndim != 3:
            return image

        out = np.empty_like(image)
        low_p = self.percentile_clip
        high_p = 100.0 - self.percentile_clip

        for c in range(image.shape[2]):
            channel = image[..., c].astype(np.float32)
            v_min, v_max = np.percentile(channel, (low_p, high_p))
            if v_max > v_min:
                stretched = (channel - v_min) * (255.0 / (v_max - v_min))
                out[..., c] = np.clip(stretched, 0, 255).astype(np.uint8)
            else:
                out[..., c] = image[..., c]

        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({"percentile_clip": self.percentile_clip})
        return params


class UnsharpMaskTransform(BaseImageTransform):
    """
    Tăng Cường Độ Sắc Nét Viền (Unsharp Masking Detail Sharpen).

    Tạo mặt nạ tương phản viền bằng Gaussian Blur và cộng dồn ngược vào ảnh gốc,
    làm nổi bật khóe mắt, con ngươi và khóe miệng tài xế.
    """

    def __init__(
        self,
        strength: float = 0.5,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 1.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            strength: Hệ số khuếch đại độ sắc nét (0.1 - 2.0).
            kernel_size: Kích thước kernel Gaussian (số lẻ, ví dụ (5, 5)).
            sigma: Độ lệch chuẩn Gaussian Blur.
        """
        super().__init__(name=name, enabled=enabled)
        self.strength = float(np.clip(strength, 0.0, 3.0))
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0 or self.strength <= 1e-4:
            return image

        blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        sharpened = cv2.addWeighted(image, 1.0 + self.strength, blurred, -self.strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "strength": self.strength,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
        })
        return params


class MultiScaleRetinexTransform(BaseImageTransform):
    """
    Tăng Cường Phản Xạ Retinex Đa Tỉ Lệ (Multi-Scale Retinex - MSR).

    Phân rã ảnh thành thành phần phản xạ (Reflectance) và trường chiếu sáng (Illumination),
    áp dụng tích chập Gaussian đa tỉ lệ để khôi phục chi tiết tại những vùng tối mịt.
    """

    def __init__(
        self,
        sigma_list: Tuple[float, ...] = (15.0, 80.0, 250.0),
        weights: Optional[Tuple[float, ...]] = None,
        dynamic_clip_factor: float = 2.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            sigma_list: Danh sách độ lệch chuẩn Gaussian đại diện cho các tỉ lệ không gian.
            weights: Trọng số của từng tỉ lệ (mặc định chia đều).
            dynamic_clip_factor: Hệ số cắt theo độ lệch chuẩn (mean ± factor*std).
        """
        super().__init__(name=name, enabled=enabled)
        self.sigma_list = sigma_list
        if weights is None:
            self.weights = tuple([1.0 / len(sigma_list)] * len(sigma_list))
        else:
            total_w = sum(weights)
            self.weights = tuple([w / total_w for w in weights])
        self.dynamic_clip_factor = float(dynamic_clip_factor)

    def apply(self, image: np.ndarray, color_format: str = "RGB", **kwargs: Any) -> np.ndarray:
        if not self.enabled or image is None or image.size == 0:
            return image

        img_float = image.astype(np.float32) + 1.0  # Tránh log(0)
        log_img = np.log(img_float)

        msr = np.zeros_like(img_float)
        for sigma, weight in zip(self.sigma_list, self.weights):
            blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
            msr += weight * (log_img - np.log(blurred + 1.0))

        # Chuẩn hóa dải động bằng Dynamic Range Normalization (mean ± factor * std)
        out = np.empty_like(image)
        channels = 1 if image.ndim == 2 else image.shape[2]

        for c in range(channels):
            msr_c = msr[..., c] if channels > 1 else msr
            mean = float(np.mean(msr_c))
            std = float(np.std(msr_c))
            min_val = mean - self.dynamic_clip_factor * std
            max_val = mean + self.dynamic_clip_factor * std
            if max_val > min_val:
                norm_c = (msr_c - min_val) * (255.0 / (max_val - min_val))
                norm_c = np.clip(norm_c, 0, 255).astype(np.uint8)
            else:
                norm_c = np.clip(msr_c, 0, 255).astype(np.uint8)

            if channels > 1:
                out[..., c] = norm_c
            else:
                out = norm_c

        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "sigma_list": self.sigma_list,
            "weights": self.weights,
            "dynamic_clip_factor": self.dynamic_clip_factor,
        })
        return params


# ==============================================================================
# 3. LỚP ĐIỀU PHỐI PIPELINE TIỀN XỬ LÝ (LOW-LIGHT IMAGE PREPROCESSOR)
# ==============================================================================
class LowLightImagePreprocessor:
    """
    Bộ điều phối Pipeline Tiền xử lý ảnh thiếu sáng và ban đêm.

    Đặc tính cốt lõi:
        1. Quản lý danh sách các phép biến đổi linh hoạt (thêm, xóa, bật, tắt động).
        2. Hỗ trợ Fluent API / Method Chaining.
        3. Cơ chế Auto Low-Light Gate có thể bật/tắt: tự động đo độ sáng trung bình
           và bypass các khung hình đã đủ sáng để bảo vệ ảnh ban ngày không bị cháy sáng.
        4. Hỗ trợ xử lý cả ảnh đơn lẻ (process) và chuỗi video clip (process_sequence).
        5. Không gian màu mặc định là RGB (chuẩn PyTorch), tự động chuyển đổi nếu đầu vào là BGR hoặc GRB.
    """

    def __init__(
        self,
        transforms: Optional[List[BaseImageTransform]] = None,
        default_color_format: str = "RGB",
        enable_auto_gate: bool = True,
        gate_threshold: float = 65.0,
    ) -> None:
        """
        Args:
            transforms: Danh sách các phép biến đổi khởi tạo ban đầu.
            default_color_format: Định dạng màu mặc định ("RGB" hoặc "BGR"). Mặc định: "RGB".
            enable_auto_gate: Cờ bật/tắt cơ chế tự phát hiện ảnh thiếu sáng.
            gate_threshold: Ngưỡng độ sáng trung bình (L_mean). Nếu L_mean >= threshold và gate bật,
                            ảnh được coi là đủ sáng và sẽ bypass pipeline.
        """
        self._transforms: List[BaseImageTransform] = []
        if transforms:
            for t in transforms:
                self.add_transform(t)

        self.default_color_format = _normalize_color_format(default_color_format)
        self.enable_auto_gate = bool(enable_auto_gate)
        self.gate_threshold = float(gate_threshold)

    # --------------------------------------------------------------------------
    # QUẢN LÝ TRANSFORMS (FLUENT INTERFACE)
    # --------------------------------------------------------------------------
    def add_transform(
        self, transform: BaseImageTransform, index: Optional[int] = None
    ) -> "LowLightImagePreprocessor":
        """
        Thêm một phép biến đổi vào pipeline.

        Args:
            transform: Thực thể kế thừa từ BaseImageTransform.
            index: Vị trí chèn trong pipeline (nếu None, chèn vào cuối cùng).

        Returns:
            LowLightImagePreprocessor: Chính instance hiện tại (hỗ trợ method chaining).
        """
        if not isinstance(transform, BaseImageTransform):
            raise TypeError(
                f"Biến đổi phải kế thừa từ BaseImageTransform, nhận được: {type(transform)}"
            )

        if index is None or index >= len(self._transforms):
            self._transforms.append(transform)
        else:
            self._transforms.insert(max(0, index), transform)
        return self

    def remove_transform(self, name: str) -> bool:
        """
        Xóa một phép biến đổi khỏi pipeline theo tên.

        Args:
            name: Tên biến đổi cần xóa.

        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy.
        """
        for i, t in enumerate(self._transforms):
            if t.name.lower() == name.lower():
                self._transforms.pop(i)
                return True
        return False

    def get_transform(self, name: str) -> Optional[BaseImageTransform]:
        """Tìm kiếm một phép biến đổi theo tên."""
        for t in self._transforms:
            if t.name.lower() == name.lower():
                return t
        return None

    def set_transform_enabled(self, name: str, enabled: bool) -> bool:
        """
        Bật hoặc tắt một phép biến đổi theo tên mà không cần xóa khỏi danh sách.

        Args:
            name: Tên biến đổi.
            enabled: Trạng thái True hoặc False.

        Returns:
            bool: True nếu cập nhật thành công, False nếu không tìm thấy.
        """
        t = self.get_transform(name)
        if t is not None:
            t.set_enabled(enabled)
            return True
        return False

    def clear_transforms(self) -> "LowLightImagePreprocessor":
        """Xóa toàn bộ danh sách các biến đổi."""
        self._transforms.clear()
        return self

    def list_transforms(self) -> List[Dict[str, Any]]:
        """Trả về thông tin chi tiết danh sách các biến đổi hiện có."""
        return [t.get_params() for t in self._transforms]

    # --------------------------------------------------------------------------
    # CƠ CHẾ AUTO LOW-LIGHT GATE
    # --------------------------------------------------------------------------
    def enable_gate(self, enabled: bool) -> "LowLightImagePreprocessor":
        """Bật hoặc tắt cơ chế Auto Low-Light Gate."""
        self.enable_auto_gate = bool(enabled)
        return self

    def set_gate_threshold(self, threshold: float) -> "LowLightImagePreprocessor":
        """Cài đặt ngưỡng độ sáng kích hoạt (0.0 - 255.0)."""
        self.gate_threshold = float(np.clip(threshold, 0.0, 255.0))
        return self

    def is_low_light(
        self, image: np.ndarray, color_format: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Kiểm tra xem ảnh có thuộc diện thiếu sáng hay không.

        Args:
            image: Ảnh numpy.
            color_format: Định dạng màu (mặc định lấy theo self.default_color_format).

        Returns:
            Tuple[bool, float]: (is_low_light, mean_luminance).
        """
        fmt = _normalize_color_format(color_format or self.default_color_format)
        mean_lum = _compute_mean_luminance(image, color_format=fmt)
        return (mean_lum < self.gate_threshold, mean_lum)

    # --------------------------------------------------------------------------
    # PHƯƠNG THỨC XỬ LÝ (PROCESS METHODS)
    # --------------------------------------------------------------------------
    def process(self, image: np.ndarray, color_format: Optional[str] = None) -> np.ndarray:
        """
        Xử lý tăng cường một khung hình ảnh đơn lẻ.

        Args:
            image: Ảnh numpy uint8 [H, W] hoặc [H, W, 3].
            color_format: Định dạng màu ("RGB" hoặc "BGR"). Mặc định dùng cấu hình của preprocessor.

        Returns:
            np.ndarray: Ảnh đã được tăng cường (hoặc ảnh gốc nếu gate xác định ảnh đủ sáng).
        """
        if image is None or image.size == 0:
            return image

        fmt = _normalize_color_format(color_format or self.default_color_format)

        # Kiểm tra Auto Low-Light Gate nếu được kích hoạt
        if self.enable_auto_gate:
            is_low, mean_lum = self.is_low_light(image, color_format=fmt)
            if not is_low:
                # Ảnh ban ngày hoặc đủ sáng -> bypass toàn bộ pipeline
                return image

        # Thực thi tuần tự chuỗi các biến đổi đang được kích hoạt (enabled=True)
        current = image
        for transform in self._transforms:
            if transform.enabled:
                current = transform.apply(current, color_format=fmt)

        return current

    def process_sequence(
        self, images: Sequence[np.ndarray], color_format: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Xử lý tăng cường một chuỗi các khung hình video.

        Áp dụng kiểm tra gate trên khung hình đầu tiên (hoặc tính trung bình chuỗi)
        để đảm bảo tính nhất quán thời gian (temporal consistency) cho toàn clip.

        Args:
            images: Chuỗi danh sách các khung hình numpy [T, H, W, 3].
            color_format: Định dạng màu ("RGB" hoặc "BGR").

        Returns:
            List[np.ndarray]: Danh sách các khung hình đã qua tiền xử lý.
        """
        if not images:
            return []

        fmt = _normalize_color_format(color_format or self.default_color_format)

        # Kiểm tra Gate dựa trên khung hình đầu tiên để tiết kiệm tính toán
        if self.enable_auto_gate:
            is_low, _ = self.is_low_light(images[0], color_format=fmt)
            if not is_low:
                return list(images)

        # Xử lý chuỗi
        output_frames: List[np.ndarray] = []
        for frame in images:
            current = frame
            for transform in self._transforms:
                if transform.enabled:
                    current = transform.apply(current, color_format=fmt)
            output_frames.append(current)

        return output_frames

    def __call__(
        self,
        input_data: Union[np.ndarray, Sequence[np.ndarray]],
        color_format: Optional[str] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Cho phép gọi instance trực tiếp như một hàm callable."""
        if isinstance(input_data, (list, tuple)):
            return self.process_sequence(input_data, color_format=color_format)
        if isinstance(input_data, np.ndarray) and input_data.ndim == 4:
            # Dạng mảng 4D [T, H, W, C]
            return np.stack(self.process_sequence(list(input_data), color_format=color_format), axis=0)
        return self.process(input_data, color_format=color_format)

    def __repr__(self) -> str:
        names = [f"{t.name}({'on' if t.enabled else 'off'})" for t in self._transforms]
        return (
            f"LowLightImagePreprocessor(transforms=[{', '.join(names)}], "
            f"color_format='{self.default_color_format}', "
            f"auto_gate={self.enable_auto_gate}, gate_threshold={self.gate_threshold})"
        )

    # --------------------------------------------------------------------------
    # FACTORY PRESETS DỰNG SẴN
    # --------------------------------------------------------------------------
    @classmethod
    def create_default_night_pipeline(
        cls,
        enable_auto_gate: bool = True,
        gate_threshold: float = 65.0,
        default_color_format: str = "RGB",
    ) -> "LowLightImagePreprocessor":
        """
        Pipeline Mặc Định Đề Xuất (Balanced Night Pipeline):
        Cân bằng tối ưu giữa tăng sáng, tái tạo tương phản, khử ám vàng đèn đường,
        làm sắc nét viền mắt/miệng và dập tắt nhiễu hạt bằng Bilateral Filter.
        """
        pipeline = cls(
            default_color_format=default_color_format,
            enable_auto_gate=enable_auto_gate,
            gate_threshold=gate_threshold,
        )
        pipeline.add_transform(ColorBalanceTransform(percentile_clip=1.0))
        pipeline.add_transform(AdaptiveGammaCorrection(auto_gamma=True, target_mean=125.0))
        pipeline.add_transform(CLAHETransform(clip_limit=2.5, tile_grid_size=(8, 8)))
        pipeline.add_transform(BilateralDenoiseTransform(d=5, sigma_color=50.0, sigma_space=50.0))
        pipeline.add_transform(UnsharpMaskTransform(strength=0.4, kernel_size=(5, 5), sigma=1.0))
        return pipeline

    @classmethod
    def create_fast_pipeline(
        cls,
        enable_auto_gate: bool = True,
        gate_threshold: float = 65.0,
        default_color_format: str = "RGB",
    ) -> "LowLightImagePreprocessor":
        """
        Pipeline Siêu Tốc (Fast Pipeline):
        Tối ưu hóa tối đa cho các hệ thống nhúng / Edge AI / FPS cao (>100 FPS),
        chỉ gồm Adaptive Gamma và CLAHE.
        """
        pipeline = cls(
            default_color_format=default_color_format,
            enable_auto_gate=enable_auto_gate,
            gate_threshold=gate_threshold,
        )
        pipeline.add_transform(AdaptiveGammaCorrection(auto_gamma=True, target_mean=120.0))
        pipeline.add_transform(CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)))
        return pipeline

    @classmethod
    def create_retinex_pipeline(
        cls,
        enable_auto_gate: bool = True,
        gate_threshold: float = 65.0,
        default_color_format: str = "RGB",
    ) -> "LowLightImagePreprocessor":
        """
        Pipeline Retinex Chuyên Sâu (Deep Night / Retinex Pipeline):
        Dành cho các tình huống khoang cabin cực tối, ánh sáng gần như bằng không.
        Sử dụng phân rã Retinex đa tỷ lệ kết hợp cân bằng trắng và Bilateral Denoise.
        """
        pipeline = cls(
            default_color_format=default_color_format,
            enable_auto_gate=enable_auto_gate,
            gate_threshold=gate_threshold,
        )
        pipeline.add_transform(ColorBalanceTransform(percentile_clip=1.5))
        pipeline.add_transform(MultiScaleRetinexTransform(sigma_list=(15.0, 80.0, 250.0)))
        pipeline.add_transform(BilateralDenoiseTransform(d=5, sigma_color=40.0, sigma_space=40.0))
        pipeline.add_transform(UnsharpMaskTransform(strength=0.5, kernel_size=(5, 5), sigma=1.0))
        return pipeline
