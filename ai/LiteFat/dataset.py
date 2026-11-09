# =====================================================================
# File: dataset.py
# Dự án: Driver Guardian - Module LiteFat
# Tác vụ: Nạp tập dữ liệu phát hiện buồn ngủ, trích xuất 68 điểm mốc khuôn mặt
#         bằng PIPNet (FaceBoxesV2 + ResNet18 300W) và cắt vùng khuôn mặt
#         chuẩn hóa cho mô hình LiteFat (MobileNetV3-Small + STGL).
# Tối ưu hóa:
#   - Tốc độ: Batched GPU FaceBoxesV2, Cached Priors, Torchvision GPU NMS,
#             Batched GPU PIPNet với giải mã NRM vector hóa, Fast Video Seeking.
#   - Chính xác: Đồng bộ kích thước letterbox (ultis.py), theo dõi khuôn mặt tài xế
#               qua IoU, Landmark-Guided BBox Refinement, nội suy tuyến tính &
#               làm mượt thời gian (Temporal Smoothing) cho khung hình khuôn mặt.
# =====================================================================
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Monkey patch cho scipy >= 1.14 để tương thích với torchlm metrics
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

# Cấu hình môi trường Windows và nạp DLL PyTorch CUDA
torch_lib_path = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.exists(torch_lib_path):
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

# Import hàm chuẩn hóa và tăng cường từ ultis.py
from ultis import letterbox, _parse_size, compute_iou, crop_faces_from_boxes, augment_multimodal_clip

# Import khởi tạo PIPNet & hàm vẽ điểm mốc từ pipnet_68_camera.py
from pipnet_68_camera import init_pipnet_68_gpu, draw_68_landmarks_custom

logger = logging.getLogger("DriverGuardian.LiteFat")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Các hàm _parse_size, compute_iou, crop_faces_from_boxes đã được tập trung trong ultis.py.


# =====================================================================
# 1. BỘ TRÍCH XUẤT PIPNET TỐI ƯU HÓA CAO CẤP (FAST PIPNET EXTRACTOR)
# =====================================================================
class FastPIPNetExtractor:
    """
    Bộ trích xuất 68 điểm mốc khuôn mặt và cắt khung hình cho LiteFat
    được tối ưu hóa tối đa:
      1. Batched FaceBoxesV2 trên GPU: Nạp một mảng T khung hình cùng lúc,
         tái sử dụng PriorBox anchors đã được tính toán sẵn 1 lần trên GPU,
         sử dụng torchvision.ops.nms trực tiếp trên GPU.
      2. Temporal Driver Tracking: Lựa chọn khuôn mặt tài xế ổn định dựa trên
         điểm số, diện tích và độ tương đồng IoU qua các khung hình liên tiếp.
      3. Batched PIPNet 300W trên GPU: Gom toàn bộ vùng mặt phát hiện được thành
         1 batch tensor [N_valid, 3, 256, 256], chạy 1 lần suy luận duy nhất,
         giải mã NRM (Neighbor Regression) hoàn toàn bằng tensor PyTorch trên GPU.
      4. Landmark-Guided Box Refinement: Mở rộng hộp bao kết hợp giữa BBox và
         cực trị 68 điểm mốc, làm mượt theo thời gian (EMA) và nội suy tuyến tính
         cho các khung hình mất mặt, đảm bảo không bị cắt lẹm cằm/mắt.
    """

    def __init__(
            self,
            backbone: str = "resnet18",
            device: Optional[str] = None,
            letterbox_size: Tuple[int, int] = (640, 640),
            face_size: Tuple[int, int] = (224, 224),
            conf_threshold: float = 0.5,
            nms_threshold: float = 0.3,
            bbox_margin: float = 0.15,
            box_smooth_alpha: float = 0.7,
            use_fp16: bool = False,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.letterbox_size = _parse_size(letterbox_size)  # (target_w, target_h)
        self.face_size = _parse_size(face_size)  # (crop_w, crop_h)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.bbox_margin = float(bbox_margin)
        self.box_smooth_alpha = float(box_smooth_alpha)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")

        # Khởi tạo mô hình thông qua runtime của torchlm
        from torchlm.runtime._wrappers import RuntimeWrapper
        if getattr(RuntimeWrapper, "face_base", None) is None or getattr(RuntimeWrapper, "landmarks_base",
                                                                         None) is None:
            init_pipnet_68_gpu(backbone=backbone)

        self.face_base = RuntimeWrapper.face_base
        self.face_net = self.face_base.net.to(self.device)
        self.face_net.eval()

        self.lmk_base = RuntimeWrapper.landmarks_base
        self.lmk_net = self.lmk_base.to(self.device)
        self.lmk_net.eval()

        # Cache PriorBox và hằng số chuẩn hóa trên GPU cho kích thước letterbox cố định
        target_w, target_h = self.letterbox_size
        from torchlm.tools._faceboxesv2 import PriorBox
        pbox = PriorBox(self.face_base.cfg, image_size=(target_h, target_w))
        self.cached_priors = pbox.forward().to(self.device)  # [P, 4]
        self.cached_variances = self.face_base.cfg["variance"]
        self.cached_scale = torch.tensor(
            [target_w, target_h, target_w, target_h], dtype=torch.float32, device=self.device
        )
        self.faceboxes_mean = torch.tensor(
            [104.0, 117.0, 123.0], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        # Hằng số chuẩn hóa PIPNet (ImageNet RGB)
        self.pipnet_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.pipnet_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        # Cache bảng chỉ số NRM cho PIPNet trên GPU
        rev1 = self.lmk_base.reverse_index1
        rev2 = self.lmk_base.reverse_index2
        num_nb = self.lmk_base.num_nb
        self.rev_indices = torch.tensor(
            [r1 * num_nb + r2 for r1, r2 in zip(rev1, rev2)],
            dtype=torch.long,
            device=self.device,
        )

        # Hằng số chuẩn hóa ImageNet cho MobileNetV3 (CPU / GPU)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    @torch.no_grad()
    def _detect_faces_batched(self, frames: List[np.ndarray], max_chunk: int = 32) -> List[np.ndarray]:
        """
        Chạy phát hiện khuôn mặt theo lô T khung hình trên GPU.
        Tự động chia nhỏ theo max_chunk để tránh tràn VRAM GPU khi chuỗi khung hình dài (ví dụ video 300-1000 frames).
        Mỗi khung hình trong `frames` đã được chuẩn hóa kích thước letterbox (target_h, target_w).
        Trả về danh sách mảng numpy shape [num_dets, 5] (x1, y1, x2, y2, score).
        """
        T = len(frames)
        if T == 0:
            return []

        all_dets: List[np.ndarray] = []
        priors_xy = self.cached_priors[None, :, :2]
        priors_wh = self.cached_priors[None, :, 2:]
        v0, v1 = self.cached_variances[0], self.cached_variances[1]

        for s in range(0, T, max_chunk):
            chunk = frames[s:s + max_chunk]
            cur_T = len(chunk)

            # Gom batch BGR numpy -> tensor [cur_T, 3, H, W] trên GPU
            frames_np = np.stack(chunk, axis=0)  # [cur_T, H, W, 3]
            tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(self.device).float()
            tensor = tensor - self.faceboxes_mean

            loc, conf = self.face_net(tensor)  # loc: [cur_T, P, 4], conf: [cur_T, P, 2]
            scores = conf[:, :, 1]  # [cur_T, P]

            # Giải mã tọa độ hộp giới hạn vector hóa cho cả chunk
            boxes_xy = priors_xy + loc[:, :, :2] * v0 * priors_wh
            boxes_wh = priors_wh * torch.exp(loc[:, :, 2:] * v1)
            b_x1y1 = boxes_xy - boxes_wh / 2.0
            b_x2y2 = b_x1y1 + boxes_wh
            decoded_boxes = torch.cat([b_x1y1, b_x2y2], dim=-1) * self.cached_scale  # [cur_T, P, 4]

            for b in range(cur_T):
                sc = scores[b]
                mask = sc >= self.conf_threshold
                if not mask.any():
                    all_dets.append(np.empty((0, 5), dtype=np.float32))
                    continue

                bx = decoded_boxes[b, mask]
                sc_valid = sc[mask]

                keep = torchvision.ops.nms(bx, sc_valid, iou_threshold=self.nms_threshold)
                keep = keep[:50]  # Giữ tối đa 50 mặt có độ tin cậy cao nhất
                res = torch.cat([bx[keep], sc_valid[keep, None]], dim=-1).cpu().numpy()
                all_dets.append(res)

        return all_dets

    def _select_driver_faces(
            self,
            all_dets: List[np.ndarray],
            target_w: int,
            target_h: int
    ) -> Tuple[List[Optional[np.ndarray]], List[bool]]:
        """
        Theo dõi và chọn duy nhất 1 khuôn mặt tài xế cho từng khung hình:
          - Kết hợp diện tích và độ tin cậy.
          - Duy trì tính nhất quán vị trí không gian qua IoU với khung hình liền kề.
        Trả về:
          - chosen_bboxes: danh sách bbox [x1, y1, x2, y2, score] (hoặc None nếu mất mặt)
          - found: danh sách boolean đánh dấu phát hiện được khuôn mặt
        """
        T = len(all_dets)
        chosen_bboxes: List[Optional[np.ndarray]] = [None] * T
        found: List[bool] = [False] * T

        prev_box: Optional[np.ndarray] = None
        prev_box_area = 1.0

        for t in range(T):
            dets = all_dets[t]
            if len(dets) == 0:
                continue

            best_idx = 0
            if prev_box is None:
                # Frame đầu tiên có mặt: chọn theo score * area
                best_score = -1.0
                for i, d in enumerate(dets):
                    area = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
                    metric = d[4] * (area ** 0.5)
                    if metric > best_score:
                        best_score = metric
                        best_idx = i
            else:
                # Các frame tiếp theo: ưu tiên IoU với khuôn mặt đã khóa trước đó
                best_match = -1.0
                for i, d in enumerate(dets):
                    iou = compute_iou(d[:4], prev_box[:4])
                    area = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
                    # Kết hợp IoU và độ tin cậy
                    metric = (iou * 2.0) + (d[4] * 0.5) + (min(1.0, area / (prev_box_area + 1e-6)) * 0.3)
                    if metric > best_match:
                        best_match = metric
                        best_idx = i

            chosen = dets[best_idx].copy()
            chosen_bboxes[t] = chosen
            found[t] = True
            prev_box = chosen
            prev_box_area = max(1.0, (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1]))

        return chosen_bboxes, found

    @torch.no_grad()
    def _extract_landmarks_batched(
            self,
            frames: List[np.ndarray],
            chosen_bboxes: List[Optional[np.ndarray]],
            found: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất 68 điểm mốc khuôn mặt theo chuẩn 300W bằng PIPNet theo lô trên GPU.
        Đảm bảo giữ chính xác công thức căn chỉnh vùng cắt của PIPNet 300W:
          extend = 0.2 -> x1 -= 0.1*w, y1 += 0.1*h, x2 += 0.1*w, y2 += 0.1*h
        Trả về:
          - landmarks_norm: [T, 68, 2] gồm (x/W, y/H)
          - landmarks_px:   [T, 68, 2] tọa độ pixel tuyệt đối trên khung hình letterbox
        """
        T = len(frames)
        target_w, target_h = self.letterbox_size
        extend = 0.2

        valid_indices = [t for t in range(T) if found[t] and chosen_bboxes[t] is not None]

        # Khởi tạo ma trận điểm mốc: mặc định gán 1.0 khi mất mặt (theo Section III-B bài báo LiteFat)
        landmarks_norm = np.ones((T, 68, 2), dtype=np.float32)
        landmarks_px = np.zeros((T, 68, 2), dtype=np.float32)

        if not valid_indices:
            return landmarks_norm, landmarks_px

        crop_coords = []
        crops_resized = []

        for idx in valid_indices:
            bbox = chosen_bboxes[idx]
            bx1, by1, bx2, by2 = bbox[:4]
            bw = bx2 - bx1 + 1.0
            bh = by2 - by1 + 1.0

            # Căn chỉnh lề theo đúng logic của PIPNet 300W (torchlm RuntimeWrapper)
            cx1 = int(bx1 - bw * extend / 2.0)
            cy1 = int(by1 + bh * extend / 2.0)
            cx2 = int(bx2 + bw * extend / 2.0)
            cy2 = int(by2 + bh * extend / 2.0)

            cx1 = max(0, min(target_w - 2, cx1))
            cy1 = max(0, min(target_h - 2, cy1))
            cx2 = max(cx1 + 1, min(target_w, cx2))
            cy2 = max(cy1 + 1, min(target_h, cy2))

            crop_w = float(cx2 - cx1)
            crop_h = float(cy2 - cy1)
            crop_coords.append((cx1, cy1, crop_w, crop_h))

            # Cắt ảnh và chuyển BGR -> RGB cho PIPNet
            face_crop = frames[idx][cy1:cy2, cx1:cx2, ::-1]
            face_resized = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            crops_resized.append(face_resized)

        # Gom lô N_valid vùng mặt lên GPU và giải mã theo từng sub-batch (max_chunk=64) để tránh tràn VRAM
        max_crop_chunk = 64
        batched_lms_list = []
        norm_factor = 1.0 * self.lmk_base.input_size / self.lmk_base.net_stride  # 256 / 32 = 8.0
        num_nb = self.lmk_base.num_nb
        max_len = self.lmk_base.max_len

        for s in range(0, len(crops_resized), max_crop_chunk):
            chunk_crops = crops_resized[s:s + max_crop_chunk]
            crops_np = np.stack(chunk_crops, axis=0)  # [cur_b, 256, 256, 3]
            crops_tensor = torch.from_numpy(crops_np).permute(0, 3, 1, 2).to(self.device).float() / 255.0
            crops_tensor = (crops_tensor - self.pipnet_mean) / self.pipnet_std

            out_cls, out_x, out_y, out_nb_x, out_nb_y = self.lmk_net(crops_tensor)
            cur_B, num_lms, H_map, W_map = out_cls.shape

            out_cls = out_cls.view(cur_B, num_lms, -1)
            max_ids = torch.argmax(out_cls, dim=2)
            max_ids_col = max_ids.unsqueeze(-1)

            out_x_flat = out_x.view(cur_B, num_lms, -1)
            out_y_flat = out_y.view(cur_B, num_lms, -1)
            x_sel = torch.gather(out_x_flat, 2, max_ids_col).squeeze(2)
            y_sel = torch.gather(out_y_flat, 2, max_ids_col).squeeze(2)

            max_ids_nb = max_ids_col.repeat(1, 1, num_nb).view(cur_B, num_lms * num_nb, 1)
            out_nb_x_flat = out_nb_x.view(cur_B, num_lms * num_nb, -1)
            out_nb_y_flat = out_nb_y.view(cur_B, num_lms * num_nb, -1)
            nb_x_sel = torch.gather(out_nb_x_flat, 2, max_ids_nb).view(cur_B, num_lms, num_nb)
            nb_y_sel = torch.gather(out_nb_y_flat, 2, max_ids_nb).view(cur_B, num_lms, num_nb)

            pred_x = ((max_ids % W_map).float() + x_sel) / norm_factor
            pred_y = (torch.floor(max_ids.float() / W_map) + y_sel) / norm_factor

            pred_nb_x = ((max_ids % W_map).float().unsqueeze(-1) + nb_x_sel) / norm_factor
            pred_nb_y = (torch.floor(max_ids.float() / W_map).unsqueeze(-1) + nb_y_sel) / norm_factor

            # Giải mã NRM vector hóa qua bảng ánh xạ reverse indices
            pred_nb_x_flat = pred_nb_x.view(cur_B, -1)
            pred_nb_y_flat = pred_nb_y.view(cur_B, -1)

            nb_x_rev = pred_nb_x_flat[:, self.rev_indices].view(cur_B, num_lms, max_len)
            nb_y_rev = pred_nb_y_flat[:, self.rev_indices].view(cur_B, num_lms, max_len)

            final_x = torch.mean(torch.cat([pred_x.unsqueeze(-1), nb_x_rev], dim=-1), dim=-1)
            final_y = torch.mean(torch.cat([pred_y.unsqueeze(-1), nb_y_rev], dim=-1), dim=-1)
            chunk_lms = torch.stack([final_x, final_y], dim=-1).cpu().numpy()  # [cur_B, 68, 2] in [0, 1]
            batched_lms_list.append(chunk_lms)

        batched_lms = np.concatenate(batched_lms_list, axis=0) if batched_lms_list else np.empty((0, 68, 2),
                                                                                                 dtype=np.float32)

        # Ánh xạ ngược tọa độ từ crop về khung hình letterbox
        for i, idx in enumerate(valid_indices):
            cx1, cy1, cw, ch = crop_coords[i]
            score = float(chosen_bboxes[idx][4])

            lm_crop = batched_lms[i].copy()
            abs_x = lm_crop[:, 0] * cw + cx1
            abs_y = lm_crop[:, 1] * ch + cy1

            landmarks_px[idx, :, 0] = abs_x
            landmarks_px[idx, :, 1] = abs_y

            # Chuẩn hóa về [0, 1] trên kích thước letterbox (chỉ lấy tọa độ x, y)
            landmarks_norm[idx, :, 0] = abs_x / float(target_w)
            landmarks_norm[idx, :, 1] = abs_y / float(target_h)

        return landmarks_norm, landmarks_px

    def compute_face_crop_boxes(
            self,
            chosen_bboxes: List[Optional[np.ndarray]],
            landmarks_px: np.ndarray,
            found: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính toán tọa độ điểm cắt khuôn mặt vuông cho từng khung hình:
          1. Kết hợp BBox và cực trị 68 điểm mốc (Landmark-Guided BBox Refinement).
          2. Mở rộng lề margin 15% đối xứng, giữ tỷ lệ khung hình vuông (1:1).
          3. Nội suy tuyến tính (Linear Interpolation) cho các frame mất mặt.
          4. Làm mượt thời gian bằng bộ lọc EMA (Exponential Moving Average).
          5. Bounding và ép kiểu nguyên [x1, y1, x2, y2] trong khung hình letterbox.

        Returns:
          - crop_boxes: np.ndarray [T, 4] dtype=np.int32: [x1, y1, x2, y2]
          - smoothed_boxes: np.ndarray [T, 4] dtype=np.float32: tọa độ float sau EMA
        """
        T = len(chosen_bboxes)
        target_w, target_h = self.letterbox_size
        raw_boxes = np.zeros((T, 4), dtype=np.float32)

        for t in range(T):
            if found[t] and chosen_bboxes[t] is not None:
                bx1, by1, bx2, by2 = chosen_bboxes[t][:4]
                lms = landmarks_px[t]  # [68, 2]
                lx1, ly1 = float(np.min(lms[:, 0])), float(np.min(lms[:, 1]))
                lx2, ly2 = float(np.max(lms[:, 0])), float(np.max(lms[:, 1]))

                # Hợp nhất hộp dò mặt và cực trị điểm mốc để không lẹm cằm/mắt
                fx1 = min(bx1, lx1)
                fy1 = min(by1, ly1)
                fx2 = max(bx2, lx2)
                fy2 = max(by2, ly2)

                fw = fx2 - fx1
                fh = fy2 - fy1
                cx1 = fx1 - fw * self.bbox_margin
                cy1 = fy1 - fh * self.bbox_margin
                cx2 = fx2 + fw * self.bbox_margin
                cy2 = fy2 + fh * self.bbox_margin

                # Giữ khung vuông quanh tâm khuôn mặt
                center_x = (cx1 + cx2) / 2.0
                center_y = (cy1 + cy2) / 2.0
                side = max(cx2 - cx1, cy2 - cy1)

                raw_boxes[t] = [
                    center_x - side / 2.0,
                    center_y - side / 2.0,
                    center_x + side / 2.0,
                    center_y + side / 2.0
                ]

        # 2. Nội suy tuyến tính cho các frame mất mặt
        valid_indices = [t for t in range(T) if found[t]]
        if not valid_indices:
            side = min(target_w, target_h) * 0.8
            cx, cy = target_w / 2.0, target_h / 2.0
            default_box = np.array([cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0],
                                   dtype=np.float32)
            for t in range(T):
                raw_boxes[t] = default_box
        else:
            first_idx = valid_indices[0]
            for t in range(first_idx):
                raw_boxes[t] = raw_boxes[first_idx]
            last_idx = valid_indices[-1]
            for t in range(last_idx + 1, T):
                raw_boxes[t] = raw_boxes[last_idx]
            for i in range(len(valid_indices) - 1):
                idx_a = valid_indices[i]
                idx_b = valid_indices[i + 1]
                if idx_b - idx_a > 1:
                    box_a = raw_boxes[idx_a]
                    box_b = raw_boxes[idx_b]
                    for t in range(idx_a + 1, idx_b):
                        alpha = float(t - idx_a) / float(idx_b - idx_a)
                        raw_boxes[t] = (1.0 - alpha) * box_a + alpha * box_b

        # 3. Làm mượt thời gian bằng bộ lọc EMA
        smoothed_boxes = np.zeros_like(raw_boxes)
        smoothed_boxes[0] = raw_boxes[0]
        alpha = self.box_smooth_alpha
        for t in range(1, T):
            smoothed_boxes[t] = alpha * raw_boxes[t] + (1.0 - alpha) * smoothed_boxes[t - 1]

        # 4. Giới hạn tọa độ trong kích thước khung letterbox và ép kiểu int32
        crop_boxes = np.zeros((T, 4), dtype=np.int32)
        for t in range(T):
            bx1, by1, bx2, by2 = smoothed_boxes[t]
            x1 = max(0, min(target_w - 2, int(round(bx1))))
            y1 = max(0, min(target_h - 2, int(round(by1))))
            x2 = max(x1 + 1, min(target_w, int(round(bx2))))
            y2 = max(y1 + 1, min(target_h, int(round(by2))))
            crop_boxes[t] = [x1, y1, x2, y2]

        return crop_boxes, smoothed_boxes

    def _crop_faces_for_litefat(
            self,
            frames: List[np.ndarray],
            crop_boxes: np.ndarray,
    ) -> torch.Tensor:
        """
        Cắt vùng khuôn mặt chuẩn hóa 224x224 cho nhánh MobileNetV3-Small từ crop_boxes.
        """
        return crop_faces_from_boxes(frames, crop_boxes, face_size=self.face_size)

    def extract_from_frames(
            self,
            frames: List[np.ndarray],
            crop_faces: bool = True,
            return_crop_boxes: bool = False,
    ) -> Union[
        Tuple[Optional[torch.Tensor], torch.Tensor, List[Optional[np.ndarray]], List[bool]],
        Tuple[Optional[torch.Tensor], torch.Tensor, List[Optional[np.ndarray]], List[bool], np.ndarray],
    ]:
        """
        Pipeline trích xuất hoàn chỉnh từ chuỗi khung hình đã qua letterbox:
          - Phát hiện mặt batched GPU.
          - Chọn khuôn mặt tài xế ổn định.
          - Trích xuất 68 điểm mốc batched GPU.
          - Tính tọa độ điểm cắt khung mặt và (tùy chọn) cắt/chuẩn hóa khung hình mặt.

        Returns:
          - face_frames: Tensor [T, 3, 224, 224] (hoặc None nếu crop_faces=False)
          - landmarks:   Tensor [T, 68, 2] (X, Y)
          - bboxes:      Danh sách BBox chọn lọc của từng frame
          - found:       Danh sách boolean đánh dấu phát hiện được mặt
          - crop_boxes:  (kèm theo nếu return_crop_boxes=True) mảng np.ndarray [T, 4] dạng [x1, y1, x2, y2]
        """
        target_w, target_h = self.letterbox_size
        # Đảm bảo mọi frame đầu vào đúng kích thước letterbox
        uniform_frames = []
        for f in frames:
            if f.shape[1] != target_w or f.shape[0] != target_h:
                uniform_frames.append(letterbox(f, new_size=(target_w, target_h)))
            else:
                uniform_frames.append(f)

        # 1. Phát hiện mặt theo lô
        all_dets = self._detect_faces_batched(uniform_frames)

        # 2. Lựa chọn khuôn mặt tài xế ổn định
        chosen_bboxes, found = self._select_driver_faces(all_dets, target_w, target_h)

        # 3. Trích xuất 68 điểm mốc theo lô
        landmarks_norm, landmarks_px = self._extract_landmarks_batched(
            uniform_frames, chosen_bboxes, found
        )

        # 4. Tính toán tọa độ điểm cắt khuôn mặt
        crop_boxes, _ = self.compute_face_crop_boxes(chosen_bboxes, landmarks_px, found)

        # 5. Cắt khung hình khuôn mặt nếu cần
        face_frames_tensor = None
        if crop_faces:
            face_frames_tensor = self._crop_faces_for_litefat(uniform_frames, crop_boxes)

        landmarks_tensor = torch.from_numpy(landmarks_norm).float()

        if return_crop_boxes:
            return face_frames_tensor, landmarks_tensor, chosen_bboxes, found, crop_boxes
        return face_frames_tensor, landmarks_tensor, chosen_bboxes, found


# =====================================================================
# 2. ĐỌC NHANH KHUNG HÌNH TỪ VIDEO (FAST VIDEO SEEKER)
# =====================================================================
def fast_read_video_frames(
        video_path: Union[str, Path],
        target_indices: np.ndarray,
        letterbox_size: Tuple[int, int] = (640, 480),
) -> Tuple[List[np.ndarray], float, int]:
    """
    Đọc chính xác các chỉ số khung hình `target_indices` từ video với tốc độ cao:
      - Nhảy thẳng bằng CAP_PROP_POS_FRAMES tới vị trí đầu tiên.
      - Dùng cap.grab() để lướt qua các frame trung gian (tiết kiệm chi phí giải mã CPU).
      - Áp dụng ultis.letterbox về đúng kích thước chuẩn (target_w, target_h).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Không thể mở tệp video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_w, target_h = letterbox_size

    frames: List[np.ndarray] = []
    curr_frame = int(target_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

    try:
        for target_idx in target_indices:
            target_idx = int(target_idx)
            if target_idx < curr_frame and frames:
                frames.append(frames[-1].copy())
                continue

            while curr_frame < target_idx:
                if not cap.grab():
                    break
                curr_frame += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Chuẩn hóa qua ultis.letterbox
            lb_frame = letterbox(frame, new_size=(target_w, target_h))
            frames.append(lb_frame)
            curr_frame += 1
    finally:
        cap.release()

    if not frames:
        raise IOError(f"Không đọc được khung hình nào từ: {video_path}")

    # Bù khung hình nếu video kết thúc sớm hơn số lượng yêu cầu
    while len(frames) < len(target_indices):
        frames.append(frames[-1].copy())

    return frames, fps, total_frames


# =====================================================================
# 3. LỚP LITEFAT DATASET (HỖ TRỢ ĐA CHẾ ĐỘ LẤY MẪU & CACHE)
# =====================================================================
class LiteFatDataset(Dataset):
    """
    PyTorch Dataset nạp dữ liệu cho mô hình LiteFat từ E:\\LSTM\\data_processed:
      - Hỗ trợ nạp từ manifest dataset_merged_split.csv hoặc quét thư mục train/val/test.
      - Hỗ trợ 2 chế độ lấy mẫu thời gian:
          + Chế độ 1 (mặc định): Cố định num_frames (T=16) theo bài báo LiteFat.
            Train: Temporal Jitter (lấy ngẫu nhiên trong từng phân đoạn); Val/Test: Linspace đều đặn.
          + Chế độ 2: Lấy mẫu theo time_interval và window_sec.
      - Tiền xử lý toàn bộ khung hình qua ultis.letterbox (640, 480).
      - Trích xuất 68 điểm mốc bằng FastPIPNetExtractor (Batched GPU).
      - Cắt khung hình khuôn mặt 224x224 chuẩn hóa ImageNet cho MobileNetV3-Small.
      - Hỗ trợ nạp / lưu cache .pt để tăng tốc tối đa khi huấn luyện nhiều epoch.
    """

    def __init__(
            self,
            dataset_dir: Union[str, Path] = r"E:\LSTM\data_processed",
            split: str = "train",
            time_interval: float = 0.1,
            window_sec: Optional[float] = 10.0,
            manifest_filename: str = "dataset_merged_split.csv",
            letterbox_size: Tuple[int, int] = (640, 640),
            face_size: Tuple[int, int] = (224, 224),
            conf_threshold: float = 0.5,
            bbox_margin: float = 0.15,
            return_meta: bool = False,
            return_raw_frames: bool = False,
            use_cache: bool = False,
            cache_dir: Optional[Union[str, Path]] = None,
            extractor: Optional[FastPIPNetExtractor] = None,
            device: Optional[str] = None,
            extract_crops_only: bool = False,
            pipnet_backbone: str = "resnet18",
            nms_threshold: float = 0.3,
            box_smooth_alpha: float = 0.7,
            selected_datasets: Optional[Union[List[str], str]] = None,
            use_visual_context: bool = True,
            augment_train: bool = True,
            flip_prob: float = 0.5,
            scale_jitter: float = 0.08,
            shift_jitter: float = 0.05,
            roll_deg: float = 12.0,
            temporal_dropout: float = 0.05,
            landmark_noise_std: float = 0.002,
            color_jitter: float = 0.2,
            spatial_dropout: float = 0.15,
    ):
        split = split.lower().strip()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split không hợp lệ: '{split}'. Chỉ chấp nhận 'train', 'val', hoặc 'test'.")

        if time_interval is not None and float(time_interval) < 0:
            raise ValueError(f"time_interval phải là số không âm (>= 0): {time_interval}")
        if window_sec is not None and float(window_sec) <= 0:
            raise ValueError(f"window_sec phải là số dương hoặc None: {window_sec}")

        self.root = Path(dataset_dir)
        self.split = split
        self.time_interval = float(time_interval) if time_interval is not None else 0.1
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.manifest_filename = manifest_filename
        self.letterbox_size = _parse_size(letterbox_size)
        self.face_size = _parse_size(face_size)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.bbox_margin = bbox_margin
        self.box_smooth_alpha = box_smooth_alpha
        self.pipnet_backbone = pipnet_backbone
        self.device = device
        self.return_meta = bool(return_meta)
        self.return_raw_frames = bool(return_raw_frames)
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        # Cấu hình bối cảnh thị giác (MobileNetV3) & Đa phương thức
        self.use_visual_context = bool(use_visual_context)
        self.extract_crops_only = bool(extract_crops_only) or (not self.use_visual_context)

        # Cấu hình lọc bộ dữ liệu con
        if selected_datasets is not None:
            if isinstance(selected_datasets, str):
                self.selected_datasets = [selected_datasets.lower()]
            else:
                self.selected_datasets = [str(s).lower() for s in selected_datasets if str(s).strip()]
                if not self.selected_datasets or any(s in ("all", "*") for s in self.selected_datasets):
                    self.selected_datasets = None
        else:
            self.selected_datasets = None

        # Cấu hình tăng cường dữ liệu
        self.augment_train = bool(augment_train)
        self.flip_prob = float(flip_prob)
        self.scale_jitter = float(scale_jitter)
        self.shift_jitter = float(shift_jitter)
        self.roll_deg = float(roll_deg)
        self.temporal_dropout = float(temporal_dropout)
        self.landmark_noise_std = float(landmark_noise_std)
        self.color_jitter = float(color_jitter)
        self.spatial_dropout = float(spatial_dropout)

        self.cache_meta: Dict[str, Any] = {}
        self.cached_data: Optional[Dict[str, Any]] = None

        # Nạp cache từ file .pt nếu được yêu cầu
        if self.use_cache and self.cache_dir is not None:
            self._load_cache()

        # Nạp danh sách mẫu (ưu tiên manifest CSV hoặc quét thư mục, nếu rỗng thì lấy từ cache)
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            if self.cached_data is not None and len(self.cached_data) > 0:
                logger.info(f"[Cache] Tự động khởi tạo mẫu từ file cache .pt (selected_datasets={self.selected_datasets}).")
                for k, it in self.cached_data.items():
                    m = it.get("meta", {})
                    rel_p = m.get("key", k)
                    if not self._is_dataset_selected(rel_p, m.get("dataset")):
                        continue
                    self.samples.append({
                        "key": rel_p,
                        "path": self.root / rel_p,
                        "label": int(it.get("label", m.get("label", 0))),
                        "video_id": m.get("video_id", Path(rel_p).stem),
                        "fps": float(m.get("fps", 30.0)),
                        "frame_count": int(float(m.get("total_frames", 0))),
                        "duration_sec": float(m.get("duration_sec", 0.0)),
                    })
                logger.info(f"[Cache] Số mẫu sau lọc dataset: {len(self.samples)}")
            else:
                logger.warning(f"Không tìm thấy mẫu video nào trong {self.root} cho split='{self.split}'")

        # Khởi tạo bộ trích xuất PIPNet:
        # Nếu đã có cache và không cung cấp extractor từ ngoài, trì hoãn (Lazy Init)
        # để không tốn VRAM tải FaceBoxesV2 + PIPNet khi chỉ nạp từ cache.
        self.extractor = extractor
        if self.extractor is None and not (self.use_cache and self.cached_data is not None):
            self.extractor = self._get_or_create_extractor()

    def _get_or_create_extractor(self) -> FastPIPNetExtractor:
        """Khởi tạo trễ (Lazy init) FastPIPNetExtractor khi thực sự cần trích xuất trực tiếp."""
        if self.extractor is None:
            self.extractor = FastPIPNetExtractor(
                backbone=self.pipnet_backbone,
                device=self.device,
                letterbox_size=self.letterbox_size,
                face_size=self.face_size,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                bbox_margin=self.bbox_margin,
                box_smooth_alpha=self.box_smooth_alpha,
            )
        return self.extractor

    def _is_dataset_selected(self, key: str, dataset_field: Optional[str] = None) -> bool:
        """Kiểm tra xem mẫu có thuộc danh sách selected_datasets được chọn hay không."""
        if self.selected_datasets is None:
            return True
        key_lower = key.lower()
        if dataset_field and any(ds in dataset_field.lower() for ds in self.selected_datasets):
            return True
        return any(ds in key_lower for ds in self.selected_datasets)

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Đọc danh sách mẫu video từ manifest CSV hoặc quét thư mục fallback."""
        manifest_path = self.root / self.manifest_filename
        samples = []

        if manifest_path.is_file():
            with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split", "").strip().lower() == self.split:
                        rel_path = row.get("rel_path") or row.get("processed_path")
                        if not rel_path:
                            continue
                        norm_rel = Path(rel_path.replace("\\", "/")).as_posix()
                        if not self._is_dataset_selected(norm_rel, row.get("dataset")):
                            continue
                        full_path = self.root / norm_rel
                        samples.append({
                            "key": norm_rel,
                            "path": full_path,
                            "label": int(row["label"]),
                            "video_id": row.get("video_id") or Path(rel_path).stem,
                            "subject_id": str(row.get("subject_id", "") or "unknown"),
                            "dataset": str(row.get("dataset", "") or "unknown"),
                            "fps": float(row.get("fps", 30.0)),
                            "frame_count": int(float(row.get("frame_count", 0))),
                            "duration_sec": float(row.get("duration_sec", 0.0)),
                        })
        else:
            # Fallback quét thư mục train/val/test/0_alert và 1_drowsy
            for label, folder in ((0, "0_alert"), (1, "1_drowsy")):
                split_folder = self.root / self.split / folder
                if split_folder.is_dir():
                    for p in sorted(split_folder.glob("*.mp4")):
                        rel_key = p.relative_to(self.root).as_posix()
                        if not self._is_dataset_selected(rel_key):
                            continue
                        samples.append({
                            "key": rel_key,
                            "path": p,
                            "label": label,
                            "video_id": p.stem,
                            "subject_id": "unknown",
                            "dataset": "unknown",
                            "fps": 30.0,
                            "frame_count": 0,
                            "duration_sec": 0.0,
                        })
        return samples

    def _load_cache(self) -> None:
        """Nạp dữ liệu cache từ tệp .pt và lưu trữ metadata của cache."""
        cache_path = self.cache_dir / f"{self.split}.pt"
        if not cache_path.is_file():
            logger.warning(f"[Cache] Không tìm thấy tệp cache {cache_path}. Sẽ fallback trích xuất on-the-fly.")
            return

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "samples" in payload:
                self.cached_data = payload["samples"]
                self.cache_meta = {
                    "format_version": payload.get("format_version", 1),
                    "split": payload.get("split", self.split),
                    "time_interval": float(payload.get("time_interval", 0.05)),
                    "window_sec": payload.get("window_sec", None),
                    "letterbox_size": tuple(payload.get("letterbox_size", self.letterbox_size)),
                    "face_size": tuple(payload.get("face_size", self.face_size)),
                    "bbox_margin": float(payload.get("bbox_margin", self.bbox_margin)),
                    "box_smooth_alpha": float(payload.get("box_smooth_alpha", self.box_smooth_alpha)),
                    "num_samples": payload.get("num_samples", len(self.cached_data)),
                }
            elif isinstance(payload, dict):
                self.cached_data = payload
                self.cache_meta = {
                    "time_interval": 0.05,
                    "window_sec": None,
                    "letterbox_size": self.letterbox_size,
                    "face_size": self.face_size,
                }

            cached_lbs = self.cache_meta.get("letterbox_size")
            if cached_lbs and tuple(cached_lbs) != tuple(self.letterbox_size):
                logger.warning(
                    f"[Cache] File {cache_path.name} được trích xuất với letterbox_size={cached_lbs}, "
                    f"khác với cấu hình hiện tại {self.letterbox_size}. "
                    f"Sẽ sử dụng letterbox_size của cache để bảo đảm tọa độ crop_boxes khớp tuyệt đối."
                )

            logger.info(f"[Cache] Đã nạp thành công cache {cache_path.name} với {len(self.cached_data)} mẫu.")
        except Exception as e:
            logger.warning(f"[Cache] Lỗi khi nạp cache {cache_path}: {e}. Bỏ qua cache.")

    def _get_frame_indices(self, total_frames: int, fps: float) -> np.ndarray:
        """
        Tính toán danh sách chỉ số khung hình cần lấy từ video:
          - window_sec = None -> Cửa sổ trượt là TOÀN BỘ video (0 đến total_frames - 1).
          - window_sec > 0    -> Cửa sổ trượt dài window_sec giây (Random crop khi train, Center crop khi val/test).
          - time_interval == 0 -> Lấy liên tiếp mọi khung hình trong cửa sổ (không có khoảng cách thời gian giữa các frame).
          - time_interval > 0  -> Lấy mẫu cách đều sau mỗi time_interval giây trên trục thời gian thực.
        """
        if total_frames <= 0:
            return np.array([0], dtype=int)

        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps

        # TRƯỜNG HỢP 1: time_interval == 0 (Lấy các khung hình liên tiếp không bỏ sót frame nào)
        if self.time_interval == 0:
            if self.window_sec is None:
                # window_sec = None: lấy toàn bộ video liên tiếp
                return np.arange(total_frames, dtype=int)
            else:
                win_sec = min(self.window_sec, duration_sec)
                num_consec = int(round(win_sec * fps))
                num_consec = max(1, min(num_consec, total_frames))
                max_start = max(0, total_frames - num_consec)
                start_idx = random.randint(0, max_start) if (
                            self.split == "train" and max_start > 0) else max_start // 2
                return np.arange(start_idx, start_idx + num_consec, dtype=int)

        # TRƯỜNG HỢP 2: time_interval > 0 (Lấy mẫu cách đều theo thời gian thực)
        if self.window_sec is None:
            # window_sec = None: toàn bộ video cách đều mỗi time_interval giây
            timestamps = np.arange(0, duration_sec, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            indices = np.rint(timestamps * fps).astype(int)
            return np.clip(indices, 0, total_frames - 1)
        else:
            win = min(self.window_sec, duration_sec)
            max_start = max(0.0, duration_sec - win)
            start_sec = random.uniform(0.0, max_start) if (self.split == "train" and max_start > 0) else max_start / 2.0
            timestamps = start_sec + np.arange(0, win, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([start_sec])
            indices = np.rint(timestamps * fps).astype(int)
            return np.clip(indices, 0, total_frames - 1)

    def _slice_cached_sequence(self, item: Dict[str, Any], fps: float) -> np.ndarray:
        """
        Xác định các chỉ số khung hình (pos) cần lấy từ chuỗi cache .pt:
          - Dựa vào time_interval và window_sec được cấu hình trong LiteFatDataset.
          - Tự động phát hiện khoảng cách thời gian giữa các frame trong cache (cache_dt).
          - Hỗ trợ cắt ngẫu nhiên (train) và cắt chính giữa (val/test).
        """
        landmarks = item["landmarks"]
        T_cache = int(landmarks.shape[0])
        if T_cache <= 0:
            return np.array([0], dtype=int)

        cache_dt = float(self.cache_meta.get("time_interval", 0.05))
        if cache_dt <= 0:
            cache_dt = 1.0 / (fps if fps > 0 else 30.0)

        meta = item.get("meta", {})
        duration_sec = float(meta.get("duration_sec", T_cache * cache_dt))
        if duration_sec <= 0:
            duration_sec = T_cache * cache_dt

        # TRƯỜNG HỢP 1: time_interval == 0 (Lấy chuỗi khung hình liên tiếp)
        if self.time_interval == 0:
            if self.window_sec is None:
                return np.arange(T_cache, dtype=int)
            else:
                win_sec = min(self.window_sec, duration_sec)
                num_consec = int(round(win_sec / cache_dt))
                num_consec = max(1, min(num_consec, T_cache))
                max_start = max(0, T_cache - num_consec)
                start_idx = random.randint(0, max_start) if (
                            self.split == "train" and max_start > 0) else max_start // 2
                return np.arange(start_idx, start_idx + num_consec, dtype=int)

        # TRƯỜNG HỢP 2: time_interval > 0 (Lấy mẫu cách đều theo thời gian)
        if self.window_sec is None:
            # Nếu time_interval cấu hình trùng khớp với cache_dt: giữ nguyên toàn bộ cache
            if abs(self.time_interval - cache_dt) < 1e-5:
                return np.arange(T_cache, dtype=int)
            timestamps = np.arange(0, duration_sec, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            pos = np.rint(timestamps / cache_dt).astype(int)
            return np.clip(pos, 0, T_cache - 1)
        else:
            if self.split != "train":
                # Đối với tập val và test: Đánh giá trên tối đa 18.0 giây (360 frames)
                # để bao quát trọn vẹn toàn bộ chuỗi video UL-DD/VBDDD, không bị cắt cụt mất sự kiện buồn ngủ
                eval_win = min(max(self.window_sec or 10.0, 18.0), duration_sec)
                timestamps = np.arange(0, eval_win, self.time_interval)
                if len(timestamps) == 0:
                    timestamps = np.array([0.0])
                pos = np.rint(timestamps / cache_dt).astype(int)
                return np.clip(pos, 0, T_cache - 1)
            else:
                win = min(self.window_sec, duration_sec)
                max_start = max(0.0, duration_sec - win)
                start_sec = random.uniform(0.0, max_start) if max_start > 0 else 0.0
                timestamps = start_sec + np.arange(0, win, self.time_interval)
                if len(timestamps) == 0:
                    timestamps = np.array([start_sec])
                pos = np.rint(timestamps / cache_dt).astype(int)
                return np.clip(pos, 0, T_cache - 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor],
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]],
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], List[np.ndarray]],
    ]:
        sample = self.samples[index]
        key = sample["key"]

        # 1. Đường nhanh: Nạp từ RAM Cache nếu có
        if self.cached_data is not None and key in self.cached_data:
            item = self.cached_data[key]
            fps = float(sample.get("fps") or item.get("meta", {}).get("fps", 30.0))
            pos = self._slice_cached_sequence(item, fps)

            # Cắt các đặc trưng mốc và hộp giới hạn theo pos (tương thích cache 2 kênh hoặc 3 kênh cũ)
            landmarks = item["landmarks"][pos]
            if landmarks.shape[-1] > 2:
                landmarks = landmarks[..., :2]
            crop_boxes = item["crop_boxes"][pos] if "crop_boxes" in item else None
            bboxes = item["bboxes"][pos] if "bboxes" in item else None
            found = item["found"][pos] if "found" in item else None

            # Lấy target_indices tương ứng trong video gốc
            cached_target_indices = item.get("target_indices")
            if cached_target_indices is None and "meta" in item and isinstance(item["meta"], dict):
                cached_target_indices = item["meta"].get("target_indices")

            if cached_target_indices is not None:
                if isinstance(cached_target_indices, torch.Tensor):
                    selected_target_indices = cached_target_indices[pos].long().cpu().numpy()
                else:
                    selected_target_indices = np.asarray(cached_target_indices, dtype=int)[pos]
            else:
                cache_dt = float(self.cache_meta.get("time_interval", 0.05))
                selected_target_indices = np.rint(pos * (cache_dt * fps)).astype(int)

            cache_lbs = self.cache_meta.get("letterbox_size", self.letterbox_size)
            face_frames = item.get("face_frames", None)
            if face_frames is not None:
                face_frames = face_frames[pos]

            raw_frames: Optional[List[np.ndarray]] = None

            # Đọc video nếu cần tạo face_frames (cho MobileNetV3) hoặc người dùng yêu cầu return_raw_frames
            need_video_read = (not self.extract_crops_only and face_frames is None) or self.return_raw_frames
            if need_video_read:
                raw_frames, actual_fps, _ = fast_read_video_frames(
                    sample["path"], selected_target_indices, letterbox_size=cache_lbs
                )
                if not self.extract_crops_only and face_frames is None and crop_boxes is not None:
                    face_frames = crop_faces_from_boxes(raw_frames, crop_boxes, face_size=self.face_size)

            if not self.use_visual_context:
                face_frames = None

            if self.split == "train" and self.augment_train:
                face_frames, landmarks = augment_multimodal_clip(
                    face_frames,
                    landmarks,
                    is_train=True,
                    flip_prob=self.flip_prob,
                    scale_jitter=self.scale_jitter,
                    shift_jitter=self.shift_jitter,
                    roll_deg=self.roll_deg,
                    temporal_dropout=self.temporal_dropout,
                    landmark_noise_std=self.landmark_noise_std,
                    color_jitter=self.color_jitter,
                    spatial_dropout=self.spatial_dropout,
                )

            label = torch.tensor(sample["label"], dtype=torch.long)

            if found is not None:
                found_t = torch.as_tensor(found, dtype=torch.bool)
                found_ratio = float(found_t.sum().item()) / float(max(1, len(found_t)))
            else:
                found_t = torch.ones(len(pos), dtype=torch.bool)
                found_ratio = 1.0

            meta = {
                "key": key,
                "video_id": sample["video_id"],
                "subject_id": str(sample.get("subject_id") or (item.get("meta", {}).get("subject_id", "unknown"))),
                "dataset": str(sample.get("dataset") or (item.get("meta", {}).get("dataset", "unknown"))),
                "path": str(sample["path"]),
                "split": self.split,
                "label": int(sample["label"]),
                "fps": float(fps),
                "total_frames": int(sample.get("frame_count") or item.get("meta", {}).get("total_frames", 0)),
                "duration_sec": float(sample.get("duration_sec") or item.get("meta", {}).get("duration_sec", 0.0)),
                "target_indices": torch.as_tensor(selected_target_indices, dtype=torch.long),
                "num_frames": int(landmarks.shape[0]),
                "found_faces_ratio": found_ratio,
                "letterbox_size": cache_lbs,
                "cached": True,
            }
            if crop_boxes is not None:
                meta["crop_boxes"] = torch.as_tensor(crop_boxes, dtype=torch.int32)
            if bboxes is not None:
                meta["bboxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
            if found is not None:
                meta["found"] = found_t

            if "meta" in item and isinstance(item["meta"], dict):
                for k_meta, v_meta in item["meta"].items():
                    if k_meta not in meta and k_meta != "path":
                        meta[k_meta] = v_meta
                if "path" in item["meta"] and str(item["meta"]["path"]) != str(sample["path"]):
                    meta["remote_path"] = str(item["meta"]["path"])

            if self.return_raw_frames and self.return_meta:
                return face_frames, landmarks, label, meta, (raw_frames if raw_frames is not None else [])
            if self.return_meta:
                return face_frames, landmarks, label, meta
            return face_frames, landmarks, label

        # 2. Đường chuẩn: Đọc video và trích xuất on-the-fly
        extractor = self._get_or_create_extractor()
        fps = sample["fps"]
        total_frames = sample["frame_count"]

        if fps <= 0 or total_frames <= 0:
            cap = cv2.VideoCapture(str(sample["path"]))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)
            cap.release()
            sample["fps"] = fps
            sample["frame_count"] = total_frames

        target_indices = self._get_frame_indices(total_frames, fps)
        raw_frames, actual_fps, _ = fast_read_video_frames(
            sample["path"], target_indices, letterbox_size=self.letterbox_size
        )

        crop_faces = not self.extract_crops_only
        face_frames, landmarks, bboxes, found, crop_boxes = extractor.extract_from_frames(
            raw_frames,
            crop_faces=crop_faces,
            return_crop_boxes=True,
        )

        if not self.use_visual_context:
            face_frames = None

        if self.split == "train" and self.augment_train:
            face_frames, landmarks = augment_multimodal_clip(
                face_frames,
                landmarks,
                is_train=True,
                flip_prob=self.flip_prob,
                scale_jitter=self.scale_jitter,
                shift_jitter=self.shift_jitter,
                roll_deg=self.roll_deg,
                temporal_dropout=self.temporal_dropout,
                landmark_noise_std=self.landmark_noise_std,
                color_jitter=self.color_jitter,
                spatial_dropout=self.spatial_dropout,
            )

        label = torch.tensor(sample["label"], dtype=torch.long)

        # Chuẩn hóa BBoxes và Found thành Tensor để dễ lưu .pt và xử lý
        T_frames = len(found)
        bboxes_tensor = torch.zeros((T_frames, 5), dtype=torch.float32)
        for t_idx, b in enumerate(bboxes):
            if b is not None and len(b) >= 4:
                b_arr = np.asarray(b, dtype=np.float32)
                bboxes_tensor[t_idx, :min(5, len(b_arr))] = torch.from_numpy(b_arr[:min(5, len(b_arr))])
        found_tensor = torch.tensor(found, dtype=torch.bool)
        crop_boxes_tensor = torch.from_numpy(crop_boxes).int()

        meta = {
            "key": key,
            "video_id": sample["video_id"],
            "subject_id": str(sample.get("subject_id", "unknown")),
            "dataset": str(sample.get("dataset", "unknown")),
            "path": str(sample["path"]),
            "split": self.split,
            "label": int(sample["label"]),
            "fps": float(actual_fps),
            "total_frames": int(total_frames),
            "duration_sec": float(total_frames / actual_fps) if actual_fps > 0 else 0.0,
            "target_indices": torch.from_numpy(target_indices).long(),
            "num_frames": int(landmarks.shape[0]),
            "found_faces_ratio": float(sum(found)) / float(max(1, len(found))),
            "letterbox_size": self.letterbox_size,
            "crop_boxes": crop_boxes_tensor,
            "bboxes": bboxes_tensor,
            "found": found_tensor,
            "raw_bboxes_list": bboxes,
            "cached": False,
        }

        if self.return_raw_frames and self.return_meta:
            return face_frames, landmarks, label, meta, raw_frames
        if self.return_meta:
            return face_frames, landmarks, label, meta
        return face_frames, landmarks, label


# =====================================================================
# 4. HÀM COLLATE & XÂY DỰNG DATALOADER
# =====================================================================
def collate_litefat(batch: List[Tuple[Any, ...]]):
    """
    Collate function cho DataLoader LiteFat.
    Hỗ trợ tự động đệm (padding) theo chiều thời gian T nếu độ dài giữa các mẫu khác nhau
    (khi sử dụng chế độ num_frames=None, time_interval).
    Hỗ trợ cả trường hợp face_frames là None (khi chạy ở chế độ extract_crops_only=True).
    """
    first_item = batch[0]
    num_elements = len(first_item)

    face_frames_list = [item[0] for item in batch]
    landmarks_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]

    lengths = [lm.shape[0] for lm in landmarks_list]
    max_len = max(lengths)

    has_face_frames = all(f is not None for f in face_frames_list)

    # Đệm theo chiều thời gian T nếu các video có số frame khác nhau
    padded_frames = []
    padded_landmarks = []
    for idx, (lm, cur_len) in enumerate(zip(landmarks_list, lengths)):
        f = face_frames_list[idx]
        if cur_len < max_len:
            pad_t = max_len - cur_len
            if has_face_frames and f is not None:
                # Đệm frame bằng frame cuối cùng
                last_frame = f[-1:].repeat(pad_t, 1, 1, 1)
                padded_f = torch.cat([f, last_frame], dim=0)
                padded_frames.append(padded_f)
            # Đệm landmark bằng giá trị 1.0 (chuẩn bài báo LiteFat Section III-B)
            pad_lm = torch.ones((pad_t, lm.shape[1], lm.shape[2]), dtype=lm.dtype)
            padded_l = torch.cat([lm, pad_lm], dim=0)
            padded_landmarks.append(padded_l)
        else:
            if has_face_frames and f is not None:
                padded_frames.append(f)
            padded_landmarks.append(lm)

    batch_frames = torch.stack(padded_frames, dim=0) if has_face_frames else None  # [B, T, 3, 224, 224] hoặc None
    batch_landmarks = torch.stack(padded_landmarks, dim=0)  # [B, T, 68, 2]
    batch_labels = torch.stack(labels_list, dim=0)  # [B]
    batch_lengths = torch.tensor(lengths, dtype=torch.long)  # [B]

    if num_elements >= 4:
        metas = [item[3] for item in batch]
        if num_elements == 5:
            raw_frames_list = [item[4] for item in batch]
            return batch_frames, batch_landmarks, batch_labels, batch_lengths, metas, raw_frames_list
        return batch_frames, batch_landmarks, batch_labels, batch_lengths, metas

    return batch_frames, batch_landmarks, batch_labels, batch_lengths


def build_litefat_dataloaders(
        dataset_dir: Union[str, Path] = r"E:\LSTM\data_processed",
        batch_size: int = 8,
        num_workers: int = 0,
        time_interval: float = 0.05,
        window_sec: Optional[float] = 15.0,
        letterbox_size: Tuple[int, int] = (640, 640),
        face_size: Tuple[int, int] = (224, 224),
        manifest_filename: str = "dataset_merged_split.csv",
        use_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        device: Optional[str] = None,
        pipnet_backbone: str = "resnet18",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        bbox_margin: float = 0.15,
        box_smooth_alpha: float = 0.7,
        extract_crops_only: bool = False,
        selected_datasets: Optional[Union[List[str], str]] = None,
        use_visual_context: bool = True,
        augment_train: bool = True,
        flip_prob: float = 0.5,
        scale_jitter: float = 0.08,
        shift_jitter: float = 0.05,
        roll_deg: float = 12.0,
        temporal_dropout: float = 0.05,
        landmark_noise_std: float = 0.0003,
        color_jitter: float = 0.2,
        spatial_dropout: float = 0.15,
        subject_balanced_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khởi tạo nhanh bộ 3 DataLoader cho tập train, val, test của mô hình LiteFat theo cơ chế window_sec và time_interval.
    """
    # Khởi tạo 1 extractor dùng chung cho cả 3 tập nếu không dùng cache để tiết kiệm VRAM và tăng tốc khởi tạo
    shared_extractor = FastPIPNetExtractor(
        backbone=pipnet_backbone,
        device=device,
        letterbox_size=letterbox_size,
        face_size=face_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        bbox_margin=bbox_margin,
        box_smooth_alpha=box_smooth_alpha,
    ) if not use_cache else None

    common_cfg = dict(
        dataset_dir=dataset_dir,
        time_interval=time_interval,
        window_sec=window_sec,
        manifest_filename=manifest_filename,
        letterbox_size=letterbox_size,
        face_size=face_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        bbox_margin=bbox_margin,
        box_smooth_alpha=box_smooth_alpha,
        pipnet_backbone=pipnet_backbone,
        use_cache=use_cache,
        cache_dir=cache_dir,
        extractor=shared_extractor,
        device=device,
        extract_crops_only=extract_crops_only,
        selected_datasets=selected_datasets,
        use_visual_context=use_visual_context,
    )

    train_ds = LiteFatDataset(
        split="train",
        augment_train=augment_train,
        flip_prob=flip_prob,
        scale_jitter=scale_jitter,
        shift_jitter=shift_jitter,
        roll_deg=roll_deg,
        temporal_dropout=temporal_dropout,
        landmark_noise_std=landmark_noise_std,
        color_jitter=color_jitter,
        spatial_dropout=spatial_dropout,
        **common_cfg,
    )
    val_ds = LiteFatDataset(split="val", augment_train=False, **common_cfg)
    test_ds = LiteFatDataset(split="test", augment_train=False, **common_cfg)

    train_sampler = None
    if subject_balanced_sampling and len(train_ds) > 0:
        from collections import Counter
        sub_label_counts = Counter()
        label_counts = Counter()
        for s in train_ds.samples:
            sub = s.get("subject_id", "unknown")
            lbl = s.get("label", 0)
            sub_label_counts[(sub, lbl)] += 1
            label_counts[lbl] += 1

        sample_weights = []
        for s in train_ds.samples:
            sub = s.get("subject_id", "unknown")
            lbl = s.get("label", 0)
            cnt_sub = max(1, sub_label_counts[(sub, lbl)])
            cnt_lbl = max(1, label_counts[lbl])
            w = 1.0 / (math.sqrt(cnt_sub) * cnt_lbl)
            sample_weights.append(w)

        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(shuffle_train if train_sampler is None else False),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_litefat
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_litefat
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_litefat
    )

    return train_loader, val_loader, test_loader


# =====================================================================
# 5. TRỰC QUAN HÓA MẪU DỮ LIỆU & MÔ PHỎNG KINEMATIC FEATURE EXTRACTOR
# =====================================================================
# Các đường nối giải phẫu 68 điểm mốc theo chuẩn iBUG 300W cho mô phỏng động học
KINEMATIC_REGIONS = [
    (0, 17, False, (255, 190, 0)),   # Đường viền hàm (0 - 16) - Xanh lam
    (17, 22, False, (0, 230, 255)),  # Lông mày trái (17 - 21) - Vàng
    (22, 27, False, (0, 230, 255)),  # Lông mày phải (22 - 26) - Vàng
    (27, 31, False, (220, 0, 255)),  # Sống mũi (27 - 30) - Tím hồng
    (31, 36, False, (220, 0, 255)),  # Chóp mũi dưới (31 - 35) - Tím hồng
    (36, 42, True, (0, 255, 100)),   # Mắt trái (36 - 41) - Xanh lá
    (42, 48, True, (0, 255, 100)),   # Mắt phải (42 - 47) - Xanh lá
    (48, 60, True, (0, 160, 255)),   # Viền môi ngoài (48 - 59) - Cam
    (60, 68, True, (0, 70, 255))     # Viền môi trong (60 - 67) - Đỏ
]


def render_kinematic_frame(
        pos_t: np.ndarray,
        vel_t: np.ndarray,
        acc_t: np.ndarray,
        vel_all: np.ndarray,
        acc_all: np.ndarray,
        t: int,
        T: int,
        is_missing: bool = False,
        video_id: str = "sample",
        label_str: str = "",
        label_color: Tuple[int, int, int] = (0, 255, 0),
        raw_iod: float = 0.0,
        raw_landmarks: Optional[np.ndarray] = None,
        target_w: int = 640,
        target_h: int = 480,
        zoom_scale: float = 260.0,
) -> np.ndarray:
    """
    Dựng khung hình trực quan hóa mô phỏng cho lớp KinematicFeatureExtractor:
      - Kích thước: 1280 x 720 pixel (chuẩn HD 720p, nền tối than chì sang trọng).
      - Panel trái (720x672): Hệ tọa độ Cục bộ Tâm Mũi (0, 0), tỷ lệ phóng to zoom_scale,
        chỉ vẽ các điểm mốc và đoạn thẳng nằm trong vùng nhìn (Viewport), xén viền tự nhiên qua cv2.clipLine,
        không dính mép viền, Quiver Velocity Arrows, và Vết quỹ đạo rơi của đầu (Motion Trail).
      - Panel phải (560x672):
        + Bảng chỉ số Telemetry HUD thời gian thực (Vận tốc gục đầu, Gia tốc, Tỷ lệ IOD, Số node nhìn thấy).
        + Hai đồ thị sóng thời gian thực lớn Vy(t) và Ay(t) có vạch con trỏ chạy đồng bộ theo frame t.
    """
    canvas = np.full((720, 1280, 3), (22, 26, 32), dtype=np.uint8)

    # 1. Thanh tiêu đề Header
    cv2.rectangle(canvas, (0, 0), (1280, 48), (14, 17, 22), -1)
    cv2.putText(canvas, "DRIVER GUARDIAN | KINEMATIC FEATURE EXTRACTOR SIMULATION", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    top_info = f"Clip: {video_id} | Frame {t + 1}/{T} | State: {label_str} | Zoom: {zoom_scale:.0f}px"
    cv2.putText(canvas, top_info, (740, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 48), (1280, 48), (45, 55, 70), 1)

    # Đường phân tách 2 Panel (Left 720, Right 560)
    cv2.line(canvas, (720, 48), (720, 720), (45, 55, 70), 1)

    # 2. PANEL TRÁI: Hệ tọa độ Cục bộ Tâm mũi (0,0) & IOD = 1.0 (Phóng to)
    vx1, vy1, vx2, vy2 = 15, 55, 705, 705
    viewport_rect = (vx1, vy1, vx2 - vx1, vy2 - vy1)  # (15, 55, 690, 650)
    cx, cy = 360, 380
    S = float(zoom_scale)

    cv2.putText(canvas, "LOCAL NOSE-CENTERED SPACE (0,0) & IOD = 1.0 (ZOOMED VIEW)", (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 160, 190), 1, cv2.LINE_AA)

    # Vẽ trục tọa độ chữ thập xén trong Viewport
    cv2.line(canvas, (vx1, cy), (vx2, cy), (38, 48, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, vy1), (cx, vy2), (38, 48, 60), 1, cv2.LINE_AA)

    # Vòng tròn bán kính tham chiếu IOD = 1.0 (R = S) và IOD = 0.5
    cv2.circle(canvas, (cx, cy), int(S), (32, 40, 52), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(S * 0.5), (28, 35, 45), 1, cv2.LINE_AA)
    if cx + int(S) - 65 < vx2:
        cv2.putText(canvas, "R=1.0 IOD", (cx + int(S) - 65, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 90, 110), 1)
    if cx + int(S * 0.5) - 40 < vx2:
        cv2.putText(canvas, "R=0.5", (cx + int(S * 0.5) - 40, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 80, 100), 1)

    # Tọa độ gốc tâm mũi (0,0)
    cv2.circle(canvas, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 11, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Nose (0,0)", (cx + 15, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    num_visible = 0
    if is_missing:
        cv2.putText(canvas, "[CANH BAO: MAT MAT / HEAD TURNED]", (120, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "C_kinematic = 1.0 (Fallback to d^T)", (165, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 1, cv2.LINE_AA)
    else:
        # Tính tọa độ pixel chưa xén cho tất cả 68 điểm mốc
        pts_raw = []
        for i in range(68):
            px = int(round(cx + pos_t[i, 0] * S))
            py = int(round(cy + pos_t[i, 1] * S))
            pts_raw.append((px, py))

        # 1. Vẽ các đường nối giải phẫu (Anatomical Mesh) được xén bằng cv2.clipLine
        for start, end, is_closed, color in KINEMATIC_REGIONS:
            pts_region = pts_raw[start:end]
            n_pts = len(pts_region)
            for j in range(n_pts - 1):
                ret, c1, c2 = cv2.clipLine(viewport_rect, pts_region[j], pts_region[j + 1])
                if ret:
                    cv2.line(canvas, c1, c2, color, 2, cv2.LINE_AA)
            if is_closed and n_pts > 2:
                ret, c1, c2 = cv2.clipLine(viewport_rect, pts_region[-1], pts_region[0])
                if ret:
                    cv2.line(canvas, c1, c2, color, 2, cv2.LINE_AA)

        # 2. Vẽ đường IOD giữa 2 mắt (36 <-> 45) có xén biên
        ret, c1, c2 = cv2.clipLine(viewport_rect, pts_raw[36], pts_raw[45])
        if ret:
            cv2.line(canvas, c1, c2, (255, 255, 0), 1, cv2.LINE_AA)
            iod_mid = ((c1[0] + c2[0]) // 2 - 25, (c1[1] + c2[1]) // 2 - 8)
            cv2.putText(canvas, "IOD=1.0", iod_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

        # 3. Vẽ các chấm điểm mốc: CHỈ vẽ nếu nằm hoàn toàn TRONG vùng nhìn (không dính mép)
        for i in range(68):
            px, py = pts_raw[i]
            if vx1 <= px <= vx2 and vy1 <= py <= vy2:
                cv2.circle(canvas, (px, py), 3, (0, 255, 120), -1, cv2.LINE_AA)
                num_visible += 1

        # 4. Vẽ Quiver Velocity Arrows cho các điểm mốc (chỉ vẽ từ các điểm trong Viewport)
        S_v = 40.0  # Hệ số khuếch đại vector vận tốc
        for i in range(68):
            sp = pts_raw[i]
            if vx1 <= sp[0] <= vx2 and vy1 <= sp[1] <= vy2:
                vx = vel_t[i, 0] * S_v
                vy = vel_t[i, 1] * S_v
                v_mag = float(np.hypot(vx, vy))
                if v_mag > 2.0:
                    ep = (int(round(sp[0] + vx)), int(round(sp[1] + vy)))
                    ret, c1, c2 = cv2.clipLine(viewport_rect, sp, ep)
                    if ret:
                        arr_c = (0, 0, 255) if v_mag > 15 else ((0, 165, 255) if v_mag > 7 else (0, 255, 255))
                        cv2.arrowedLine(canvas, c1, c2, arr_c, 1, tipLength=0.35, line_type=cv2.LINE_AA)

        # 5. Làm nổi bật Vector vận tốc gục đầu của Tâm Mũi (Point 30)
        nvx = float(vel_t[30, 0]) * S_v * 1.8
        nvy = float(vel_t[30, 1]) * S_v * 1.8
        n_mag = float(np.hypot(nvx, nvy))
        if n_mag > 1.5:
            nep = (int(round(cx + nvx)), int(round(cy + nvy)))
            ret, c1, c2 = cv2.clipLine(viewport_rect, (cx, cy), nep)
            if ret:
                cv2.arrowedLine(canvas, c1, c2, (0, 69, 255), 3, tipLength=0.25, line_type=cv2.LINE_AA)
                cv2.putText(canvas, f"V_head ({vel_t[30, 0]:+.2f}, {vel_t[30, 1]:+.2f})",
                            (c2[0] + 8, c2[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 69, 255), 1, cv2.LINE_AA)

        # 6. Vẽ vết quỹ đạo di chuyển của đầu (Motion Trail) qua các frame trước
        if raw_landmarks is not None and len(raw_landmarks) > 0 and raw_iod > 1.0:
            trail_pts = []
            for k in range(max(0, t - 10), t + 1):
                rdx = (raw_landmarks[k, 30, 0] - raw_landmarks[t, 30, 0]) * target_w / max(1.0, raw_iod)
                rdy = (raw_landmarks[k, 30, 1] - raw_landmarks[t, 30, 1]) * target_h / max(1.0, raw_iod)
                tx = int(round(cx + rdx * S))
                ty = int(round(cy + rdy * S))
                trail_pts.append((tx, ty))
            for k in range(len(trail_pts) - 1):
                ret, c1, c2 = cv2.clipLine(viewport_rect, trail_pts[k], trail_pts[k + 1])
                if ret:
                    alpha = int(90 + (k / max(1, len(trail_pts) - 1)) * 165)
                    cv2.line(canvas, c1, c2, (0, alpha, 255), 2, cv2.LINE_AA)

    # Khung viền mỏng quanh Viewport
    cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (45, 55, 70), 1)

    # 3. PANEL PHẢI: Telemetry HUD & Real-time Waveform Graphs
    # 3.1 Bảng Telemetry HUD
    hud_x1, hud_x2, hud_y1, hud_y2 = 740, 1260, 60, 225
    cv2.rectangle(canvas, (hud_x1, hud_y1), (hud_x2, hud_y2), (16, 20, 26), -1)
    cv2.rectangle(canvas, (hud_x1, hud_y1), (hud_x2, hud_y2), (40, 50, 65), 1)

    cv2.putText(canvas, "REAL-TIME KINEMATIC TELEMETRY", (hud_x1 + 18, hud_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1, cv2.LINE_AA)

    nvx_val = float(vel_t[30, 0])
    nvy_val = float(vel_t[30, 1])
    nv_mag_val = float(np.hypot(nvx_val, nvy_val))
    nax_val = float(acc_t[30, 0])
    nay_val = float(acc_t[30, 1])
    na_mag_val = float(np.hypot(nax_val, nay_val))

    cv2.putText(canvas, f"Head Velocity (Nose)    : |V| = {nv_mag_val:.4f}  [Vx: {nvx_val:+.4f}, Vy: {nvy_val:+.4f}]",
                (hud_x1 + 18, hud_y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Head Acceleration       : |A| = {na_mag_val:.4f}  [Ax: {nax_val:+.4f}, Ay: {nay_val:+.4f}]",
                (hud_x1 + 18, hud_y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Inter-Ocular Distance   : Raw = {raw_iod:.1f} px  ->  Normalized IOD = 1.000",
                (hud_x1 + 18, hud_y1 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Active Landmark Nodes   : {num_visible}/68 Visible in Viewport (Auto-Clipped)",
                (hud_x1 + 18, hud_y1 + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    # Đánh giá trạng thái động học
    if nvy_val > 0.04:
        motion_state = "MOTION: HEAD DROPPING DOWNWARD (NGUY CO NGU GAT)"
        motion_c = (0, 0, 255)
    elif nvy_val < -0.04:
        motion_state = "MOTION: HEAD RECOVERING UPWARD (NGANG DAU LEN)"
        motion_c = (0, 255, 255)
    else:
        motion_state = "MOTION: HEAD POSTURE STABLE (TU THE ON DINH)"
        motion_c = (0, 255, 0)
    cv2.putText(canvas, motion_state, (hud_x1 + 18, hud_y1 + 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, motion_c, 1, cv2.LINE_AA)

    # 3.2 Đồ thị sóng thời gian thực Vy(t)
    g1_x1, g1_x2, g1_y1, g1_y2 = 740, 1260, 250, 465
    cv2.rectangle(canvas, (g1_x1, g1_y1), (g1_x2, g1_y2), (12, 14, 18), -1)
    cv2.rectangle(canvas, (g1_x1, g1_y1), (g1_x2, g1_y2), (40, 50, 65), 1)
    cv2.putText(canvas, "Vertical Velocity Vy(t) [Guc dau / Roi dau]", (g1_x1 + 15, g1_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Đường trục 0
    zy1 = (g1_y1 + g1_y2) // 2
    cv2.line(canvas, (g1_x1, zy1), (g1_x2, zy1), (35, 45, 60), 1, cv2.LINE_AA)

    # Vẽ đường cong Vy(t)
    vy_max = 0.15
    pts_vy = []
    for i in range(T):
        gx = int(g1_x1 + (i / max(1, T - 1)) * (g1_x2 - g1_x1))
        val = np.clip(float(vel_all[i, 30, 1]), -vy_max, vy_max)
        gy = int(zy1 - (val / vy_max) * ((g1_y2 - g1_y1) // 2 - 15))
        pts_vy.append((gx, gy))

    for i in range(len(pts_vy) - 1):
        cv2.line(canvas, pts_vy[i], pts_vy[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
    for p in pts_vy:
        cv2.circle(canvas, p, 3, (0, 200, 200), -1, cv2.LINE_AA)

    # Vạch con trỏ đỏ tại frame t hiện tại
    cur_x_t = int(g1_x1 + (t / max(1, T - 1)) * (g1_x2 - g1_x1))
    cv2.line(canvas, (cur_x_t, g1_y1), (cur_x_t, g1_y2), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"F{t+1}", (cur_x_t - 12, g1_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1, cv2.LINE_AA)

    # 3.3 Đồ thị sóng thời gian thực Ay(t)
    g2_x1, g2_x2, g2_y1, g2_y2 = 740, 1260, 490, 705
    cv2.rectangle(canvas, (g2_x1, g2_y1), (g2_x2, g2_y2), (12, 14, 18), -1)
    cv2.rectangle(canvas, (g2_x1, g2_y1), (g2_x2, g2_y2), (40, 50, 65), 1)
    cv2.putText(canvas, "Vertical Acceleration Ay(t) [Gia toc xung luc]", (g2_x1 + 15, g2_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    zy2 = (g2_y1 + g2_y2) // 2
    cv2.line(canvas, (g2_x1, zy2), (g2_x2, zy2), (35, 45, 60), 1, cv2.LINE_AA)

    ay_max = 0.15
    pts_ay = []
    for i in range(T):
        gx = int(g2_x1 + (i / max(1, T - 1)) * (g2_x2 - g2_x1))
        val = np.clip(float(acc_all[i, 30, 1]), -ay_max, ay_max)
        gy = int(zy2 - (val / ay_max) * ((g2_y2 - g2_y1) // 2 - 15))
        pts_ay.append((gx, gy))

    for i in range(len(pts_ay) - 1):
        cv2.line(canvas, pts_ay[i], pts_ay[i + 1], (255, 100, 220), 2, cv2.LINE_AA)
    for p in pts_ay:
        cv2.circle(canvas, p, 3, (200, 80, 180), -1, cv2.LINE_AA)

    cur_x_t2 = int(g2_x1 + (t / max(1, T - 1)) * (g2_x2 - g2_x1))
    cv2.line(canvas, (cur_x_t2, g2_y1), (cur_x_t2, g2_y2), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"F{t+1}", (cur_x_t2 - 12, g2_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1, cv2.LINE_AA)

    # Hướng dẫn phím tắt ở góc dưới
    cv2.putText(canvas, "[Space]: Tam dung | [a/d]: Lui/Tien frame | [+/-]: Zoom In/Out | [0]: Reset Zoom | [r]: Phat lai | [q]: Thoat",
                (25, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 150, 170), 1, cv2.LINE_AA)

    return canvas


def visualize_kinematic_simulation(
        landmarks: torch.Tensor,
        label: Union[torch.Tensor, int] = 0,
        video_id: str = "sample",
        delay_ms: int = 100,
        window_name: str = "Driver Guardian - KinematicFeatureExtractor Simulation",
):
    """
    Hàm độc lập để hiển thị trực quan hóa mô phỏng KinematicFeatureExtractor từ tensor điểm mốc:
      - Nhận vào landmarks [T, 68, 2] hoặc [1, T, 68, 2] hoặc [T, 68, 3].
      - Không yêu cầu tệp video gốc hay ảnh khuôn mặt.
    """
    if landmarks.dim() == 4:
        landmarks = landmarks.squeeze(0)
    if landmarks.shape[-1] > 2:
        landmarks = landmarks[..., :2]

    T = landmarks.shape[0]
    lms_np = landmarks.cpu().numpy()
    lbl_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    label_str = "DROWSY (1)" if lbl_val == 1 else "ALERT (0)"
    label_color = (0, 0, 255) if lbl_val == 1 else (0, 255, 0)

    from model import KinematicFeatureExtractor
    kfe = KinematicFeatureExtractor()
    lms_in = landmarks.unsqueeze(0).permute(0, 3, 1, 2).float()
    kfe_out = kfe(lms_in)
    pos_norm_all = kfe_out[0, :2].permute(1, 2, 0).cpu().numpy()
    vel_norm_all = kfe_out[0, 2:4].permute(1, 2, 0).cpu().numpy()
    acc_norm_all = kfe_out[0, 4:6].permute(1, 2, 0).cpu().numpy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    t = 0
    paused = False
    zoom_scale = 260.0
    while True:
        is_missing = np.allclose(lms_np[t], 1.0)
        p_left = lms_np[t, 36, :2] * np.array([640, 480])
        p_right = lms_np[t, 45, :2] * np.array([640, 480])
        raw_iod = float(np.linalg.norm(p_right - p_left)) if not is_missing else 0.0

        canvas = render_kinematic_frame(
            pos_t=pos_norm_all[t],
            vel_t=vel_norm_all[t],
            acc_t=acc_norm_all[t],
            vel_all=vel_norm_all,
            acc_all=acc_norm_all,
            t=t,
            T=T,
            is_missing=is_missing,
            video_id=video_id,
            label_str=label_str,
            label_color=label_color,
            raw_iod=raw_iod,
            raw_landmarks=lms_np,
            target_w=640,
            target_h=480,
            zoom_scale=zoom_scale,
        )
        cv2.imshow(window_name, canvas)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        wait_time = 0 if paused else delay_ms
        key = cv2.waitKey(wait_time) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:
            paused = not paused
        elif key in (ord('r'), ord('R')):
            t = 0
            continue
        elif key in (ord('d'), ord('D'), 83):
            t = (t + 1) % T
            continue
        elif key in (ord('a'), ord('A'), 81):
            t = (t - 1 + T) % T
            continue
        elif key in (ord('+'), ord('=')):
            zoom_scale = min(600.0, zoom_scale + 30.0)
            continue
        elif key in (ord('-'), ord('_')):
            zoom_scale = max(100.0, zoom_scale - 30.0)
            continue
        elif key == ord('0'):
            zoom_scale = 260.0
            continue

        if not paused:
            t = (t + 1) % T

    cv2.destroyAllWindows()


def visualize_litefat_sample(
        face_frames: Optional[torch.Tensor],
        landmarks: torch.Tensor,
        label: Union[torch.Tensor, int],
        raw_frames: List[np.ndarray],
        meta: Optional[Dict[str, Any]] = None,
        delay_ms: int = 120,
        window_name: str = "Driver Guardian - LiteFat Video Preview",
        grid_window_name: str = "Driver Guardian - All Cropped Face Frames (Grid)",
        kinematic_window_name: str = "Driver Guardian - KinematicFeatureExtractor Simulation",
        show_kinematic: bool = True,
):
    """
    Hiển thị trực quan toàn diện mẫu dữ liệu của LiteFat:
      1. Cửa sổ 1: Video phát động (có thể lặp lại và tạm dừng) hiển thị song song:
         - Bên trái: Khung hình letterbox gốc (640x480) được vẽ 68 điểm mốc giải phẫu
           (đường nối viền mắt, sống mũi, viền môi trong/ngoài) + Bounding Box khuôn mặt tài xế + CropBox viền vàng.
         - Bên phải: Khung hình khuôn mặt đã được cắt và chuẩn hóa (224x224) cho MobileNetV3-Small (nếu có).
      2. Cửa sổ 2: Bảng lưới (Montage Grid 4x4) hiển thị đồng thời toàn bộ T khung hình khuôn mặt
         đã cắt trong chuỗi thời gian để dễ dàng soi xét từng vi biểu cảm (nếu có face_frames).
      3. Cửa sổ 3: Mô phỏng không gian chuẩn hóa và trường vận tốc/gia tốc động học của lớp
         KinematicFeatureExtractor (Tâm mũi, IOD=1.0, Quiver Arrows, Waveform Graphs).
    """
    T = len(raw_frames)
    lbl_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    label_str = "DROWSY (1)" if lbl_val == 1 else "ALERT (0)"
    label_color = (0, 0, 255) if lbl_val == 1 else (0, 255, 0)
    video_id = meta.get("video_id", "sample") if meta else "sample"
    bboxes_list = meta.get("bboxes", [None] * T) if meta else [None] * T
    crop_boxes_list = meta.get("crop_boxes", None) if meta else None

    # 1. Giải chuẩn hóa ImageNet trên face_frames (nếu có) để hiển thị BGR [0, 255]
    has_faces = face_frames is not None
    faces_bgr = []
    if has_faces:
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        unnorm_faces = (face_frames * norm_std + norm_mean).clamp(0.0, 1.0)
        faces_rgb = (unnorm_faces.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        faces_bgr = [cv2.cvtColor(faces_rgb[i], cv2.COLOR_RGB2BGR) for i in range(T)]

        # 2. Tạo ảnh bảng lưới (Montage Grid) hiển thị tất cả T khung hình khuôn mặt đã cắt
        cols = 4 if T >= 4 else T
        rows = (T + cols - 1) // cols
        cell_w, cell_h = 160, 160
        grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i in range(T):
            r = i // cols
            c = i % cols
            cell = cv2.resize(faces_bgr[i], (cell_w, cell_h))
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (50, 50, 50), 1)
            cv2.putText(cell, f"F{i:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cell, f"F{i:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            grid_img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

        cv2.namedWindow(grid_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(grid_window_name, grid_img)
        cv2.resizeWindow(grid_window_name, cols * cell_w, rows * cell_h)

    # 3. Chuẩn bị trích xuất đặc trưng động học KinematicFeatureExtractor (nếu show_kinematic=True)
    has_kinematics = False
    pos_norm_all, vel_norm_all, acc_norm_all = None, None, None
    if show_kinematic and landmarks is not None and len(landmarks) > 0:
        try:
            from model import KinematicFeatureExtractor
            kfe = KinematicFeatureExtractor()
            lms_in = landmarks.unsqueeze(0).permute(0, 3, 1, 2).float()
            kfe_out = kfe(lms_in)
            pos_norm_all = kfe_out[0, :2].permute(1, 2, 0).cpu().numpy()
            vel_norm_all = kfe_out[0, 2:4].permute(1, 2, 0).cpu().numpy()
            acc_norm_all = kfe_out[0, 4:6].permute(1, 2, 0).cpu().numpy()
            has_kinematics = True
            cv2.namedWindow(kinematic_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(kinematic_window_name, 1280, 720)
        except Exception as e:
            logger.warning(f"Không thể khởi tạo KinematicFeatureExtractor cho trực quan hóa: {e}")

    # 4. Chuẩn bị cửa sổ phát video side-by-side
    lms_np = landmarks.cpu().numpy()  # [T, 68, 2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[Preview] Đang phát mẫu video: {video_id} ({T} frames | Trạng thái: {label_str})")
    print(">>> Phím điều khiển: [Space] Tạm dừng/Tiếp tục | [a/d] Lùi/Tiến frame | [+/-] Zoom In/Out | [0] Reset Zoom | [r] Phát lại | [q/ESC] Thoát <<<")

    t = 0
    paused = False
    zoom_scale = 260.0

    while True:
        left_img = raw_frames[t].copy()
        target_h, target_w = left_img.shape[:2]

        # Lấy BBox và Landmarks của frame t
        bbox = bboxes_list[t] if t < len(bboxes_list) else None
        is_missing = np.allclose(lms_np[t], 1.0)

        if not is_missing:
            pts_px = lms_np[t, :, :2].copy()
            pts_px[:, 0] *= float(target_w)
            pts_px[:, 1] *= float(target_h)

            bbox_arr = None
            if bbox is not None:
                if isinstance(bbox, torch.Tensor):
                    bbox_arr = bbox.detach().cpu().numpy()
                else:
                    bbox_arr = np.asarray(bbox, dtype=np.float32)

            bbox_score = float(bbox_arr[4]) if (bbox_arr is not None and len(bbox_arr) >= 5) else 1.0
            if bbox_arr is not None and len(bbox_arr) >= 5 and float(bbox_arr[4]) > 0:
                bboxes_to_draw = np.array([bbox_arr[:5]], dtype=np.float32)
            else:
                # Tự tính BBox bao quanh điểm mốc nếu bbox gốc bị thiếu
                lx1, ly1 = np.min(pts_px[:, 0]), np.min(pts_px[:, 1])
                lx2, ly2 = np.max(pts_px[:, 0]), np.max(pts_px[:, 1])
                bboxes_to_draw = np.array([[lx1, ly1, lx2, ly2, bbox_score]], dtype=np.float32)

            draw_68_landmarks_custom(
                left_img,
                np.expand_dims(pts_px, 0),
                bboxes_to_draw,
                conf_threshold=0.1
            )
        else:
            cv2.putText(left_img, "[CANH BAO: MAT MAT / HEAD TURNED]", (30, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # Vẽ viền crop_box nếu có
        if crop_boxes_list is not None and t < len(crop_boxes_list):
            cb = crop_boxes_list[t]
            if isinstance(cb, torch.Tensor):
                cb = cb.detach().cpu().numpy()
            if len(cb) >= 4:
                cx1, cy1, cx2, cy2 = int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])
                cv2.rectangle(left_img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1, cv2.LINE_AA)

        if has_faces:
            right_panel = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            header_h = 45
            face_display_size = target_h - header_h
            face_resized = cv2.resize(faces_bgr[t], (face_display_size, face_display_size),
                                      interpolation=cv2.INTER_LINEAR)

            x_offset = (target_h - face_display_size) // 2
            right_panel[header_h:header_h + face_display_size, x_offset:x_offset + face_display_size] = face_resized

            cv2.putText(right_panel, f"CROPPED FACE (224x224) - F{t:02d}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(right_panel, (x_offset, header_h),
                          (x_offset + face_display_size - 1, header_h + face_display_size - 1), (0, 255, 0), 2)

            combined = np.hstack([left_img, right_panel])
        else:
            combined = left_img

        top_bar = f"Video: {video_id} | Status: {label_str} | Frame {t + 1}/{T}"
        if paused:
            top_bar += "  [DANG TAM DUNG]"
        cv2.putText(combined, top_bar, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, top_bar, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)

        guide_bar = "[Space]: Pause/Resume | [a/d]: Prev/Next | [+/-]: Zoom | [0]: Reset Zoom | [r]: Replay | [q]: Thoat"
        cv2.putText(combined, guide_bar, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, guide_bar, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window_name, combined)

        # Hiển thị cửa sổ mô phỏng KinematicFeatureExtractor (nếu show_kinematic=True)
        if has_kinematics and pos_norm_all is not None:
            raw_iod_val = 0.0
            if not is_missing:
                p_left = lms_np[t, 36, :2] * np.array([target_w, target_h])
                p_right = lms_np[t, 45, :2] * np.array([target_w, target_h])
                raw_iod_val = float(np.linalg.norm(p_right - p_left))

            kin_canvas = render_kinematic_frame(
                pos_t=pos_norm_all[t],
                vel_t=vel_norm_all[t],
                acc_t=acc_norm_all[t],
                vel_all=vel_norm_all,
                acc_all=acc_norm_all,
                t=t,
                T=T,
                is_missing=is_missing,
                video_id=video_id,
                label_str=label_str,
                label_color=label_color,
                raw_iod=raw_iod_val,
                raw_landmarks=lms_np,
                target_w=target_w,
                target_h=target_h,
                zoom_scale=zoom_scale,
            )
            cv2.imshow(kinematic_window_name, kin_canvas)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or (
                has_kinematics and cv2.getWindowProperty(kinematic_window_name, cv2.WND_PROP_VISIBLE) < 1
        ):
            break

        wait_time = 0 if paused else delay_ms
        key = cv2.waitKey(wait_time) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            print("Đã đóng cửa sổ hiển thị.")
            break
        elif key == 32:  # Phím Space
            paused = not paused
        elif key in (ord('r'), ord('R')):
            t = 0
            continue
        elif key in (ord('d'), ord('D'), 83):
            t = (t + 1) % T
            continue
        elif key in (ord('a'), ord('A'), 81):
            t = (t - 1 + T) % T
            continue
        elif key in (ord('+'), ord('=')):
            zoom_scale = min(600.0, zoom_scale + 30.0)
            continue
        elif key in (ord('-'), ord('_')):
            zoom_scale = max(100.0, zoom_scale - 30.0)
            continue
        elif key == ord('0'):
            zoom_scale = 260.0
            continue

        if not paused:
            t = (t + 1) % T

    cv2.destroyAllWindows()


# =====================================================================
# 6. KIỂM THỬ TÍCH HỢP & HIỂN THỊ
# =====================================================================
if __name__ == "__main__":
    DATASET_DIR = r"E:\LSTM\data_processed"
    print("=" * 70)
    print("KIỂM THỬ LITEFAT DATASET (TỐI ƯU HÓA GPU BATCHED)")
    print("=" * 70)

    if not Path(DATASET_DIR).is_dir():
        print(f"[ERROR] Không tìm thấy thư mục {DATASET_DIR} trên máy này.")
        sys.exit(1)

    # 1. Khởi tạo Dataset theo cửa sổ window_sec = 1.6s và time_interval = 0.1s (tương đương 16 frames cho LiteFat)
    print("\n[1] Đang nạp LiteFatDataset (split='test', window_sec=1.6, time_interval=0.1)...")
    ds = LiteFatDataset(
        dataset_dir=DATASET_DIR,
        use_cache=True,
        cache_dir=r"E:\LSTM\LiteFat\cache_pt",
        split="test",
        time_interval=0.05,
        window_sec=None,
        return_meta=True,
        return_raw_frames=True,
    )
    print(f"Tổng số mẫu kiểm định (test samples): {len(ds)}")

    if len(ds) > 0:
        # Lấy mẫu kiểm định
        sample_idx = min(10, len(ds) - 1)
        face_frames, landmarks, label, meta, raw_frames = ds[sample_idx]
        T_actual = face_frames.shape[0]
        print(f"\n--- KẾT QUẢ TRÍCH XUẤT MẪU {sample_idx} ---")
        print(f"Video ID            : {meta['video_id']}")
        print(f"Đường dẫn video     : {meta['path']}")
        print(f"FPS thực tế         : {meta['fps']:.2f}")
        print(f"Tỷ lệ frame có mặt  : {meta['found_faces_ratio'] * 100:.1f}%")
        print(f"Kích thước face_frames: {face_frames.shape} (T={T_actual}, C=3, H=224, W=224)")
        print(f"Kích thước landmarks  : {landmarks.shape}   (T={T_actual}, V=68, C=2)")
        print(f"Nhãn mẫu (Label)      : {label.item()} ({'Drowsy' if label.item() == 1 else 'Alert'})")
        print(">>> Xác nhận hình dạng Tensor đầu vào hoàn toàn chính xác! <<<")

        # 2. Kiểm thử truyền batch qua DataLoader & Model LiteFat
        print("\n[2] Kiểm thử chuyển tiếp (forward pass) qua mô hình LiteFat...")
        from model import LiteFat
        from loss import LiteFatLoss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LiteFat(num_nodes=68, feature_dim=64, num_classes=2, pretrained_backbone=False).to(device)
        criterion = LiteFatLoss()

        batch_frames = face_frames.unsqueeze(0).to(device)  # [1, T, 3, 224, 224]
        batch_landmarks = landmarks.unsqueeze(0).to(device)  # [1, T, 68, 2]
        batch_labels = label.unsqueeze(0).to(device)  # [1]

        model.eval()
        with torch.no_grad():
            frame_logits, clip_logits = model(batch_frames, batch_landmarks)
            loss_val = criterion(frame_logits, batch_labels)

        print(f"Frame logits shape : {frame_logits.shape} (Kỳ vọng: [1, {T_actual}, 2])")
        print(f"Clip logits shape  : {clip_logits.shape}  (Kỳ vọng: [1, 2])")
        print(f"Loss value         : {loss_val.item():.4f}")
        print(">>> Mô hình LiteFat xử lý trơn tru dữ liệu từ LiteFatDataset! <<<")

        # 3. Hiển thị trực quan mẫu nếu người dùng chạy tương tác (bỏ qua nếu đặt biến HEADLESS=1)
        if not os.environ.get("HEADLESS"):
            print("\n[3] Hiển thị video preview (nhấn 'q' để bỏ qua hoặc đóng)...")
            try:
                visualize_litefat_sample(face_frames, landmarks, label, raw_frames, meta=meta, delay_ms=80)
            except Exception as e:
                print(f"[INFO] Không thể mở GUI hiển thị ({e}). Bỏ qua bước trực quan hóa.")
        else:
            print("\n[3] Biến môi trường HEADLESS=1 được bật. Bỏ qua bước mở cửa sổ OpenCV GUI.")
