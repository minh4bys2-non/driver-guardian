# =====================================================================
# File: dataset.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, 10 FPS)
# Tác vụ: Nạp tập dữ liệu dạng E:\LSTM\data_processed_v1 từ manifest
#         (.csv hoặc .json), trích xuất 68 điểm mốc khuôn mặt bằng
#         PIPNet (FaceBoxesV2 + ResNet18 300W) và cắt vùng khuôn mặt
#         cho mô hình LiteFat_v1 (cửa sổ 60s @ 10 FPS, sliding 30s).
# =====================================================================
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
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
import torchvision
from torch.utils.data import DataLoader, Dataset

# Import các hàm tiện ích dùng chung từ ultis.py
from ultis import (
    _parse_size,
    atomic_save,
    augment_multimodal_clip,
    compute_iou,
    crop_faces_from_boxes,
    letterbox,
)

logger = logging.getLogger("DriverGuardian.LiteFat_v1")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# 0. HẰNG SỐ VÙNG GIẢI PHẪU & TIỆN ÍCH PIPNET 68 ĐIỂM
#    (Các hàm _parse_size, letterbox, compute_iou, crop_faces_from_boxes,
#     augment_multimodal_clip, atomic_save & FLIP_MAP_68 tập trung ở ultis.py)
# =====================================================================
KINEMATIC_REGIONS = [
    (0, 17, False, (255, 190, 0)),   # Đường viền hàm (0 - 16)
    (17, 22, False, (0, 230, 255)),  # Lông mày trái (17 - 21)
    (22, 27, False, (0, 230, 255)),  # Lông mày phải (22 - 26)
    (27, 31, False, (220, 0, 255)),  # Sống mũi (27 - 30)
    (31, 36, False, (220, 0, 255)),  # Chóp mũi dưới (31 - 35)
    (36, 42, True, (0, 255, 100)),   # Mắt trái (36 - 41)
    (42, 48, True, (0, 255, 100)),   # Mắt phải (42 - 47)
    (48, 60, True, (0, 160, 255)),   # Viền môi ngoài (48 - 59)
    (60, 68, True, (0, 70, 255)),    # Viền môi trong (60 - 67)
]


def init_pipnet_68_gpu(backbone: str = "resnet18") -> None:
    """Khởi tạo pipeline FaceBoxesV2 + PIPNet (68 điểm chuẩn 300W) qua torchlm."""
    import torchlm
    from torchlm.models import pipnet
    from torchlm.tools import faceboxesv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Đang khởi tạo PIPNet ({backbone} - 68 điểm 300W) trên thiết bị: {device.upper()}")
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone=backbone,
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            map_location=device,
        )
    )


def draw_68_landmarks_custom(
        frame: np.ndarray,
        landmarks: np.ndarray,
        bboxes: np.ndarray,
        conf_threshold: float = 0.6,
) -> None:
    """Vẽ Bounding Box và 68 điểm mốc theo từng vùng giải phẫu (chuẩn 300W / iBUG 68)."""
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Face: {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        pts = landmarks[i].astype(np.int32)
        for start, end, is_closed, color in KINEMATIC_REGIONS:
            cv2.polylines(
                frame,
                [pts[start:end]],
                isClosed=is_closed,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        for (x, y) in pts:
            cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)


# =====================================================================
# 1. BỘ TRÍCH XUẤT PIPNET TỐI ƯU HÓA CAO CẤP (FAST PIPNET EXTRACTOR)
# =====================================================================
class FastPIPNetExtractor:
    """
    Bộ trích xuất 68 điểm mốc khuôn mặt và cắt khung hình cho LiteFat_v1:
      1. Batched FaceBoxesV2 trên GPU với cached PriorBox anchors và torchvision GPU NMS.
      2. Temporal Driver Tracking: Chọn khuôn mặt tài xế ổn định qua score, diện tích & IoU.
      3. Batched PIPNet 300W trên GPU với giải mã NRM vector hóa.
      4. Landmark-Guided Box Refinement + nội suy tuyến tính + làm mượt EMA.
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
        self.letterbox_size = _parse_size(letterbox_size)
        self.face_size = _parse_size(face_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.bbox_margin = float(bbox_margin)
        self.box_smooth_alpha = float(box_smooth_alpha)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")

        from torchlm.runtime._wrappers import RuntimeWrapper
        if getattr(RuntimeWrapper, "face_base", None) is None or getattr(RuntimeWrapper, "landmarks_base", None) is None:
            init_pipnet_68_gpu(backbone=backbone)

        self.face_base = RuntimeWrapper.face_base
        self.face_net = self.face_base.net.to(self.device)
        self.face_net.eval()

        self.lmk_base = RuntimeWrapper.landmarks_base
        self.lmk_net = self.lmk_base.to(self.device)
        self.lmk_net.eval()

        target_w, target_h = self.letterbox_size
        from torchlm.tools._faceboxesv2 import PriorBox
        pbox = PriorBox(self.face_base.cfg, image_size=(target_h, target_w))
        self.cached_priors = pbox.forward().to(self.device)
        self.cached_variances = self.face_base.cfg["variance"]
        self.cached_scale = torch.tensor(
            [target_w, target_h, target_w, target_h], dtype=torch.float32, device=self.device
        )
        self.faceboxes_mean = torch.tensor(
            [104.0, 117.0, 123.0], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        self.pipnet_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.pipnet_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        rev1 = self.lmk_base.reverse_index1
        rev2 = self.lmk_base.reverse_index2
        num_nb = self.lmk_base.num_nb
        self.rev_indices = torch.tensor(
            [r1 * num_nb + r2 for r1, r2 in zip(rev1, rev2)],
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def _detect_faces_batched(self, frames: List[np.ndarray], max_chunk: int = 32) -> List[np.ndarray]:
        """Chạy phát hiện khuôn mặt theo lô T khung hình trên GPU."""
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

            frames_np = np.stack(chunk, axis=0)
            tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(self.device).float()
            tensor = tensor - self.faceboxes_mean

            loc, conf = self.face_net(tensor)
            scores = conf[:, :, 1]

            boxes_xy = priors_xy + loc[:, :, :2] * v0 * priors_wh
            boxes_wh = priors_wh * torch.exp(loc[:, :, 2:] * v1)
            b_x1y1 = boxes_xy - boxes_wh / 2.0
            b_x2y2 = b_x1y1 + boxes_wh
            decoded_boxes = torch.cat([b_x1y1, b_x2y2], dim=-1) * self.cached_scale

            for b in range(cur_T):
                sc = scores[b]
                mask = sc >= self.conf_threshold
                if not mask.any():
                    all_dets.append(np.empty((0, 5), dtype=np.float32))
                    continue

                bx = decoded_boxes[b, mask]
                sc_valid = sc[mask]
                keep = torchvision.ops.nms(bx, sc_valid, iou_threshold=self.nms_threshold)[:50]
                res = torch.cat([bx[keep], sc_valid[keep, None]], dim=-1).cpu().numpy()
                all_dets.append(res)

        return all_dets

    def _select_driver_faces(
            self,
            all_dets: List[np.ndarray],
    ) -> Tuple[List[Optional[np.ndarray]], List[bool]]:
        """Theo dõi và chọn duy nhất 1 khuôn mặt tài xế ổn định qua các khung hình."""
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
                best_score = -1.0
                for i, d in enumerate(dets):
                    area = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
                    metric = d[4] * (area ** 0.5)
                    if metric > best_score:
                        best_score = metric
                        best_idx = i
            else:
                best_match = -1.0
                for i, d in enumerate(dets):
                    iou = compute_iou(d[:4], prev_box[:4])
                    area = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
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
        """Trích xuất 68 điểm mốc khuôn mặt chuẩn 300W bằng PIPNet theo lô trên GPU."""
        T = len(frames)
        target_w, target_h = self.letterbox_size
        extend = 0.2

        valid_indices = [t for t in range(T) if found[t] and chosen_bboxes[t] is not None]
        landmarks_norm = np.ones((T, 68, 2), dtype=np.float32)
        landmarks_px = np.zeros((T, 68, 2), dtype=np.float32)

        if not valid_indices:
            return landmarks_norm, landmarks_px

        crop_coords = []
        crops_resized = []

        for idx in valid_indices:
            bx1, by1, bx2, by2 = chosen_bboxes[idx][:4]
            bw = bx2 - bx1 + 1.0
            bh = by2 - by1 + 1.0

            cx1 = int(bx1 - bw * extend / 2.0)
            cy1 = int(by1 + bh * extend / 2.0)
            cx2 = int(bx2 + bw * extend / 2.0)
            cy2 = int(by2 + bh * extend / 2.0)

            cx1 = max(0, min(target_w - 2, cx1))
            cy1 = max(0, min(target_h - 2, cy1))
            cx2 = max(cx1 + 1, min(target_w, cx2))
            cy2 = max(cy1 + 1, min(target_h, cy2))

            crop_coords.append((cx1, cy1, float(cx2 - cx1), float(cy2 - cy1)))
            face_crop = frames[idx][cy1:cy2, cx1:cx2, ::-1]
            crops_resized.append(cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LINEAR))

        max_crop_chunk = 64
        batched_lms_list = []
        norm_factor = 1.0 * self.lmk_base.input_size / self.lmk_base.net_stride
        num_nb = self.lmk_base.num_nb
        max_len = self.lmk_base.max_len

        for s in range(0, len(crops_resized), max_crop_chunk):
            crops_np = np.stack(crops_resized[s:s + max_crop_chunk], axis=0)
            crops_tensor = torch.from_numpy(crops_np).permute(0, 3, 1, 2).to(self.device).float() / 255.0
            crops_tensor = (crops_tensor - self.pipnet_mean) / self.pipnet_std

            out_cls, out_x, out_y, out_nb_x, out_nb_y = self.lmk_net(crops_tensor)
            cur_B, num_lms, _, W_map = out_cls.shape

            out_cls = out_cls.view(cur_B, num_lms, -1)
            max_ids = torch.argmax(out_cls, dim=2)
            max_ids_col = max_ids.unsqueeze(-1)

            x_sel = torch.gather(out_x.view(cur_B, num_lms, -1), 2, max_ids_col).squeeze(2)
            y_sel = torch.gather(out_y.view(cur_B, num_lms, -1), 2, max_ids_col).squeeze(2)

            max_ids_nb = max_ids_col.repeat(1, 1, num_nb).view(cur_B, num_lms * num_nb, 1)
            nb_x_sel = torch.gather(out_nb_x.view(cur_B, num_lms * num_nb, -1), 2, max_ids_nb).view(cur_B, num_lms, num_nb)
            nb_y_sel = torch.gather(out_nb_y.view(cur_B, num_lms * num_nb, -1), 2, max_ids_nb).view(cur_B, num_lms, num_nb)

            pred_x = ((max_ids % W_map).float() + x_sel) / norm_factor
            pred_y = (torch.floor(max_ids.float() / W_map) + y_sel) / norm_factor

            pred_nb_x = ((max_ids % W_map).float().unsqueeze(-1) + nb_x_sel) / norm_factor
            pred_nb_y = (torch.floor(max_ids.float() / W_map).unsqueeze(-1) + nb_y_sel) / norm_factor

            nb_x_rev = pred_nb_x.view(cur_B, -1)[:, self.rev_indices].view(cur_B, num_lms, max_len)
            nb_y_rev = pred_nb_y.view(cur_B, -1)[:, self.rev_indices].view(cur_B, num_lms, max_len)

            final_x = torch.mean(torch.cat([pred_x.unsqueeze(-1), nb_x_rev], dim=-1), dim=-1)
            final_y = torch.mean(torch.cat([pred_y.unsqueeze(-1), nb_y_rev], dim=-1), dim=-1)
            batched_lms_list.append(torch.stack([final_x, final_y], dim=-1).cpu().numpy())

        batched_lms = (
            np.concatenate(batched_lms_list, axis=0)
            if batched_lms_list
            else np.empty((0, 68, 2), dtype=np.float32)
        )

        for i, idx in enumerate(valid_indices):
            cx1, cy1, cw, ch = crop_coords[i]
            lm_crop = batched_lms[i]
            abs_x = lm_crop[:, 0] * cw + cx1
            abs_y = lm_crop[:, 1] * ch + cy1
            landmarks_px[idx, :, 0] = abs_x
            landmarks_px[idx, :, 1] = abs_y
            landmarks_norm[idx, :, 0] = abs_x / float(target_w)
            landmarks_norm[idx, :, 1] = abs_y / float(target_h)

        return landmarks_norm, landmarks_px

    def compute_face_crop_boxes(
            self,
            chosen_bboxes: List[Optional[np.ndarray]],
            landmarks_px: np.ndarray,
            found: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán hộp cắt khuôn mặt vuông kèm nội suy mất mặt và làm mượt EMA."""
        T = len(chosen_bboxes)
        target_w, target_h = self.letterbox_size
        raw_boxes = np.zeros((T, 4), dtype=np.float32)

        for t in range(T):
            if found[t] and chosen_bboxes[t] is not None:
                bx1, by1, bx2, by2 = chosen_bboxes[t][:4]
                lms = landmarks_px[t]
                fx1 = min(bx1, float(np.min(lms[:, 0])))
                fy1 = min(by1, float(np.min(lms[:, 1])))
                fx2 = max(bx2, float(np.max(lms[:, 0])))
                fy2 = max(by2, float(np.max(lms[:, 1])))

                fw, fh = fx2 - fx1, fy2 - fy1
                cx1, cy1 = fx1 - fw * self.bbox_margin, fy1 - fh * self.bbox_margin
                cx2, cy2 = fx2 + fw * self.bbox_margin, fy2 + fh * self.bbox_margin

                center_x, center_y = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
                side = max(cx2 - cx1, cy2 - cy1)
                raw_boxes[t] = [
                    center_x - side / 2.0,
                    center_y - side / 2.0,
                    center_x + side / 2.0,
                    center_y + side / 2.0,
                ]

        valid_indices = [t for t in range(T) if found[t]]
        if not valid_indices:
            side = min(target_w, target_h) * 0.8
            cx, cy = target_w / 2.0, target_h / 2.0
            default_box = np.array([cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0], dtype=np.float32)
            raw_boxes[:] = default_box
        else:
            first_idx, last_idx = valid_indices[0], valid_indices[-1]
            raw_boxes[:first_idx] = raw_boxes[first_idx]
            raw_boxes[last_idx + 1:] = raw_boxes[last_idx]
            for i in range(len(valid_indices) - 1):
                idx_a, idx_b = valid_indices[i], valid_indices[i + 1]
                if idx_b - idx_a > 1:
                    box_a, box_b = raw_boxes[idx_a], raw_boxes[idx_b]
                    for t in range(idx_a + 1, idx_b):
                        alpha = float(t - idx_a) / float(idx_b - idx_a)
                        raw_boxes[t] = (1.0 - alpha) * box_a + alpha * box_b

        smoothed_boxes = np.zeros_like(raw_boxes)
        smoothed_boxes[0] = raw_boxes[0]
        alpha = self.box_smooth_alpha
        for t in range(1, T):
            smoothed_boxes[t] = alpha * raw_boxes[t] + (1.0 - alpha) * smoothed_boxes[t - 1]

        crop_boxes = np.zeros((T, 4), dtype=np.int32)
        for t in range(T):
            bx1, by1, bx2, by2 = smoothed_boxes[t]
            x1 = max(0, min(target_w - 2, int(round(bx1))))
            y1 = max(0, min(target_h - 2, int(round(by1))))
            x2 = max(x1 + 1, min(target_w, int(round(bx2))))
            y2 = max(y1 + 1, min(target_h, int(round(by2))))
            crop_boxes[t] = [x1, y1, x2, y2]

        return crop_boxes, smoothed_boxes

    def extract_from_frames(
            self,
            frames: List[np.ndarray],
            crop_faces: bool = True,
            return_crop_boxes: bool = False,
            visual_frame_stride: int = 1,
    ) -> Union[
        Tuple[Optional[torch.Tensor], torch.Tensor, List[Optional[np.ndarray]], List[bool]],
        Tuple[Optional[torch.Tensor], torch.Tensor, List[Optional[np.ndarray]], List[bool], np.ndarray],
    ]:
        """Pipeline trích xuất 68 điểm mốc + ảnh cắt khuôn mặt từ chuỗi khung hình BGR."""
        target_w, target_h = self.letterbox_size
        uniform_frames = [
            letterbox(f, new_size=(target_w, target_h)) if (f.shape[1] != target_w or f.shape[0] != target_h) else f
            for f in frames
        ]

        all_dets = self._detect_faces_batched(uniform_frames)
        chosen_bboxes, found = self._select_driver_faces(all_dets)
        landmarks_norm, landmarks_px = self._extract_landmarks_batched(uniform_frames, chosen_bboxes, found)
        crop_boxes, _ = self.compute_face_crop_boxes(chosen_bboxes, landmarks_px, found)

        face_frames_tensor = None
        if crop_faces:
            v_stride = max(1, int(visual_frame_stride))
            if v_stride > 1:
                face_frames_tensor = crop_faces_from_boxes(
                    uniform_frames[::v_stride], crop_boxes[::v_stride], face_size=self.face_size
                )
            else:
                face_frames_tensor = crop_faces_from_boxes(uniform_frames, crop_boxes, face_size=self.face_size)

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
        letterbox_size: Tuple[int, int] = (640, 640),
) -> Tuple[List[np.ndarray], float, int]:
    """Đọc chính xác các chỉ số `target_indices` từ video bằng seek + cap.grab() + letterbox."""
    if len(target_indices) == 0:
        target_indices = np.array([0], dtype=int)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Không thể mở tệp video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_w, target_h = _parse_size(letterbox_size)

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

            frames.append(letterbox(frame, new_size=(target_w, target_h)))
            curr_frame += 1
    finally:
        cap.release()

    if not frames:
        raise IOError(f"Không đọc được khung hình nào từ: {video_path}")

    while len(frames) < len(target_indices):
        frames.append(frames[-1].copy())

    return frames, fps, total_frames


# =====================================================================
# 3. LỚP LITEFAT DATASET (NẠP MANIFEST .CSV / .JSON CHO KSS 1-9 @ 10 FPS)
# =====================================================================
def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or str(val).strip() == "":
        return default
    return int(round(float(val)))


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip() == "":
        return default
    return float(val)


class LiteFatDataset(Dataset):
    """
    PyTorch Dataset nạp dữ liệu KSS (thang điểm 1-9) cho LiteFat_v1 từ cấu trúc
    như E:\\LSTM\\data_processed_v1:
      - Chỉ đọc đường dẫn file và metadata mẫu từ file manifest (.csv hoặc .json)
        trong thư mục dataset (không quét thư mục video trực tiếp).
      - Mặc định lấy mẫu ở 10 FPS (`time_interval=0.1`) với toàn bộ chiều dài video
        (`window_sec=None`) để hỗ trợ chạy Sliding Window 60s (stride 30s) tích lũy đặc trưng.
      - Trả về `label` kiểu `torch.long` (giá trị nguyên 1..9 từ cột `label`/`kss_score`)
        và lưu kèm `kss_raw`, `kss_class`, `sample_weight` trong `meta`.
    """

    _SHARED_CACHE: Dict[str, Dict[str, Any]] = {}
    _SHARED_META: Dict[str, Dict[str, Any]] = {}

    def __init__(
            self,
            dataset_dir: Union[str, Path] = r"E:\LSTM\data_processed_v1",
            split: str = "train",
            time_interval: float = 0.1,
            window_sec: Optional[float] = None,
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
            eval_window_mode: str = "center",
            drop_single_class_subjects: bool = False,
            visual_frame_stride: int = 1,
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
        self.manifest_filename = str(manifest_filename)
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
        self.eval_window_mode = str(eval_window_mode).lower().strip()
        self.drop_single_class_subjects = bool(drop_single_class_subjects)
        self.visual_frame_stride = max(1, int(visual_frame_stride))

        if self.cache_dir is not None:
            self.crop_cache_dir = self.cache_dir / f"face_crops_stride{self.visual_frame_stride}"
            try:
                self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        else:
            self.crop_cache_dir = None

        self.use_visual_context = bool(use_visual_context)
        self.extract_crops_only = bool(extract_crops_only) or (not self.use_visual_context)

        if selected_datasets is not None:
            if isinstance(selected_datasets, str):
                self.selected_datasets = [selected_datasets.lower()]
            else:
                self.selected_datasets = [str(s).lower() for s in selected_datasets if str(s).strip()]
                if not self.selected_datasets or any(s in ("all", "*") for s in self.selected_datasets):
                    self.selected_datasets = None
        else:
            self.selected_datasets = None

        self.augment_train = bool(augment_train)
        self.flip_prob = float(flip_prob)
        self.scale_jitter = float(scale_jitter)
        self.shift_jitter = float(shift_jitter)
        self.roll_deg = float(roll_deg)
        self.temporal_dropout = float(temporal_dropout)
        self.landmark_noise_std = float(landmark_noise_std)
        self.color_jitter = float(color_jitter)
        self.spatial_dropout = float(spatial_dropout)

        self.manifest_path: Optional[Path] = None
        self.manifest_summary: Dict[str, Any] = {}
        self.cache_meta: Dict[str, Any] = {}
        self.cached_data: Optional[Dict[str, Any]] = None

        if self.use_cache and self.cache_dir is not None:
            self._load_cache()

        self.samples = self._load_samples()
        if len(self.samples) == 0:
            logger.warning(f"Không tìm thấy mẫu hợp lệ nào trong {self.manifest_path} cho split='{self.split}'")

        self.extractor = extractor
        if self.extractor is None and not (self.use_cache and self.cached_data is not None):
            self.extractor = self._get_or_create_extractor()

    def _get_or_create_extractor(self) -> FastPIPNetExtractor:
        """Khởi tạo trễ (Lazy init) FastPIPNetExtractor khi cần trích xuất trực tiếp từ video."""
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
        """Kiểm tra mẫu có thuộc danh sách selected_datasets hay không."""
        if self.selected_datasets is None:
            return True
        if dataset_field and any(ds in str(dataset_field).lower() for ds in self.selected_datasets):
            return True
        key_lower = key.lower()
        return any(ds in key_lower for ds in self.selected_datasets)

    def _resolve_manifest_path(self) -> Path:
        """
        Xác định đường dẫn file manifest (.csv hoặc .json) trong thư mục dataset.
        Ưu tiên manifest_filename, fallback sang phần mở rộng đối ứng (.csv <-> .json)
        hoặc tệp .csv/.json đầu tiên có trong thư mục gốc.
        """
        candidate = Path(self.manifest_filename)
        if not candidate.is_absolute():
            candidate = self.root / self.manifest_filename

        if candidate.is_file() and candidate.suffix.lower() in (".csv", ".json"):
            return candidate

        alt_suffix = ".json" if candidate.suffix.lower() == ".csv" else ".csv"
        alt_candidate = candidate.with_suffix(alt_suffix)
        if alt_candidate.is_file():
            logger.info(f"[Manifest] Chuyển sang đọc tệp đối ứng: {alt_candidate.name}")
            return alt_candidate

        if self.root.is_dir():
            for ext in ("*.csv", "*.json"):
                matches = sorted(self.root.glob(ext))
                if matches:
                    logger.info(f"[Manifest] Tự động sử dụng manifest tìm thấy: {matches[0].name}")
                    return matches[0]

        raise FileNotFoundError(
            f"Không tìm thấy tệp manifest (.csv hoặc .json) '{self.manifest_filename}' trong thư mục: {self.root}"
        )

    def _parse_manifest_record(
            self,
            row: Dict[str, Any],
            default_split: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Chuẩn hóa 1 bản ghi từ CSV hoặc JSON sang cấu trúc sample thống nhất cho KSS 1-9."""
        raw_split = row.get("split")
        split_val = str(raw_split if raw_split is not None else (default_split or "")).strip().lower()
        if split_val != self.split:
            return None

        rel_path = row.get("rel_path") or row.get("processed_path") or row.get("path")
        if not rel_path or str(rel_path).strip() == "":
            return None

        norm_rel = Path(str(rel_path).strip().replace("\\", "/")).as_posix()
        dataset_name = str(row.get("dataset", "") or "unknown")
        if not self._is_dataset_selected(norm_rel, dataset_name):
            return None

        rel_obj = Path(norm_rel)
        full_path = rel_obj if rel_obj.is_absolute() else (self.root / norm_rel)

        raw_lbl = row.get("label")
        if raw_lbl is None or str(raw_lbl).strip() == "":
            raw_lbl = row.get("kss_score", row.get("kss_raw", 1))
        label_int = _safe_int(raw_lbl, default=1)
        kss_score = _safe_int(row.get("kss_score"), default=label_int)
        kss_raw = _safe_float(row.get("kss_raw"), default=float(kss_score))
        kss_class = max(0, min(8, label_int - 1))
        sample_weight = _safe_float(row.get("sample_weight"), default=1.0)

        return {
            "key": norm_rel,
            "path": full_path,
            "orig_path": str(row.get("orig_path", "") or ""),
            "label": label_int,
            "kss_score": kss_score,
            "kss_raw": kss_raw,
            "kss_class": kss_class,
            "sample_weight": sample_weight,
            "video_id": str(row.get("video_id") or rel_obj.stem),
            "subject_id": str(row.get("subject_id", "") or "unknown"),
            "session_id": str(row.get("session_id", "") or ""),
            "condition": str(row.get("condition", "") or ""),
            "dataset": dataset_name,
            "fps": _safe_float(row.get("fps"), default=30.0),
            "frame_count": _safe_int(row.get("frame_count"), default=0),
            "width": _safe_int(row.get("width"), default=0),
            "height": _safe_int(row.get("height"), default=0),
            "duration_sec": _safe_float(row.get("duration_sec"), default=0.0),
        }

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Chỉ đọc danh sách mẫu và đường dẫn file từ manifest .csv hoặc .json."""
        manifest_path = self._resolve_manifest_path()
        self.manifest_path = manifest_path
        ext = manifest_path.suffix.lower()
        samples: List[Dict[str, Any]] = []

        if ext == ".csv":
            with open(manifest_path, mode="r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = self._parse_manifest_record(row)
                    if parsed is not None:
                        samples.append(parsed)
        elif ext == ".json":
            with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
                data = json.load(f)

            default_split: Optional[str] = None
            if isinstance(data, dict) and isinstance(data.get("splits"), dict):
                self.manifest_summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
                records = data["splits"].get(self.split, [])
                default_split = self.split
            elif isinstance(data, dict) and ("samples" in data or "data" in data):
                self.manifest_summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
                records = data.get("samples") or data.get("data") or []
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(f"Cấu trúc tệp JSON manifest không hợp lệ: {manifest_path}")

            for row in records:
                if isinstance(row, dict):
                    parsed = self._parse_manifest_record(row, default_split=default_split)
                    if parsed is not None:
                        samples.append(parsed)
        else:
            raise ValueError(f"Chỉ hỗ trợ đọc manifest .csv hoặc .json, không hỗ trợ: {manifest_path}")

        if self.split == "train" and self.drop_single_class_subjects:
            sub_labels: Dict[str, set] = defaultdict(set)
            for s in samples:
                sub_labels[s["subject_id"]].add(s["label"])
            single_class_subs = {sub for sub, lbls in sub_labels.items() if len(lbls) < 2}
            if single_class_subs:
                orig_cnt = len(samples)
                samples = [s for s in samples if s["subject_id"] not in single_class_subs]
                logger.info(
                    f"[Dataset] Đã loại {orig_cnt - len(samples)} mẫu từ {len(single_class_subs)} "
                    f"subject chỉ có 1 mức KSS trong train: {sorted(single_class_subs)}"
                )

        return samples

    def _load_cache(self) -> None:
        """Nạp dữ liệu cache từ tệp .pt theo từng cache_dir (hỗ trợ Shared Cross-Split Cache Pool)."""
        assert self.cache_dir is not None
        pool_key = str(self.cache_dir.resolve()) if self.cache_dir.exists() else str(self.cache_dir)
        cache_pool = self._SHARED_CACHE.setdefault(pool_key, {})
        meta_pool = self._SHARED_META.setdefault(pool_key, {})

        cache_path = self.cache_dir / f"{self.split}.pt"
        if not cache_path.is_file():
            logger.warning(f"[Cache] Không tìm thấy tệp cache {cache_path}. Sẽ trích xuất on-the-fly nếu mẫu không có trong cache.")

        for split_name in (self.split, "train", "val", "test"):
            split_file = f"{split_name}.pt"
            split_path = self.cache_dir / split_file
            if split_file not in meta_pool and split_path.is_file():
                try:
                    payload = torch.load(split_path, map_location="cpu", weights_only=False)
                    if isinstance(payload, dict) and "samples" in payload:
                        items = payload["samples"]
                        header_meta = {k: v for k, v in payload.items() if k not in ("samples", "meta")}
                        nested_meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                        meta_pool[split_file] = {**header_meta, **nested_meta}
                    else:
                        items = payload
                        meta_pool[split_file] = {}
                    cache_pool.update(items)
                    logger.info(
                        f"[Cache] Đã nạp {len(items)} mẫu từ {split_file} vào Shared Cache Pool "
                        f"(Tổng: {len(cache_pool)})."
                    )
                except Exception as e:
                    logger.warning(f"[Cache] Lỗi khi nạp cache {split_path}: {e}")

        if cache_pool:
            self.cached_data = cache_pool
            split_meta = meta_pool.get(f"{self.split}.pt") or next(iter(meta_pool.values()), {})
            raw_dt = split_meta.get("time_interval", self.time_interval)
            raw_lbs = split_meta.get("letterbox_size", self.letterbox_size)
            self.cache_meta = {
                "format_version": 1,
                "split": self.split,
                "time_interval": float(self.time_interval if raw_dt is None else raw_dt),
                "window_sec": self.window_sec,
                "letterbox_size": self.letterbox_size if raw_lbs is None else raw_lbs,
                "face_size": self.face_size,
                "num_samples": len(cache_pool),
            }

    def _compute_window_span(self, duration_sec: float) -> Tuple[float, float]:
        """Xác định (start_sec, win_sec) thống nhất cho cả đọc video trực tiếp và đọc cache."""
        if self.window_sec is None:
            return 0.0, duration_sec

        if self.split != "train":
            if self.eval_window_mode == "full":
                return 0.0, duration_sec
            if self.eval_window_mode == "legacy18s":
                eval_win = min(max(self.window_sec, 18.0), duration_sec)
                return 0.0, eval_win
            if self.eval_window_mode == "start":
                eval_win = min(self.window_sec, duration_sec)
                return 0.0, eval_win
            # "center"
            eval_win = min(self.window_sec, duration_sec)
            max_start = max(0.0, duration_sec - eval_win)
            return max_start / 2.0, eval_win

        win = min(self.window_sec, duration_sec)
        max_start = max(0.0, duration_sec - win)
        start_sec = random.uniform(0.0, max_start) if max_start > 0 else 0.0
        return start_sec, win

    def _build_timestamps(self, start_sec: float, span_sec: float) -> np.ndarray:
        """Tạo mảng mốc thời gian lấy mẫu tránh lỗi làm tròn dấu phẩy động (vd: 60.0s @ 0.1s -> đúng 600)."""
        if span_sec <= 0.0:
            return np.array([start_sec], dtype=np.float64)
        end_bound = max(self.time_interval * 0.5, span_sec) - 1e-6
        timestamps = start_sec + np.arange(0.0, max(1e-7, end_bound), self.time_interval, dtype=np.float64)
        if len(timestamps) == 0:
            timestamps = np.array([start_sec], dtype=np.float64)
        return timestamps

    def _get_frame_indices(self, total_frames: int, fps: float) -> np.ndarray:
        """Tính toán chỉ số khung hình cần đọc từ video theo time_interval và window_sec."""
        if total_frames <= 0:
            return np.array([0], dtype=int)

        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps

        if self.time_interval == 0:
            if self.window_sec is None or (self.split != "train" and self.eval_window_mode == "full"):
                return np.arange(total_frames, dtype=int)
            start_sec, win_sec = self._compute_window_span(duration_sec)
            num_consec = max(1, min(int(round(win_sec * fps)), total_frames))
            start_idx = max(0, min(total_frames - num_consec, int(round(start_sec * fps))))
            return np.arange(start_idx, start_idx + num_consec, dtype=int)

        start_sec, win_sec = self._compute_window_span(duration_sec)
        timestamps = self._build_timestamps(start_sec, win_sec)
        indices = np.rint(timestamps * fps).astype(int)
        return np.clip(indices, 0, total_frames - 1)

    def _slice_cached_sequence(self, item: Dict[str, Any], fps: float) -> np.ndarray:
        """Xác định chỉ số cắt trên chuỗi cache .pt theo time_interval và window_sec."""
        landmarks = item["landmarks"]
        T_cache = int(landmarks.shape[0])
        if T_cache <= 0:
            return np.array([0], dtype=int)

        cache_dt = float(self.cache_meta.get("time_interval", 0.1))
        if cache_dt <= 0:
            cache_dt = 1.0 / (fps if fps > 0 else 30.0)

        meta = item.get("meta", {})
        duration_sec = float(meta.get("duration_sec", T_cache * cache_dt))
        if duration_sec <= 0:
            duration_sec = T_cache * cache_dt

        if self.time_interval == 0:
            if self.window_sec is None or (self.split != "train" and self.eval_window_mode == "full"):
                return np.arange(T_cache, dtype=int)
            start_sec, win_sec = self._compute_window_span(duration_sec)
            num_consec = max(1, min(int(round(win_sec / cache_dt)), T_cache))
            start_idx = max(0, min(T_cache - num_consec, int(round(start_sec / cache_dt))))
            return np.arange(start_idx, start_idx + num_consec, dtype=int)

        if (self.window_sec is None or (self.split != "train" and self.eval_window_mode == "full")) and abs(
            self.time_interval - cache_dt
        ) < 1e-5:
            return np.arange(T_cache, dtype=int)

        start_sec, win_sec = self._compute_window_span(duration_sec)
        timestamps = self._build_timestamps(start_sec, win_sec)
        pos = np.rint(timestamps / cache_dt).astype(int)
        return np.clip(pos, 0, T_cache - 1)

    def _build_kss_meta(
            self,
            sample: Dict[str, Any],
            fps: float,
            total_frames: int,
            duration_sec: float,
            selected_target_indices: np.ndarray,
            num_frames: int,
            found_ratio: float,
            letterbox_size: Tuple[int, int],
            cached: bool,
    ) -> Dict[str, Any]:
        """Khởi tạo dictionary metadata thống nhất chứa đầy đủ thông tin KSS (1-9)."""
        return {
            "key": sample["key"],
            "video_id": sample["video_id"],
            "subject_id": sample["subject_id"],
            "session_id": sample["session_id"],
            "condition": sample["condition"],
            "dataset": sample["dataset"],
            "domain_id": (0 if str(sample.get("dataset", "")).upper().startswith("UL") else 1),
            "path": str(sample["path"]),
            "orig_path": sample["orig_path"],
            "split": self.split,
            "label": int(sample["label"]),
            "kss_score": int(sample["kss_score"]),
            "kss_raw": float(sample["kss_raw"]),
            "kss_class": int(sample["kss_class"]),
            "sample_weight": float(sample["sample_weight"]),
            "fps": float(fps),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "total_frames": int(total_frames),
            "duration_sec": float(duration_sec),
            "target_indices": torch.as_tensor(selected_target_indices, dtype=torch.long),
            "num_frames": int(num_frames),
            "found_faces_ratio": float(found_ratio),
            "letterbox_size": letterbox_size,
            "cached": bool(cached),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor],
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]],
        Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], List[np.ndarray]],
    ]:
        sample = self.samples[index]
        key = sample["key"]
        label = torch.tensor(sample["label"], dtype=torch.long)

        # 1. Đường nhanh: Nạp từ RAM Cache .pt nếu có
        if self.cached_data is not None and key in self.cached_data:
            item = self.cached_data[key]
            fps = float(sample.get("fps") or item.get("meta", {}).get("fps", 30.0))
            pos = self._slice_cached_sequence(item, fps)

            landmarks = item["landmarks"][pos]
            if landmarks.shape[-1] > 2:
                landmarks = landmarks[..., :2]
            crop_boxes = item["crop_boxes"][pos] if "crop_boxes" in item else None
            bboxes = item["bboxes"][pos] if "bboxes" in item else None
            found = item["found"][pos] if "found" in item else None

            cached_target_indices = item.get("target_indices")
            if cached_target_indices is None and isinstance(item.get("meta"), dict):
                cached_target_indices = item["meta"].get("target_indices")

            if cached_target_indices is not None:
                if isinstance(cached_target_indices, torch.Tensor):
                    selected_target_indices = cached_target_indices[pos].long().cpu().numpy()
                else:
                    selected_target_indices = np.asarray(cached_target_indices, dtype=int)[pos]
            else:
                cache_dt = float(self.cache_meta.get("time_interval", 0.1))
                selected_target_indices = np.rint(pos * (cache_dt * fps)).astype(int)

            cache_lbs = _parse_size(self.cache_meta.get("letterbox_size", self.letterbox_size))
            face_frames = item.get("face_frames", None)
            if face_frames is not None:
                face_frames = face_frames[pos]

            raw_frames: Optional[List[np.ndarray]] = None
            v_stride = self.visual_frame_stride
            expected_v_len = len(range(0, len(selected_target_indices), v_stride))

            if (
                not self.extract_crops_only
                and face_frames is None
                and not self.return_raw_frames
                and self.crop_cache_dir is not None
                and self.window_sec is None
            ):
                cache_key = hashlib.md5(key.encode()).hexdigest()
                crop_file = self.crop_cache_dir / f"{cache_key}.pt"
                if crop_file.is_file():
                    try:
                        cached_uint8 = torch.load(crop_file, map_location="cpu", weights_only=True)
                        if cached_uint8.shape[0] == expected_v_len:
                            norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                            norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                            face_frames = (cached_uint8.float() / 255.0 - norm_mean) / norm_std
                    except Exception:
                        face_frames = None

            need_video_read = (not self.extract_crops_only and face_frames is None) or self.return_raw_frames
            if need_video_read:
                if v_stride > 1 and not self.return_raw_frames:
                    v_pos = np.arange(0, len(selected_target_indices), v_stride)
                    v_indices = selected_target_indices[v_pos]
                    v_crop_boxes = crop_boxes[v_pos] if crop_boxes is not None else None
                else:
                    v_indices = selected_target_indices
                    v_crop_boxes = crop_boxes

                raw_frames, actual_fps, _ = fast_read_video_frames(
                    sample["path"], v_indices, letterbox_size=cache_lbs
                )
                fps = actual_fps or fps
                if not self.extract_crops_only and face_frames is None and v_crop_boxes is not None:
                    face_frames = crop_faces_from_boxes(raw_frames, v_crop_boxes, face_size=self.face_size)
                    if self.crop_cache_dir is not None and self.window_sec is None:
                        try:
                            cache_key = hashlib.md5(key.encode()).hexdigest()
                            crop_file = self.crop_cache_dir / f"{cache_key}.pt"
                            norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                            norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                            uint8_crops = ((face_frames * norm_std + norm_mean).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                            atomic_save(uint8_crops, crop_file)
                        except Exception:
                            pass

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

            if found is not None:
                found_t = torch.as_tensor(found, dtype=torch.bool)
                found_ratio = float(found_t.sum().item()) / float(max(1, len(found_t)))
            else:
                found_t = torch.ones(len(pos), dtype=torch.bool)
                found_ratio = 1.0

            item_meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            meta = self._build_kss_meta(
                sample=sample,
                fps=fps,
                total_frames=int(sample.get("frame_count") or item_meta.get("total_frames", 0)),
                duration_sec=float(sample.get("duration_sec") or item_meta.get("duration_sec", 0.0)),
                selected_target_indices=selected_target_indices,
                num_frames=int(landmarks.shape[0]),
                found_ratio=found_ratio,
                letterbox_size=cache_lbs,
                cached=True,
            )
            if crop_boxes is not None:
                meta["crop_boxes"] = torch.as_tensor(crop_boxes, dtype=torch.int32)
            if bboxes is not None:
                meta["bboxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
            if found is not None:
                meta["found"] = found_t

            for k_meta, v_meta in item_meta.items():
                if k_meta not in meta and k_meta != "path":
                    meta[k_meta] = v_meta
            if "path" in item_meta and str(item_meta["path"]) != str(sample["path"]):
                meta["remote_path"] = str(item_meta["path"])

            if self.return_raw_frames and self.return_meta:
                return face_frames, landmarks, label, meta, (raw_frames if raw_frames is not None else [])
            if self.return_meta:
                return face_frames, landmarks, label, meta
            return face_frames, landmarks, label

        # 2. Đường chuẩn: Đọc trực tiếp từ đường dẫn file trong manifest (.csv / .json)
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
            visual_frame_stride=self.visual_frame_stride,
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

        T_frames = len(found)
        bboxes_tensor = torch.zeros((T_frames, 5), dtype=torch.float32)
        for t_idx, b in enumerate(bboxes):
            if b is not None and len(b) >= 4:
                b_arr = np.asarray(b, dtype=np.float32)
                bboxes_tensor[t_idx, :min(5, len(b_arr))] = torch.from_numpy(b_arr[:min(5, len(b_arr))])

        duration_sec = float(sample["duration_sec"]) if sample["duration_sec"] > 0 else (
            float(total_frames / actual_fps) if actual_fps > 0 else 0.0
        )
        meta = self._build_kss_meta(
            sample=sample,
            fps=actual_fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            selected_target_indices=target_indices,
            num_frames=int(landmarks.shape[0]),
            found_ratio=float(sum(found)) / float(max(1, len(found))),
            letterbox_size=self.letterbox_size,
            cached=False,
        )
        meta["crop_boxes"] = torch.from_numpy(crop_boxes).int()
        meta["bboxes"] = bboxes_tensor
        meta["found"] = torch.tensor(found, dtype=torch.bool)
        meta["raw_bboxes_list"] = bboxes

        if self.return_raw_frames and self.return_meta:
            return face_frames, landmarks, label, meta, raw_frames
        if self.return_meta:
            return face_frames, landmarks, label, meta
        return face_frames, landmarks, label


# =====================================================================
# 4. HÀM COLLATE & XÂY DỰNG DATALOADER CHO LITEFAT_V1 (KSS 1-9)
# =====================================================================
def collate_litefat(batch: List[Tuple[Any, ...]]):
    """
    Collate function cho DataLoader LiteFat_v1:
      - Tự động đệm (padding) theo chiều thời gian T khi các mẫu có độ dài khác nhau.
      - Hỗ trợ face_frames=None (chế độ Landmark-Only).
      - batch_labels có kiểu torch.long [B] chứa nhãn KSS (1..9).
    """
    num_elements = len(batch[0])
    face_frames_list = [item[0] for item in batch]
    landmarks_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]

    lengths = [lm.shape[0] for lm in landmarks_list]
    max_len = max(lengths)

    has_face_frames = all(f is not None for f in face_frames_list)
    v_lengths = [f.shape[0] for f in face_frames_list] if has_face_frames else []
    max_v_len = max(v_lengths) if has_face_frames else 0

    padded_frames = []
    padded_landmarks = []
    for idx, (lm, cur_len) in enumerate(zip(landmarks_list, lengths)):
        f = face_frames_list[idx]
        if cur_len < max_len:
            pad_lm = torch.ones((max_len - cur_len, lm.shape[1], lm.shape[2]), dtype=lm.dtype)
            padded_landmarks.append(torch.cat([lm, pad_lm], dim=0))
        else:
            padded_landmarks.append(lm)

        if has_face_frames and f is not None:
            cur_v_len = f.shape[0]
            if cur_v_len < max_v_len:
                last_frame = f[-1:].repeat(max_v_len - cur_v_len, 1, 1, 1)
                padded_frames.append(torch.cat([f, last_frame], dim=0))
            else:
                padded_frames.append(f)

    batch_frames = torch.stack(padded_frames, dim=0) if has_face_frames else None
    batch_landmarks = torch.stack(padded_landmarks, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)
    batch_lengths = torch.tensor(lengths, dtype=torch.long)

    if num_elements >= 4:
        metas = [item[3] for item in batch]
        if num_elements == 5:
            raw_frames_list = [item[4] for item in batch]
            return batch_frames, batch_landmarks, batch_labels, batch_lengths, metas, raw_frames_list
        return batch_frames, batch_landmarks, batch_labels, batch_lengths, metas

    return batch_frames, batch_landmarks, batch_labels, batch_lengths


def build_litefat_dataloaders(
        dataset_dir: Union[str, Path] = r"E:\LSTM\data_processed_v1",
        batch_size: int = 8,
        num_workers: int = 0,
        time_interval: float = 0.1,
        window_sec: Optional[float] = None,
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
        use_manifest_sample_weights: bool = True,
        eval_window_mode: str = "center",
        drop_single_class_subjects: bool = False,
        visual_frame_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Khởi tạo bộ 3 DataLoader (train, val, test) cho mô hình LiteFat_v1 (KSS 1-9 @ 10 FPS)."""
    shared_extractor = (
        FastPIPNetExtractor(
            backbone=pipnet_backbone,
            device=device,
            letterbox_size=letterbox_size,
            face_size=face_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            bbox_margin=bbox_margin,
            box_smooth_alpha=box_smooth_alpha,
        )
        if not use_cache
        else None
    )

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
        eval_window_mode=eval_window_mode,
        visual_frame_stride=visual_frame_stride,
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
        drop_single_class_subjects=drop_single_class_subjects,
        **common_cfg,
    )
    val_ds = LiteFatDataset(split="val", augment_train=False, **common_cfg)
    test_ds = LiteFatDataset(split="test", augment_train=False, **common_cfg)

    train_sampler = None
    if len(train_ds) > 0:
        if subject_balanced_sampling:
            ds_label_counts: Counter = Counter()
            sub_label_counts: Counter = Counter()
            for s in train_ds.samples:
                ds_name = str(s.get("dataset", "unknown"))
                sub = str(s.get("subject_id", "unknown"))
                lbl = int(s.get("label", 1))
                ds_label_counts[(ds_name, lbl)] += 1
                sub_label_counts[(ds_name, sub, lbl)] += 1

            raw_weights: List[float] = []
            ds_totals: Counter = Counter()
            for s in train_ds.samples:
                ds_name = str(s.get("dataset", "unknown"))
                sub = str(s.get("subject_id", "unknown"))
                lbl = int(s.get("label", 1))
                w = 1.0 / (
                    float(max(1, ds_label_counts[(ds_name, lbl)]))
                    * math.sqrt(float(max(1, sub_label_counts[(ds_name, sub, lbl)])))
                )
                if use_manifest_sample_weights:
                    w *= max(1e-6, float(s.get("sample_weight", 1.0)))
                raw_weights.append(w)
                ds_totals[ds_name] += w

            sample_weights = [
                w / max(1e-12, float(ds_totals[str(s.get("dataset", "unknown"))]))
                for s, w in zip(train_ds.samples, raw_weights)
            ]
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
        elif use_manifest_sample_weights:
            sample_weights = [max(1e-6, float(s.get("sample_weight", 1.0))) for s in train_ds.samples]
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(shuffle_train if train_sampler is None else False),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_litefat,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_litefat,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_litefat,
    )

    return train_loader, val_loader, test_loader


# =====================================================================
# 5. TRỰC QUAN HÓA MẪU KSS (1-9) & MÔ PHỎNG KINEMATIC FEATURE EXTRACTOR
# =====================================================================
def _format_kss_label(label: Union[torch.Tensor, int, float]) -> Tuple[str, Tuple[int, int, int]]:
    """Định dạng nhãn KSS (1..9) và màu hiển thị BGR."""
    val = float(label.item()) if isinstance(label, torch.Tensor) else float(label)
    if val >= 7.0:
        return f"KSS {val:.1f}/9 (DROWSY)", (0, 0, 255)
    if val >= 5.0:
        return f"KSS {val:.1f}/9 (MID-FATIGUE)", (0, 180, 255)
    return f"KSS {val:.1f}/9 (ALERT)", (0, 255, 0)


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
        target_h: int = 640,
        zoom_scale: float = 260.0,
        coord_mode: str = "nose",
) -> np.ndarray:
    """Dựng khung hình trực quan hóa mô phỏng cho lớp KinematicFeatureExtractor."""
    canvas = np.full((720, 1280, 3), (22, 26, 32), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (1280, 48), (14, 17, 22), -1)
    cv2.putText(canvas, "DRIVER GUARDIAN | KINEMATIC FEATURE EXTRACTOR (10 FPS)", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)
    top_info = f"Clip: {video_id} | Frame {t + 1}/{T} | {label_str} | Zoom: {zoom_scale:.0f}px"
    cv2.putText(canvas, top_info, (680, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 48), (1280, 48), (45, 55, 70), 1)
    cv2.line(canvas, (720, 48), (720, 720), (45, 55, 70), 1)

    vx1, vy1, vx2, vy2 = 15, 55, 705, 705
    viewport_rect = (vx1, vy1, vx2 - vx1, vy2 - vy1)
    cx, cy = 360, 380
    S = float(zoom_scale)
    is_model_mode = (coord_mode == "model")

    if is_model_mode:
        cv2.putText(canvas, "MODEL INPUT SPACE (ORGAN-CENTERED, static_shape_ratio=0.25) | [M]: Nose IOD=1.0",
                    (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 160, 190), 1, cv2.LINE_AA)
        organ_groups = [
            ("L-Eye/Brow (0,0)", (cx - 170, cy - 165)),
            ("R-Eye/Brow (0,0)", (cx + 170, cy - 165)),
            ("Jaw/Nose (0,0)", (cx, cy + 5)),
            ("Mouth (0,0)", (cx, cy + 205)),
        ]
        for g_label, (ox, oy) in organ_groups:
            cv2.line(canvas, (ox - 55, oy), (ox + 55, oy), (38, 48, 60), 1, cv2.LINE_AA)
            cv2.line(canvas, (ox, oy - 55), (ox, oy + 55), (38, 48, 60), 1, cv2.LINE_AA)
            cv2.circle(canvas, (ox, oy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, g_label, (ox - 55, oy - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 160, 190), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "LOCAL NOSE-CENTERED SPACE (0,0) & IOD = 1.0 | [M]: Model Input", (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 160, 190), 1, cv2.LINE_AA)
        cv2.line(canvas, (vx1, cy), (vx2, cy), (38, 48, 60), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, vy1), (cx, vy2), (38, 48, 60), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(S), (32, 40, 52), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(S * 0.5), (28, 35, 45), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)

    num_visible = 0
    if is_missing:
        cv2.putText(canvas, "[CANH BAO: MAT MAT / HEAD TURNED]", (120, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        if is_model_mode:
            node_anchors = [(cx, cy + 5)] * 68
            for i in (*range(17, 22), *range(36, 42)):
                node_anchors[i] = (cx - 170, cy - 165)
            for i in (*range(22, 27), *range(42, 48)):
                node_anchors[i] = (cx + 170, cy - 165)
            for i in range(48, 68):
                node_anchors[i] = (cx, cy + 205)
        else:
            node_anchors = [(cx, cy)] * 68

        pts_raw = [
            (int(round(node_anchors[i][0] + pos_t[i, 0] * S)), int(round(node_anchors[i][1] + pos_t[i, 1] * S)))
            for i in range(68)
        ]
        for start, end, is_closed, color in KINEMATIC_REGIONS:
            pts_region = pts_raw[start:end]
            for j in range(len(pts_region) - 1):
                ret, c1, c2 = cv2.clipLine(viewport_rect, pts_region[j], pts_region[j + 1])
                if ret:
                    cv2.line(canvas, c1, c2, color, 2, cv2.LINE_AA)
            if is_closed and len(pts_region) > 2:
                ret, c1, c2 = cv2.clipLine(viewport_rect, pts_region[-1], pts_region[0])
                if ret:
                    cv2.line(canvas, c1, c2, color, 2, cv2.LINE_AA)

        for px, py in pts_raw:
            if vx1 <= px <= vx2 and vy1 <= py <= vy2:
                cv2.circle(canvas, (px, py), 3, (0, 255, 120), -1, cv2.LINE_AA)
                num_visible += 1

        if (not is_model_mode) and raw_landmarks is not None and len(raw_landmarks) > 0 and raw_iod > 1.0:
            trail_pts = []
            for k in range(max(0, t - 10), t + 1):
                rdx = (raw_landmarks[k, 30, 0] - raw_landmarks[t, 30, 0]) * target_w / max(1.0, raw_iod)
                rdy = (raw_landmarks[k, 30, 1] - raw_landmarks[t, 30, 1]) * target_h / max(1.0, raw_iod)
                trail_pts.append((int(round(cx + rdx * S)), int(round(cy + rdy * S))))
            for k in range(len(trail_pts) - 1):
                ret, c1, c2 = cv2.clipLine(viewport_rect, trail_pts[k], trail_pts[k + 1])
                if ret:
                    alpha = int(90 + (k / max(1, len(trail_pts) - 1)) * 165)
                    cv2.line(canvas, c1, c2, (0, alpha, 255), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (45, 55, 70), 1)

    hud_x1, hud_x2, hud_y1, hud_y2 = 740, 1260, 60, 225
    cv2.rectangle(canvas, (hud_x1, hud_y1), (hud_x2, hud_y2), (16, 20, 26), -1)
    cv2.rectangle(canvas, (hud_x1, hud_y1), (hud_x2, hud_y2), (40, 50, 65), 1)
    cv2.putText(canvas, "REAL-TIME KINEMATIC TELEMETRY", (hud_x1 + 18, hud_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1, cv2.LINE_AA)

    nvx_val, nvy_val = float(vel_t[30, 0]), float(vel_t[30, 1])
    nax_val, nay_val = float(acc_t[30, 0]), float(acc_t[30, 1])
    cv2.putText(canvas, f"Head Velocity (Nose): |V|={np.hypot(nvx_val, nvy_val):.4f} [Vx:{nvx_val:+.3f}, Vy:{nvy_val:+.3f}]",
                (hud_x1 + 18, hud_y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Head Acceleration   : |A|={np.hypot(nax_val, nay_val):.4f} [Ax:{nax_val:+.3f}, Ay:{nay_val:+.3f}]",
                (hud_x1 + 18, hud_y1 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Raw IOD: {raw_iod:.1f} px | Visible Nodes: {num_visible}/68",
                (hud_x1 + 18, hud_y1 + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    for series, title, color, y1_g, y2_g in [
        (vel_all[:, 30, 1], "Vertical Velocity Vy(t)", (0, 255, 255), 250, 465),
        (acc_all[:, 30, 1], "Vertical Acceleration Ay(t)", (255, 100, 220), 490, 705),
    ]:
        gx1, gx2 = 740, 1260
        cv2.rectangle(canvas, (gx1, y1_g), (gx2, y2_g), (12, 14, 18), -1)
        cv2.rectangle(canvas, (gx1, y1_g), (gx2, y2_g), (40, 50, 65), 1)
        cv2.putText(canvas, title, (gx1 + 15, y1_g - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        zy = (y1_g + y2_g) // 2
        cv2.line(canvas, (gx1, zy), (gx2, zy), (35, 45, 60), 1, cv2.LINE_AA)

        max_amp = 0.15
        pts = [
            (
                int(gx1 + (i / max(1, T - 1)) * (gx2 - gx1)),
                int(zy - (np.clip(float(series[i]), -max_amp, max_amp) / max_amp) * ((y2_g - y1_g) // 2 - 15)),
            )
            for i in range(T)
        ]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
        cur_x = int(gx1 + (t / max(1, T - 1)) * (gx2 - gx1))
        cv2.line(canvas, (cur_x, y1_g), (cur_x, y2_g), (0, 0, 255), 2, cv2.LINE_AA)

    return canvas


def visualize_kinematic_simulation(
        landmarks: torch.Tensor,
        label: Union[torch.Tensor, int, float] = 1,
        video_id: str = "sample",
        delay_ms: int = 100,
        window_name: str = "Driver Guardian - KinematicFeatureExtractor Simulation",
) -> None:
    """Hiển thị trực quan hóa mô phỏng KinematicFeatureExtractor từ tensor điểm mốc."""
    if landmarks.dim() == 4:
        landmarks = landmarks.squeeze(0)
    if landmarks.shape[-1] > 2:
        landmarks = landmarks[..., :2]

    T = landmarks.shape[0]
    lms_np = landmarks.cpu().numpy()
    label_str, label_color = _format_kss_label(label)

    from model import KinematicFeatureExtractor
    kfe = KinematicFeatureExtractor(fps=10.0)
    kfe.eval()
    lms_in = landmarks.unsqueeze(0).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        kfe_out = kfe(lms_in)
        p_left_t = lms_in[:, :, :, kfe.left_eye_idx:kfe.left_eye_idx + 1]
        p_right_t = lms_in[:, :, :, kfe.right_eye_idx:kfe.right_eye_idx + 1]
        raw_iod_t = torch.norm(p_right_t - p_left_t, p=2, dim=1, keepdim=True).clamp(min=kfe.eps)
        clip_iod_t = raw_iod_t.mean(dim=2, keepdim=True).clamp(min=kfe.eps)
        iod_scale_t = torch.maximum(0.5 * raw_iod_t + 0.5 * clip_iod_t, 0.65 * clip_iod_t).clamp(min=kfe.eps)
        nose_pos_t = lms_in[:, :, :, kfe.nose_idx:kfe.nose_idx + 1]
        pos_norm_t = torch.clamp((lms_in - nose_pos_t) / iod_scale_t, min=-8.0, max=8.0)
    pos_norm_all = pos_norm_t[0].permute(1, 2, 0).cpu().numpy()
    pos_feat_all = kfe_out[0, :2].permute(1, 2, 0).cpu().numpy()
    vel_norm_all = kfe_out[0, 2:4].permute(1, 2, 0).cpu().numpy()
    acc_norm_all = kfe_out[0, 4:6].permute(1, 2, 0).cpu().numpy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    t, paused, zoom_scale, coord_mode = 0, False, 260.0, "nose"
    while True:
        is_missing = np.allclose(lms_np[t], 1.0)
        p_left = lms_np[t, 36, :2] * np.array([640, 640])
        p_right = lms_np[t, 45, :2] * np.array([640, 640])
        raw_iod = float(np.linalg.norm(p_right - p_left)) if not is_missing else 0.0
        pos_cur = pos_norm_all[t] if coord_mode == "nose" else pos_feat_all[t]

        canvas = render_kinematic_frame(
            pos_t=pos_cur,
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
            target_h=640,
            zoom_scale=zoom_scale,
            coord_mode=coord_mode,
        )
        cv2.imshow(window_name, canvas)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key == 32:
            paused = not paused
        elif key in (ord("r"), ord("R")):
            t = 0
            continue
        elif key in (ord("d"), ord("D"), 83):
            t = (t + 1) % T
            continue
        elif key in (ord("a"), ord("A"), 81):
            t = (t - 1 + T) % T
            continue
        elif key in (ord("+"), ord("=")):
            zoom_scale = min(600.0, zoom_scale + 30.0)
            continue
        elif key in (ord("-"), ord("_")):
            zoom_scale = max(100.0, zoom_scale - 30.0)
            continue
        elif key == ord("0"):
            zoom_scale = 260.0
            continue
        elif key in (ord("m"), ord("M")):
            coord_mode = "model" if coord_mode == "nose" else "nose"
            continue

        if not paused:
            t = (t + 1) % T

    cv2.destroyAllWindows()


def visualize_litefat_sample(
        face_frames: Optional[torch.Tensor],
        landmarks: torch.Tensor,
        label: Union[torch.Tensor, int, float],
        raw_frames: List[np.ndarray],
        meta: Optional[Dict[str, Any]] = None,
        delay_ms: int = 100,
        window_name: str = "Driver Guardian - LiteFat_v1 KSS Preview",
) -> None:
    """Hiển thị trực quan mẫu KSS (1-9) kèm landmarks, bounding box và ảnh mặt cắt (nếu có)."""
    T = len(raw_frames)
    if T == 0:
        return

    label_str, label_color = _format_kss_label(label)
    video_id = meta.get("video_id", "sample") if meta else "sample"
    bboxes_list = meta.get("bboxes", [None] * T) if meta else [None] * T
    crop_boxes_list = meta.get("crop_boxes", None) if meta else None

    has_faces = face_frames is not None and face_frames.shape[0] > 0
    faces_bgr = []
    if has_faces and face_frames is not None:
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        unnorm = (face_frames * norm_std + norm_mean).clamp(0.0, 1.0)
        faces_rgb = (unnorm.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        faces_bgr = [cv2.cvtColor(faces_rgb[i], cv2.COLOR_RGB2BGR) for i in range(faces_rgb.shape[0])]

    lms_np = landmarks.cpu().numpy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    t, paused = 0, False
    while True:
        left_img = raw_frames[t].copy()
        target_h, target_w = left_img.shape[:2]
        is_missing = np.allclose(lms_np[t], 1.0)

        if not is_missing:
            pts_px = lms_np[t, :, :2].copy()
            pts_px[:, 0] *= float(target_w)
            pts_px[:, 1] *= float(target_h)
            bbox = bboxes_list[t] if t < len(bboxes_list) else None
            bbox_arr = bbox.detach().cpu().numpy() if isinstance(bbox, torch.Tensor) else (
                np.asarray(bbox, dtype=np.float32) if bbox is not None else None
            )
            if bbox_arr is not None and len(bbox_arr) >= 5 and float(bbox_arr[4]) > 0:
                bboxes_to_draw = np.array([bbox_arr[:5]], dtype=np.float32)
            else:
                bboxes_to_draw = np.array(
                    [[np.min(pts_px[:, 0]), np.min(pts_px[:, 1]), np.max(pts_px[:, 0]), np.max(pts_px[:, 1]), 1.0]],
                    dtype=np.float32,
                )
            draw_68_landmarks_custom(left_img, np.expand_dims(pts_px, 0), bboxes_to_draw, conf_threshold=0.1)

        if crop_boxes_list is not None and t < len(crop_boxes_list):
            cb = crop_boxes_list[t]
            if isinstance(cb, torch.Tensor):
                cb = cb.detach().cpu().numpy()
            if len(cb) >= 4:
                cv2.rectangle(left_img, (int(cb[0]), int(cb[1])), (int(cb[2]), int(cb[3])), (0, 255, 255), 1)

        if has_faces and faces_bgr:
            f_idx = min(len(faces_bgr) - 1, int(round(t * len(faces_bgr) / float(max(1, T)))))
            right_panel = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            face_disp = cv2.resize(faces_bgr[f_idx], (target_h - 45, target_h - 45), interpolation=cv2.INTER_LINEAR)
            right_panel[45:, 22:22 + target_h - 45] = face_disp
            combined = np.hstack([left_img, right_panel])
        else:
            combined = left_img

        top_bar = f"Video: {video_id} | {label_str} | Frame {t + 1}/{T}" + (" [PAUSED]" if paused else "")
        cv2.putText(combined, top_bar, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)
        cv2.imshow(window_name, combined)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key == 32:
            paused = not paused
        elif key in (ord("r"), ord("R")):
            t = 0
            continue
        elif key in (ord("d"), ord("D"), 83):
            t = (t + 1) % T
            continue
        elif key in (ord("a"), ord("A"), 81):
            t = (t - 1 + T) % T
            continue

        if not paused:
            t = (t + 1) % T

    cv2.destroyAllWindows()


# =====================================================================
# 6. KIỂM THỬ TÍCH HỢP VỚI CACHE & TRỰC QUAN HÓA MẪU LITEFAT_V1
# =====================================================================
if __name__ == "__main__":
    DATASET_DIR = r"E:\LSTM\data_processed_v1"
    CACHE_DIR = r"E:\LSTM\litefat_v1_cache_pt\cache_pt"
    MANIFEST_FILENAME = "dataset_merged_split.csv"
    SPLIT = "test"
    SAMPLE_IDX = 10

    print("=" * 72)
    print("TRỰC QUAN HÓA MẪU LITEFAT_V1 TỪ CACHE .PT (KSS 1-9 @ 10 FPS)")
    print("=" * 72)

    if not Path(DATASET_DIR).is_dir():
        print(f"[ERROR] Không tìm thấy thư mục {DATASET_DIR}.")
        sys.exit(1)
    if not Path(CACHE_DIR).is_dir():
        print(f"[ERROR] Không tìm thấy thư mục cache {CACHE_DIR}.")
        sys.exit(1)

    ds = LiteFatDataset(
        dataset_dir=DATASET_DIR,
        manifest_filename=MANIFEST_FILENAME,
        split=SPLIT,
        time_interval=0.1,
        window_sec=60.0,
        eval_window_mode="center",
        return_meta=True,
        return_raw_frames=True,
        use_cache=True,
        cache_dir=CACHE_DIR,
        use_visual_context=True,
        visual_frame_stride=5,
    )
    print(f"Số mẫu ({SPLIT}): {len(ds)}")
    if len(ds) == 0:
        print("[ERROR] Không có mẫu hợp lệ nào để trực quan hóa.")
        sys.exit(1)

    idx = min(SAMPLE_IDX, len(ds) - 1)
    face_frames, landmarks, label, meta, raw_frames = ds[idx]
    print(f"\n--- THÔNG TIN MẪU [{idx}] ---")
    print(f"Video ID          : {meta['video_id']} | Dataset: {meta['dataset']}")
    print(f"Nguồn cache       : {meta.get('cached', False)} ({CACHE_DIR})")
    print(f"Đường dẫn video   : {meta['path']}")
    print(f"FPS / Tỷ lệ mặt   : {meta['fps']:.2f} FPS | {meta['found_faces_ratio'] * 100:.1f}%")
    print(f"face_frames shape : {tuple(face_frames.shape) if face_frames is not None else None}")
    print(f"landmarks shape   : {tuple(landmarks.shape)}")
    print(f"raw_frames count  : {len(raw_frames)}")
    print(f"Label (KSS 1-9)   : {label.item()} (kss_raw={meta['kss_raw']}, weight={meta['sample_weight']:.4f})")

    from model import LiteFatV1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteFatV1(pretrained_backbone=False).to(device)
    model.eval()
    with torch.no_grad():
        video_in = face_frames.unsqueeze(0).to(device) if face_frames is not None else None
        frame_kss, clip_kss = model(video_frames=video_in, landmarks=landmarks.unsqueeze(0).to(device))
    print(f"LiteFat_v1 out    : frame_kss={tuple(frame_kss.shape)}, clip_kss={clip_kss.item():.3f}")

    if not os.environ.get("HEADLESS"):
        print("\n[1/2] Mở cửa sổ visualize_litefat_sample (video + 68 landmarks + bbox + ảnh mặt cắt 224x224)...")
        print("Phím điều khiển: [Space] Tạm dừng/Tiếp tục | [a/d] Lùi/Tiến frame | [r] Phát lại | [q/ESC] Đóng & chuyển bước tiếp")
        try:
            visualize_litefat_sample(
                face_frames=face_frames,
                landmarks=landmarks,
                label=label,
                raw_frames=raw_frames,
                meta=meta,
                delay_ms=100,
            )
            print("\n[2/2] Mở cửa sổ visualize_kinematic_simulation (mô phỏng KinematicFeatureExtractor)...")
            print("Phím điều khiển: [Space] Tạm dừng/Tiếp tục | [a/d] Lùi/Tiến | [m] Đổi hệ tọa độ | [+/-] Zoom | [0] Reset Zoom | [r] Phát lại | [q/ESC] Thoát")
            visualize_kinematic_simulation(
                landmarks=landmarks,
                label=label,
                video_id=meta["video_id"],
                delay_ms=100,
            )
        except Exception as e:
            print(f"[INFO] Không thể mở GUI trực quan hóa ({e}).")
    else:
        print("\n[INFO] HEADLESS=1 đang bật, bỏ qua bước mở cửa sổ GUI OpenCV.")
