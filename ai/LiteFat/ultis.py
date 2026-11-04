from typing import Optional, Tuple, List

import cv2
import numpy as np


def letterbox(
        image: np.ndarray,
        new_size: Optional[int | Tuple[int, int] | List[int]] = (640, 480),
        color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ khung hình gốc với đệm màu viền về kích thước đích (w, h).
    - image: Ảnh BGR numpy array shape (H, W, 3).
    - new_size: Kích thước đích theo (w, h) (vd: (640, 480)) hoặc số nguyên S tương ứng (S, S).
    - color: Màu đệm viền BGR, mặc định (114, 114, 114).
    Trả về ảnh numpy array có shape (target_h, target_w, 3).
    """
    target_w, target_h = new_size
    if target_w <= 0 or target_h <= 0:
        return image

    h, w = image.shape[:2]
    if w == target_w and h == target_h:
        return image

    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    if nw == target_w and nh == target_h:
        return resized

    canvas = np.full((target_h, target_w, 3), color, dtype=image.dtype)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas