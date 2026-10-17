import inspect
import random
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np

def resize_letterbox(image, new_size=480, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    x, y = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def _params(cls):
    return inspect.signature(cls.__init__).parameters

def _make_gauss_noise(var_limit, p):
    if "var_limit" in _params(A.GaussNoise):
        return A.GaussNoise(var_limit=var_limit, p=p)
    std_range = tuple(np.clip(np.sqrt(v) / 255, 0, 1) for v in var_limit)
    return A.GaussNoise(std_range=std_range, p=p)

def _make_random_gamma(gamma_range, p):
    key = "gamma_range" if "gamma_range" in _params(A.RandomGamma) else "gamma_limit"
    return A.RandomGamma(**{key: gamma_range}, p=p)

def _make_clahe(clip_range, p):
    key = "clip_range" if "clip_range" in _params(A.CLAHE) else "clip_limit"
    return A.CLAHE(**{key: clip_range}, tile_grid_size=(8, 8), p=p)

def _make_shift_scale_rotate(shift, scale, rotate, p, fill=(114, 114, 114)):
    if hasattr(A, "Affine"):
        key = "fill" if "fill" in _params(A.Affine) else "cval"
        return A.Affine(scale=(1-scale, 1+scale), translate_percent=(-shift, shift),
                        rotate=(-rotate, rotate), border_mode=cv2.BORDER_CONSTANT, **{key: fill}, p=p)

    key = "fill" if "fill" in _params(A.ShiftScaleRotate) else "value"
    return A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, rotate_limit=rotate,
                              border_mode=cv2.BORDER_CONSTANT, **{key: fill}, p=p)


def letterbox(image: np.ndarray, new_size: int = 480, color=(114, 114, 114)) -> np.ndarray:
    return resize_letterbox(image, new_size, color)


class DetectionAugmenter:
    """Augmentation cho Object Detection và Video với temporal consistency."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.video_seed: Optional[int] = None

        shift, scale, rotate, ssr_p = cfg["shiftScaleRotate"]
        hue, sat, val, hsv_p = cfg["hueSaturationValue"]
        var_min, var_max, noise_p = cfg["gaussNoise"]
        blur_limit, blur_p = cfg["blur"]
        gamma_min, gamma_max, gamma_p = cfg.get("randomGamma", (55, 155, 0.5))
        clahe_range, clahe_p = cfg.get("clahe", ((1.0, 2.5), 0.2))

        self.transform = A.Compose([
            A.HorizontalFlip(p=cfg["horizontalFlip"]),
            _make_shift_scale_rotate(shift, scale, rotate, ssr_p),
            A.RandomBrightnessContrast(brightness_limit=cfg.get("brightnessLimit", 0.3),
                                       contrast_limit=cfg.get("contrastLimit", 0.2),
                                       p=cfg["randomBrightnessContrast"]),
            _make_random_gamma((gamma_min, gamma_max), gamma_p),
            _make_clahe(clahe_range, clahe_p),
            A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=hsv_p),
            _make_gauss_noise((var_min, var_max), noise_p),
            A.Blur(blur_limit=blur_limit, p=blur_p),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                                    min_visibility=0.4), p=cfg["p"])

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(seed)

    def set_video_seed(self, seed: Optional[int] = None) -> int:
        self.video_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        return self.video_seed

    def reset_video_seed(self):
        self.video_seed = None

    def __call__(self, image: np.ndarray, boxes=None, labels=None, seed: Optional[int] = None):
        seed = seed if seed is not None else self.video_seed
        if seed is not None:
            self.set_seed(seed)

        boxes = np.asarray(boxes, dtype=np.float32).tolist() if boxes is not None and len(boxes) else []
        labels = np.asarray(labels, dtype=np.int64).tolist() if labels is not None and len(labels) else []

        try:
            out = self.transform(image=image, bboxes=boxes, category_ids=labels)
            return out["image"], out["bboxes"], [int(c) for c in out["category_ids"]]
        except Exception as e:
            raise RuntimeError("Video augmentation failed") from e

    def augment_video(self, frames: List[np.ndarray], boxes_list: Optional[List] = None,
                      labels_list: Optional[List] = None, seed: Optional[int] = None) -> Tuple[List, List, List, int]:
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        aug_frames, aug_boxes, aug_labels = [], [], []

        for i, frame in enumerate(frames):
            boxes = boxes_list[i] if boxes_list is not None else []
            labels = labels_list[i] if labels_list is not None else []
            img, b, l = self(frame, boxes, labels, seed=seed)

            aug_frames.append(img)
            if boxes_list is not None:
                aug_boxes.append(b)
            if labels_list is not None:
                aug_labels.append(l)

        return aug_frames, aug_boxes, aug_labels, seed

    def apply_sequence(self, frames: List[np.ndarray], seed: Optional[int] = None) -> List[np.ndarray]:
        return self.augment_video(frames, seed=seed)[0]

    def augment_frame_only(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        return self(image, seed=seed)[0]

DEFAULT_AUGMENTATION = {
    "horizontalFlip": 0.5,
    "shiftScaleRotate": (0.08, 0.10, 15, 0.8),
    "randomBrightnessContrast": 0.6,
    "brightnessLimit": 0.3,
    "contrastLimit": 0.2,
    "randomGamma": (55, 155, 0.5),
    "clahe": ((1.0, 2.5), 0.2),
    "hueSaturationValue": (15, 20, 15, 0.3),
    "gaussNoise": (5.0, 25.0, 0.25),
    "blur": (5, 0.15),
    "p": 1.0,
}

def get_video_augmenter(cfg: Optional[dict] = None) -> DetectionAugmenter:
    return DetectionAugmenter(DEFAULT_AUGMENTATION if cfg is None else cfg)