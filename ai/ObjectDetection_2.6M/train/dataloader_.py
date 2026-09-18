import os
import math
import json
import random
import pickle
import inspect
import threading
from datetime import datetime
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from src.config import TrainConfig
from utils.artifacts import normalize_categories, signature

# ----------------------------- Augmentation ----------------------------- #
def _make_gauss_noise(var_limit, p):
    params = inspect.signature(A.GaussNoise.__init__).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=var_limit, p=p)
    if "std_range" in params:
        var_min, var_max = var_limit
        std_min = max(0.0, min(1.0, (var_min ** 0.5) / 255.0))
        std_max = max(0.0, min(1.0, (var_max ** 0.5) / 255.0))
        return A.GaussNoise(std_range=(std_min, std_max), p=p)
    raise RuntimeError("Phiên bản albumentations hiện tại không hỗ trợ tham số GaussNoise đã biết.")

def _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, p, fill_color=(114, 114, 114)):
    params = inspect.signature(A.ShiftScaleRotate.__init__).parameters
    kwargs = dict(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                  border_mode=cv2.BORDER_CONSTANT, p=p)
    kwargs["value" if "value" in params else "fill"] = fill_color
    return A.ShiftScaleRotate(**kwargs)

class DetectionAugmenter:
    def __init__(self, cfg: TrainConfig):
        shift_limit, scale_limit, rotate_limit, ssr_p = cfg.shiftScaleRotate
        hue_shift, sat_shift, val_shift, hsv_p = cfg.hueSaturationValue
        var_min, var_max, gn_p = cfg.gaussNoise
        blur_limit, blur_p = cfg.blur

        self.transform = A.Compose([
            A.HorizontalFlip(p=cfg.horizontalFlip),
            _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, ssr_p),
            A.RandomBrightnessContrast(p=cfg.randomBrightnessContrast),
            A.HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift,
                                  val_shift_limit=val_shift, p=hsv_p),
            _make_gauss_noise((var_min, var_max), p=gn_p),
            A.Blur(blur_limit=blur_limit, p=blur_p),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.4))

    def __call__(self, image, boxes, labels):
        boxes = np.array(boxes, dtype=np.float32).tolist()
        labels = np.array(labels, dtype=np.int64).tolist()
        if not boxes:
            return image, boxes, labels
        try:
            out = self.transform(image=image, bboxes=boxes, category_ids=labels)
            return out["image"], out["bboxes"], out["category_ids"]
        except Exception as e:
            print(f"[Augmenter][Warning] Bỏ qua augment do lỗi: {e}")
            return image, boxes, labels


# ------------------------------- Utilities ------------------------------- #

_ERROR_LOG_LOCK = threading.Lock()

def _log_bad_sample(error_log_path, split_name, image_id, index, error):
    if not error_log_path:
        return
    try:
        os.makedirs(os.path.dirname(error_log_path) or ".", exist_ok=True)
        line = (f"{datetime.now().isoformat()}\tsplit={split_name}\t"
                f"image_id={image_id}\tindex={index}\terror={error}\n")
        with _ERROR_LOG_LOCK, open(error_log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

def letterbox(image, new_size, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left, pad_top = (new_size - new_w) // 2, (new_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


# ------------------------- Cached index builders ------------------------- #
# 3 hàm build_* dưới đây đều theo cùng 1 khuôn: có cache thì load, không thì
# build rồi lưu lại -> gom logic cache/log dùng chung vào 2 helper bên dưới.

def _load_or_build(cache_path, force_rebuild, build_fn, desc):
    if cache_path and os.path.isfile(cache_path) and not force_rebuild:
        print(f"[Data] Load {desc} đã cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"[Data] Đang build {desc} ...")
    result = build_fn()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Data] Đã lưu cache tại: {cache_path}")
    return result


def _iter_jsonl(path):
    """Yield (offset, record) cho từng dòng hợp lệ; dòng hỏng bị bỏ qua và đếm lại."""
    n_bad = 0
    with open(path, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            if line.strip():
                try:
                    yield offset, json.loads(line)
                except Exception as e:
                    n_bad += 1
                    print(f"[Data][Warning] Bỏ qua dòng hỏng tại offset={offset} trong '{path}': {e}")
            offset = f.tell()
            line = f.readline()
    if n_bad:
        print(f"[Data][Warning] Tổng cộng {n_bad:,} dòng hỏng bị bỏ qua khi build index từ '{path}'.")


def build_id_offset_index(jsonl_path, id_field="id", cache_path=None, force_rebuild=False):
    def _build():
        index = {rec[id_field]: offset for offset, rec in _iter_jsonl(jsonl_path)}
        print(f"[Data] Xong: {len(index):,} dòng đã được index.")
        return index
    return _load_or_build(cache_path, force_rebuild, _build, f"id->offset index từ '{jsonl_path}'")


def build_annotation_group_index(annotations_path, cache_path=None, force_rebuild=False):
    def _build():
        index = defaultdict(list)
        for offset, rec in _iter_jsonl(annotations_path):
            index[rec["image_id"]].append(offset)
        index = dict(index)
        n_ann = sum(len(v) for v in index.values())
        print(f"[Data] Xong: {len(index):,} ảnh có annotation, tổng {n_ann:,} annotation.")
        return index
    return _load_or_build(cache_path, force_rebuild, _build, f"annotation group index từ '{annotations_path}'")


def load_image_path_map(path_map_file, cache_path=None, force_rebuild=False):
    def _build():
        mapping, n_bad = {}, 0
        with open(path_map_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mapping[rec["image_name"]] = rec["path"]
                except Exception as e:
                    n_bad += 1
                    print(f"[Data][Warning] Bỏ qua dòng hỏng (dòng số {line_no}) trong '{path_map_file}': {e}")
        if n_bad:
            print(f"[Data][Warning] Tổng cộng {n_bad:,} dòng hỏng bị bỏ qua khi load '{path_map_file}'.")
        print(f"[Data] Xong: {len(mapping):,} ảnh trong image_path_map.")
        return mapping
    return _load_or_build(cache_path, force_rebuild, _build, f"image_path_map từ '{path_map_file}'")


def load_categories(categories_path):
    with open(categories_path, "r", encoding="utf-8") as f:
        records = json.load(f) if str(categories_path).endswith(".json") else [json.loads(line) for line in f if line.strip()]
    records = normalize_categories(records)
    cat_id_to_idx = {r["id"]: i for i, r in enumerate(records)}
    return cat_id_to_idx, records, len(records)


def load_class_sampling(path, cat_id_to_idx):
    if not path or not os.path.isfile(path):
        print(f"[Data] Không tìm thấy class_sampling_path={path!r}; mặc định mọi class là 100%.")
        return {category_id: 1.0 for category_id in cat_id_to_idx}
    weights = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                category_id, probability = rec["id"], rec["probability"]
                if type(category_id) is not int or category_id not in cat_id_to_idx:
                    raise ValueError(f"id={category_id!r} không có trong categories của dataset")
                if category_id in weights:
                    raise ValueError(f"id={category_id} bị lặp")
                if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                    raise ValueError("probability phải là số")
                if not math.isfinite(probability) or probability < 0:
                    raise ValueError("probability phải hữu hạn và >= 0")
                weights[category_id] = max(1.0, probability / 100.0)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Sampling JSONL lỗi tại dòng {line_no}: {e}") from e
    if not weights:
        raise ValueError("Sampling JSONL rỗng")
    return weights


def annotation_target(ann, cat_id_to_idx, width, height, skip_iscrowd, skip_isfake):
    if (skip_iscrowd and ann.get("iscrowd", 0) == 1 or
            skip_isfake and ann.get("isfake", 0) == 1 or ann.get("category_id") not in cat_id_to_idx):
        return None
    try:
        x, y, w, h = map(float, ann["bbox"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
        return None
    box = [max(0., x), max(0., y), min(width, x + w), min(height, y + h)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box, cat_id_to_idx[ann["category_id"]]


def collect_image_categories(image_ids, images_info_path, images_offset_index,
                             annotations_path, ann_group_index, cat_id_to_idx,
                             skip_iscrowd, skip_isfake):
    image_categories = {}
    with open(images_info_path, "rb") as images, open(annotations_path, "rb") as annotations:
        for image_id in image_ids:
            images.seek(images_offset_index[image_id])
            info = json.loads(images.readline())
            categories = set()
            for offset in ann_group_index.get(image_id, []):
                annotations.seek(offset)
                ann = json.loads(annotations.readline())
                if annotation_target(ann, cat_id_to_idx, info["width"], info["height"],
                                     skip_iscrowd, skip_isfake) is not None:
                    categories.add(ann["category_id"])
            image_categories[image_id] = categories
    return image_categories


def oversample_image_ids(image_ids, image_categories, class_sampling, seed=42):
    rng = random.Random(seed)
    sampled = []
    for image_id in image_ids:
        # Một ảnh nhiều class chỉ lấy hệ số lớn nhất, không nhân theo số bbox.
        factor = max((class_sampling.get(c, 1.0) for c in image_categories[image_id]), default=1.0)
        count = int(factor) + (rng.random() < factor % 1)
        sampled.extend([image_id] * count)

    before = Counter(c for i in image_ids for c in image_categories[i])
    after = Counter(c for i in sampled for c in image_categories[i])
    print(f"[Data] Oversample: {len(image_ids):,} -> {len(sampled):,} samples "
          f"({len(sampled) / max(1, len(image_ids)):.2f}x)")
    for category_id, factor in sorted(class_sampling.items()):
        if factor > 1:
            print(f"[Data] Class id={category_id}: {before[category_id]:,} -> "
                  f"{after[category_id]:,} ảnh/epoch; repeat_factor={factor:.3f}")
    return sampled


# --------------------------------- Dataset -------------------------------- #

class ObjectDetectionDataset(Dataset):
    def __init__(self, images_info_path, annotations_path, image_path_map,
                 images_root_dir, images_split_dir, cat_id_to_idx, image_ids,
                 images_offset_index, ann_group_index,
                 imgsz=480, augmenter=None,
                 skip_iscrowd=True, skip_isfake=True,
                 split_name="train", error_log_path=None, max_load_retries=10):
        self.images_info_path = images_info_path
        self.annotations_path = annotations_path
        self.image_path_map = image_path_map
        self.images_split_root = os.path.join(images_root_dir, images_split_dir)
        self.cat_id_to_idx = cat_id_to_idx
        self.image_ids = image_ids
        self.images_offset_index = images_offset_index
        self.ann_group_index = ann_group_index
        self.imgsz = imgsz
        self.augmenter = augmenter
        self.skip_iscrowd = skip_iscrowd
        self.skip_isfake = skip_isfake
        self.split_name = split_name
        self.error_log_path = error_log_path
        self.max_load_retries = max(1, int(max_load_retries))

        n_ann = sum(len(self.ann_group_index.get(i, [])) for i in self.image_ids)
        print(f"[Data] Dataset sẵn sàng với {len(self.image_ids):,} ảnh, {n_ann:,} annotation "
              f"(augment={'ON' if augmenter is not None else 'OFF'}). RAM cho pixel data: ~0MB.")

    def __len__(self):
        return len(self.image_ids)

    def _read_image_info(self, image_id):
        with open(self.images_info_path, "rb") as f:
            f.seek(self.images_offset_index[image_id])
            return json.loads(f.readline())

    def _read_annotations(self, image_id):
        offsets = self.ann_group_index.get(image_id, [])
        if not offsets:
            return []
        with open(self.annotations_path, "rb") as f:
            records = []
            for off in offsets:
                f.seek(off)
                records.append(json.loads(f.readline()))
        return records

    def _load_item(self, index):
        image_id = self.image_ids[index]
        info = self._read_image_info(image_id)

        file_name = info["file_name"]
        rel_path = self.image_path_map.get(file_name)
        if rel_path is None:
            raise KeyError(f"Không tìm thấy file_name='{file_name}' (image_id={image_id}) trong image_path_map.")
        img_path = os.path.join(self.images_split_root, rel_path)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc tệp ảnh tại đường dẫn: {img_path}")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Ảnh tại '{img_path}' có shape bất thường {img_bgr.shape} "
                              f"(kỳ vọng H,W,3) - có thể là ảnh grayscale/CMYK/hỏng.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[0] == self.imgsz and img_rgb.shape[1] == self.imgsz:
            img_resized, scale, pad_left, pad_top = img_rgb, 1.0, 0, 0
        else:
            img_resized, scale, pad_left, pad_top = letterbox(img_rgb, self.imgsz)

        boxes, labels = [], []
        for ann in self._read_annotations(image_id):
            target = annotation_target(ann, self.cat_id_to_idx, img_rgb.shape[1], img_rgb.shape[0],
                                       self.skip_iscrowd, self.skip_isfake)
            if target is None:
                continue
            box, label = target
            boxes.append([box[0] * scale + pad_left, box[1] * scale + pad_top,
                          box[2] * scale + pad_left, box[3] * scale + pad_top])
            labels.append(label)

        if self.augmenter is not None:
            img_resized, boxes, labels = self.augmenter(img_resized, boxes, labels)

        img_numpy = np.ascontiguousarray(img_resized, dtype=np.uint8).copy()
        img_tensor = torch.from_numpy(img_numpy).permute(2, 0, 1).float() / 255.0

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, {"boxes": boxes_tensor, "labels": labels_tensor}

    def __getitem__(self, index):
        sample_seed = index
        if isinstance(index, tuple):
            sample_seed, index = index
        rng = random.Random(sample_seed)
        if self.augmenter is not None:
            self.augmenter.transform.set_random_seed(sample_seed)
        last_err, cur_index = None, index
        for attempt in range(self.max_load_retries):
            try:
                return self._load_item(cur_index)
            except Exception as e:
                last_err = e
                image_id = self.image_ids[cur_index] if cur_index < len(self.image_ids) else "?"
                print(f"[Data][Warning] Lỗi khi đọc sample idx={cur_index} "
                      f"(image_id={image_id}, lần thử {attempt + 1}/{self.max_load_retries}): {e} "
                      f"- thử lấy sample khác thay thế.")
                _log_bad_sample(self.error_log_path, self.split_name, image_id, cur_index, e)
                cur_index = rng.randrange(len(self))
        raise RuntimeError(f"Không thể load được sample sau {self.max_load_retries} lần thử liên tiếp "
                            f"(bắt đầu từ index={index}, split={self.split_name}). Lỗi gần nhất: {last_err}")


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


class EpochBatchSampler:
    def __init__(self, dataset, batch_size, shuffle, drop_last, seed):
        self.size, self.batch_size = len(dataset), batch_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = self.start_batch = 0

    def __len__(self):
        return (self.size // self.batch_size if self.drop_last else
                (self.size + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist() if self.shuffle else range(self.size)
        for batch in range(self.start_batch, len(self)):
            start = batch * self.batch_size
            yield [(self.seed + self.epoch * self.size + i, i)
                   for i in indices[start:start + self.batch_size]]


# ------------------------------ Loader builders ---------------------------- #

def build_split_dataset(cfg: TrainConfig, split_dir, is_train, image_path_map_filename, images_split_dir, cat_id_to_idx):
    class_sampling = load_class_sampling(cfg.class_sampling_path, cat_id_to_idx) if is_train and cfg.train_class_sampling else {}
    images_info_path = os.path.join(split_dir, cfg.images_info_filename)
    annotations_path = os.path.join(split_dir, cfg.annotations_filename)
    image_path_map_path = os.path.join(split_dir, image_path_map_filename)

    os.makedirs(cfg.index_cache_dir, exist_ok=True)
    split_name = os.path.basename(os.path.normpath(split_dir))

    images_offset_index = build_id_offset_index(
        images_info_path, id_field="id",
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_images_info.idx.pkl"),
        force_rebuild=cfg.rebuild_index,
    )
    ann_group_index = build_annotation_group_index(
        annotations_path,
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_annotations_group.idx.pkl"),
        force_rebuild=cfg.rebuild_index,
    )
    image_path_map = load_image_path_map(
        image_path_map_path,
        cache_path=os.path.join(cfg.index_cache_dir, f"{split_name}_image_path_map.pkl"),
        force_rebuild=cfg.rebuild_index,
    )

    image_ids = list(images_offset_index)
    image_categories = collect_image_categories(
        image_ids, images_info_path, images_offset_index, annotations_path, ann_group_index,
        cat_id_to_idx, cfg.skip_iscrowd, cfg.skip_isfake)
    if not cfg.include_images_without_annotations:
        image_ids = [i for i in image_ids if image_categories[i]]

    if is_train and cfg.train_class_sampling:
        image_ids = oversample_image_ids(image_ids, image_categories, class_sampling, cfg.seed)
        cfg.sampling_sha256 = signature({"weights": class_sampling, "image_ids": image_ids})
    elif is_train:
        cfg.sampling_sha256 = None

    return ObjectDetectionDataset(
        images_info_path=images_info_path,
        annotations_path=annotations_path,
        image_path_map=image_path_map,
        images_root_dir=cfg.images_root_dir,
        images_split_dir=images_split_dir,
        cat_id_to_idx=cat_id_to_idx,
        image_ids=image_ids,
        images_offset_index=images_offset_index,
        ann_group_index=ann_group_index,
        imgsz=cfg.img_size,
        augmenter=DetectionAugmenter(cfg) if is_train else None,
        skip_iscrowd=cfg.skip_iscrowd,
        skip_isfake=cfg.skip_isfake,
        split_name=split_name,
        error_log_path=os.path.join(
            cfg.log_dir, cfg.data_error_log_filename.format(split=split_name)
        ),
        max_load_retries=cfg.max_load_retries,
    )


def build_dataloaders(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_dir = os.path.join(cfg.labels_root, cfg.train_subdir)
    val_dir = os.path.join(cfg.labels_root, cfg.val_subdir)

    cat_id_to_idx, classes, num_classes = load_categories(os.path.join(train_dir, cfg.categories_filename))
    if cfg.nc != num_classes:
        raise ValueError(f"nc={cfg.nc} trong config không khớp {num_classes} categories; không tự đổi kiến trúc")
    val_categories = os.path.join(val_dir, cfg.categories_filename)
    if os.path.isfile(os.path.join(val_dir, cfg.images_info_filename)):
        _, val_classes, _ = load_categories(val_categories)
        if classes != val_classes:
            raise ValueError("Categories train/val không khớp id/name/index")

    train_dataset = build_split_dataset(
        cfg, train_dir, is_train=True,
        image_path_map_filename=cfg.train_image_path_map_filename,
        images_split_dir=cfg.images_train_subdir, cat_id_to_idx=cat_id_to_idx,
    )

    val_dataset = None
    val_images_info = os.path.join(val_dir, cfg.images_info_filename)
    if os.path.isfile(val_images_info):
        val_dataset = build_split_dataset(
            cfg, val_dir, is_train=False,
            image_path_map_filename=cfg.val_image_path_map_filename,
            images_split_dir=cfg.images_val_subdir, cat_id_to_idx=cat_id_to_idx,
        )
    else:
        print(f"[Data][Notice] Không tìm thấy '{val_images_info}' - bỏ qua val_loader.")

    def _dl_kwargs(is_train):
        kwargs = dict(
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle if is_train else False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last if is_train else False,
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        if cfg.num_workers > 0:
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return kwargs

    train_kwargs = _dl_kwargs(is_train=True)
    for key in ("batch_size", "shuffle", "drop_last"):
        train_kwargs.pop(key)
    train_loader = DataLoader(train_dataset, batch_sampler=EpochBatchSampler(
        train_dataset, cfg.batch_size, cfg.shuffle, cfg.drop_last, cfg.seed), **train_kwargs)
    val_loader = DataLoader(val_dataset, **_dl_kwargs(is_train=False)) if val_dataset is not None else None

    print(f"[Data] Train: {len(train_dataset):,} ảnh | "
          f"Val: {len(val_dataset) if val_dataset else 0:,} ảnh | num_classes={num_classes}")

    return train_loader, val_loader, classes, num_classes

if __name__ == "__main__":
    cfg = TrainConfig()
    train_loader, val_loader, classes, num_classes = build_dataloaders(cfg)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for batch_idx, (images, targets) in enumerate(val_loader):
        num_show = min(4, len(images))
        # targets: list[dict] độ dài batch_size -> {"boxes": (N,4) [PIXEL] xyxy, "labels": (N,)}
        fig, axes = plt.subplots(1, num_show, figsize=(6 * num_show, 6))
        if num_show == 1:
            axes = [axes]

        for ax, image, target in zip(axes, images[:num_show], targets[:num_show]):
            img = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)  # CHW -> HWC
            ax.imshow(img)
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1, str(int(label)), color="white", fontsize=9,
                         bbox=dict(facecolor="red", alpha=0.7, pad=1))
            ax.set_title(f"{len(boxes)} objects")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        if batch_idx >= 5:
            break
