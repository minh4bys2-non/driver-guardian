"""UL-DD: lấy cửa sổ 10 giây, 5 FPS, đầu ra [B, T, 3, H, W]."""
import csv
import logging
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ai.LSTM.src import augment

class RawVideoFramesDataset(Dataset):
    def __init__(self, dataset_dir, split="train", window_sec=10, target_fps=5, img_size=480, augmenter=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        if not np.isfinite(window_sec) or not np.isfinite(target_fps) or window_sec <= 0 or target_fps <= 0:
            raise ValueError("window_sec and target_fps must be positive")

        if type(img_size) is not int or img_size <= 0:
            raise ValueError("img_size must be a positive integer")
        if round(window_sec * target_fps) < 1:
            raise ValueError("window_sec * target_fps must yield at least one frame")
        self.root, self.split = Path(dataset_dir), split
        self.window_sec, self.target_fps, self.img_size = float(window_sec), float(target_fps), int(img_size)
        self.num_frames = round(self.window_sec * self.target_fps)
        self.augmenter = augmenter if split == "train" else None
        self._frame_counts = {}
        self.samples = self._filter_samples(self._discover())
        if not self.samples:
            raise ValueError(f"No videos found: {self.root / split}")

    def _discover(self):
        manifest = self.root / "dataset_merged_split.csv"
        if manifest.is_file():
            with manifest.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                schemas = (("processed_path", "binary_label", "clip_id"), ("rel_path", "label", "video_id"))
                schema = next((keys for keys in schemas if set((*keys, "split")) <= set(columns)), None)
                if schema is None:
                    raise ValueError(f"Invalid dataset manifest {manifest}: missing columns for supported schemas "
                                     f"{schemas}, each with split; found {columns}")
                path_key, label_key, id_key = schema
                rows = [r for r in reader if r["split"].strip().lower() == self.split]
            samples = [{"path": self.root / r[path_key].replace("\\", "/"), "label": int(r[label_key]),
                        "video_id": r[id_key], "subject_id": (r.get("subject_id") or "").strip() or None,
                        "source_video_id": (r.get("source_video_id") or r.get("orig_path") or "").strip().replace("\\", "/") or None,
                        "original_label": r.get("original_label")}
                       for r in rows]
        else:
            samples = []
            for label, folder in ((0, "0_alert"), (1, "1_drowsy")):
                for path in sorted((self.root / self.split / folder).rglob("*")):
                    if not path.is_file() or path.suffix.lower() != ".mp4":
                        continue
                    match = re.match(r"^(uldd_sub[A-Z])_", path.stem)
                    samples.append({"path": path, "label": label, "video_id": path.stem,
                                    "subject_id": match.group(1) if match else None,
                                    "source_video_id": path.stem.rsplit("_w", 1)[0], "original_label": None})

        for sample in samples:
            if sample["label"] not in (0, 1):
                raise ValueError(f"Invalid binary label: {sample['label']}")
            if not sample["path"].is_file():
                raise FileNotFoundError(sample["path"])
        return samples

    def _filter_samples(self, samples):
        valid, short, low_fps, invalid = [], 0, 0, 0
        for sample in samples:
            cap = cv2.VideoCapture(str(sample["path"]))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            finally:
                cap.release()

            if not np.isfinite(fps) or fps <= 0 or total <= 0:
                invalid += 1
                continue
            short += total / fps < self.window_sec
            low_fps += fps < self.target_fps
            if total / fps < self.window_sec or fps < self.target_fps:
                continue
            valid.append(sample)

        if len(valid) < len(samples):
            logging.getLogger("ai.LSTM").info(f"[{self.split}] Skipped {len(samples) - len(valid)} videos "
                  f"(short: {short}, low FPS: {low_fps}, invalid: {invalid})")
        return valid

    def __len__(self):
        return len(self.samples)

    def _read_window(self, path):
        try:
            return self._decode_window(path, self._frame_counts.get(path))
        except OSError:
            if path in self._frame_counts:
                raise
            cap = cv2.VideoCapture(str(path))
            total = 0
            try:
                reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                while cap.read()[0]:
                    total += 1
            finally:
                cap.release()
            if total == 0 or total >= reported:
                raise
            logging.getLogger("ai.LSTM").warning(
                "Video frame count mismatch: metadata=%s, decoded=%s; retrying window: %s",
                reported, total, path)
            self._frame_counts[path] = total
            return self._decode_window(path, total)

    def _decode_window(self, path, total=None):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if total is None:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid FPS: {path}")

            if fps < self.target_fps:
                raise ValueError(f"FPS below {self.target_fps}: {path}")
            window_frames = int(np.ceil(self.window_sec * fps))
            if total < window_frames:
                raise ValueError(f"Video shorter than {self.window_sec}s: {path}")

            max_start = total - window_frames
            if self.split == "train":
                alpha = 2.0
                start = round(max_start * torch.rand(()).item() ** (1 / alpha))
            else:
                start = max_start // 2
            indices = start + np.rint(np.arange(self.num_frames) * fps / self.target_fps).astype(int)
            if indices[-1] >= start + window_frames:
                raise ValueError(f"target_fps exceeds video sampling capacity: {path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames, pos = [], start
            for idx in indices:
                while pos < idx:
                    if not cap.grab():
                        raise OSError(f"Cannot skip frame {pos}: {path}")
                    pos += 1
                ok, frame = cap.read()
                if not ok:
                    raise OSError(f"Cannot decode frame {idx}: {path}")
                pos += 1
                frame = augment.letterbox(frame, new_size=self.img_size)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames, fps, start / fps, total / fps
        finally:
            cap.release()

    def __getitem__(self, index):
        sample = self.samples[index]
        frames, fps, start_sec, duration = self._read_window(sample["path"])
        if self.augmenter is not None:
            seed = int(torch.randint(2**31, (1,)).item())
            frames = self.augmenter.apply_sequence(frames, seed=seed)

        frames = torch.from_numpy(np.ascontiguousarray(np.stack(frames).transpose(0, 3, 1, 2)))
        meta = {"video_id": sample["video_id"], "video_path": str(sample["path"]),
                "subject_id": sample["subject_id"], "source_video_id": sample["source_video_id"],
                "original_label": sample["original_label"], "split": self.split,
                "fps": fps, "start_sec": start_sec, "end_sec": start_sec + self.window_sec,
                "duration_s": duration}
        return frames, torch.tensor(sample["label"], dtype=torch.long), torch.tensor(self.num_frames), meta

def collate_video_frames(batch):
    frames, labels, lengths, meta = zip(*batch)
    return torch.stack(frames), torch.stack(labels), torch.stack(lengths), list(meta)

def _seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)

def build_raw_video_dataloaders(dataset_dir, batch_size=4, num_workers=2, img_size=480,
                                window_sec=10, target_fps=5, augmenter=None,
                                use_augmentation=True, pin_memory=True):
    if use_augmentation:
        augmenter = augmenter if augmenter is not None else augment.get_video_augmenter()
    else:
        augmenter = None

    datasets = {split: RawVideoFramesDataset(dataset_dir, split, window_sec, target_fps, img_size, augmenter)
                for split in ("train", "val", "test")}
    for key in ("subject_id", "source_video_id", "path"):
        groups = [{str(s[key]).strip() if key != "path" else s[key].resolve()
                   for s in ds.samples if s[key]} for ds in datasets.values()]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Data leakage detected between train/val/test: {key}")

    def make_loader(split):
        return DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False,
                          collate_fn=collate_video_frames, worker_init_fn=_seed_worker)

    return make_loader("train"), make_loader("val"), make_loader("test")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    DATASET_DIR = "/home/tranmanhduy/Workspace/dataE/UL-DD_Processed"  # Đổi thành đường dẫn dataset của bạn
    train_loader, val_loader, test_loader = build_raw_video_dataloaders(
        DATASET_DIR, batch_size=1, num_workers=0, img_size=480,
        window_sec=10, target_fps=5, use_augmentation=True,
    )
    print("Dataset sizes:", *(len(loader.dataset) for loader in (train_loader, val_loader, test_loader)))
    frames, labels, lengths, metas = next(iter(train_loader))
    print("Frames:", frames.shape, frames.dtype, "| Label:", labels[0].item(), "| Metadata:", metas[0])

    video = frames[0].permute(0, 2, 3, 1).numpy()  # [50, H, W, 3], RGB uint8
    fig, ax = plt.subplots()
    image = ax.imshow(video[0])
    ax.axis("off")
    fig.tight_layout()

    def update(i):
        image.set_data(video[i])
        ax.set_title(f"{metas[0]['video_id']} | Label {labels[0].item()} | {i / 5:.1f}s")
        return (image,)

    animation = FuncAnimation(fig, update, frames=len(video), interval=1000 / 5, repeat=True)
    plt.show()
