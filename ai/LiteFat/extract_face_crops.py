import os
import sys
import time
import hashlib
from pathlib import Path
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from dataset import fast_read_video_frames
from ultis import crop_faces_from_boxes

def process_single_video(task):
    key, target_indices, crop_boxes, dataset_dir, crop_cache_dir, stride, face_size = task
    try:
        cache_key = hashlib.md5(key.encode()).hexdigest()
        out_file = Path(crop_cache_dir) / f"{cache_key}.pt"
        if out_file.is_file() and out_file.stat().st_size > 1000:
            return True, key, "cached"

        video_path = Path(dataset_dir) / key
        if not video_path.is_file():
            return False, key, f"Video not found: {video_path}"

        v_pos = np.arange(0, len(target_indices), stride)
        sel_indices = target_indices[v_pos]
        if isinstance(sel_indices, torch.Tensor):
            sel_indices = sel_indices.numpy()
        sel_boxes = crop_boxes[v_pos]
        if isinstance(sel_boxes, torch.Tensor):
            sel_boxes = sel_boxes.numpy()

        raw_frames, _, _ = fast_read_video_frames(video_path, sel_indices, letterbox_size=(640, 640))
        face_tensor = crop_faces_from_boxes(raw_frames, sel_boxes, face_size=face_size)

        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        uint8_crops = ((face_tensor * norm_std + norm_mean).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        tmp_file = Path(crop_cache_dir) / f"{cache_key}.tmp_{os.getpid()}"
        torch.save(uint8_crops, tmp_file)
        tmp_file.replace(out_file)
        return True, key, "extracted"
    except Exception as e:
        return False, key, str(e)

def extract_all(
    dataset_dir="/home/riftuser/train/dataset/data-processed-v3/data_processed",
    cache_dir="/home/riftuser/train/dataset/cache-dataset-processed-v3/LiteFat/cache_pt",
    stride=8,
    face_size=(224, 224),
    num_workers=4,
    max_samples=None
):
    out_dir = Path(cache_dir) / f"face_crops_stride{stride}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load all items from cache .pt
    all_samples = {}
    for split in ["train", "val", "test"]:
        pt_path = Path(cache_dir) / f"{split}.pt"
        if pt_path.is_file():
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            items = data.get("samples", data)
            for k, v in items.items():
                if k not in all_samples:
                    all_samples[k] = v

    print(f"Total unique cached items found: {len(all_samples)}")

    tasks = []
    for key, item in all_samples.items():
        t_indices = item.get("target_indices")
        if t_indices is None and "meta" in item:
            t_indices = item["meta"].get("target_indices")
        cb = item.get("crop_boxes")
        if t_indices is not None and cb is not None:
            tasks.append((key, t_indices, cb, dataset_dir, str(out_dir), stride, face_size))

    if max_samples is not None:
        tasks = tasks[:max_samples]

    print(f"Total tasks to process: {len(tasks)} with {num_workers} workers (stride={stride})...")
    t0 = time.time()
    extracted_cnt = 0
    cached_cnt = 0
    error_cnt = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video, t): t[0] for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting face crops")
        for f in pbar:
            ok, key, msg = f.result()
            if ok:
                if msg == "cached":
                    cached_cnt += 1
                else:
                    extracted_cnt += 1
            else:
                error_cnt += 1
                print(f"[Error] {key}: {msg}")
            pbar.set_postfix({"ext": extracted_cnt, "hit": cached_cnt, "err": error_cnt})

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s! Extracted: {extracted_cnt}, Already cached: {cached_cnt}, Errors: {error_cnt}")

if __name__ == '__main__':
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_s = int(sys.argv[2]) if len(sys.argv) > 2 else None
    extract_all(stride=stride, max_samples=max_s)
