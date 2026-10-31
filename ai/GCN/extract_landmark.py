# File: extract_landmark.py
"""
Kịch bản trích xuất trước 68 điểm mốc khuôn mặt (Landmarks) từ video bằng GCNDataset.
Lưu kết quả thành các tệp HDF5 riêng biệt (train.h5, val.h5, test.h5).

Tính năng chính:
1. Tận dụng trực tiếp GCNDataset trong dataset.py và TrainConfig trong config.py.
2. Tách riêng 3 tệp train.h5, val.h5, test.h5 với cấu trúc /samples/{sample_key}.
3. Cơ chế khôi phục (Resume): Tự động phát hiện và bỏ qua các video đã trích xuất thành công,
   dọn dẹp các bản ghi ghi dở khi bị tắt đột ngột, tiếp tục từ mẫu chưa xử lý.
4. Cơ chế ngắt an toàn (Graceful Shutdown): Bắt tín hiệu Ctrl+C (SIGINT) để hoàn tất ghi mẫu hiện tại,
   flush bộ nhớ đệm và đóng file HDF5 an toàn trước khi thoát.
5. Hỗ trợ chạy dòng lệnh (CLI với argparse), chạy trực tiếp qua hàm Python, hoặc kiểm tra tệp HDF5 (--verify).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Cấu hình UTF-8 cho console trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import GCNDataset

# Thiết lập logger
logger = logging.getLogger("DriverGuardian.ExtractLandmark")

# Cờ ngắt toàn cục khi nhận tín hiệu SIGINT (Ctrl+C)
_INTERRUPTED = False


def _signal_handler(signum: int, frame: Any) -> None:
    """Xử lý tín hiệu dừng chương trình (Ctrl+C hoặc SIGTERM) an toàn."""
    global _INTERRUPTED
    if not _INTERRUPTED:
        logger.warning(
            "\n[SIGINT] Đã nhận tín hiệu dừng (Ctrl+C)! Đang hoàn tất ghi mẫu hiện tại và đóng tệp HDF5 an toàn..."
        )
        logger.warning("Vui lòng đợi giây lát hoặc nhấn Ctrl+C lần nữa để buộc dừng ngay.")
        _INTERRUPTED = True
    else:
        logger.error("\n[BUỘC DỪNG] Đã nhận tín hiệu lần 2. Dừng chương trình ngay lập tức!")
        sys.exit(1)


def register_signal_handlers() -> None:
    """Đăng ký bắt tín hiệu ngắt an toàn."""
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception as e:
        logger.debug(f"Không thể gán signal.SIGINT: {e}")
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception as e:
            logger.debug(f"Không thể gán signal.SIGTERM: {e}")


def configure_logging(log_file: Optional[Path | str] = None, verbose: bool = False) -> None:
    """Cấu hình logging đồng thời ra Console và tệp log (nếu chỉ định)."""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def seed_everything(seed: int = 42) -> None:
    """Cố định seed để đảm bảo tính nhất quán khi lấy mẫu cửa sổ train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_h5_key(video_id: str, index: int) -> str:
    """
    Chuẩn hóa tên định danh group trong file HDF5.
    Loại bỏ các ký tự '/' và '\\' vì trong HDF5 chúng đóng vai trò phân tách nhóm (nested groups).
    """
    clean_id = str(video_id).strip().replace("/", "_").replace("\\", "_")
    if clean_id:
        return clean_id
    return f"sample_{index:06d}"


def scan_existing_samples(samples_grp: h5py.Group) -> Tuple[Set[str], List[str]]:
    """
    Quét nhóm /samples để tìm các mẫu đã trích xuất hợp lệ và các nhóm bị lỗi dở dang.
    Một mẫu được coi là hợp lệ khi:
      - Là h5py.Group
      - Chứa dataset 'landmarks'
      - Dataset có ndim == 3 (C=2, T>0, V=68)
    """
    valid_keys: Set[str] = set()
    corrupted_keys: List[str] = []

    for key in samples_grp.keys():
        item = samples_grp[key]
        if isinstance(item, h5py.Group) and "landmarks" in item:
            ds = item["landmarks"]
            # shape phải là (2, T, 68)
            if ds.ndim == 3 and ds.shape[0] == 2 and ds.shape[2] == 68 and ds.shape[1] > 0:
                valid_keys.add(key)
                continue
        corrupted_keys.append(key)

    return valid_keys, corrupted_keys


def extract_split_to_h5(
    split: str,
    output_dir: Path | str,
    cfg: Optional[TrainConfig] = None,
    resume: bool = True,
    flush_every: int = 20,
    compression: str = "gzip",
    compression_opts: int = 4,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Trích xuất toàn bộ 68 điểm mốc cho một split ('train', 'val', 'test') và lưu vào file {split}.h5.

    Tham số:
        split: Tên tập dữ liệu ('train', 'val', hoặc 'test').
        output_dir: Thư mục chứa các tệp .h5.
        cfg: Đối tượng TrainConfig chứa các tham số cấu hình dataset.
        resume: Bật/tắt chế độ khôi phục (tiếp tục từ các mẫu đã xong).
        flush_every: Số mẫu hoàn thành trước mỗi lần commit dữ liệu xuống đĩa (h5f.flush()).
        compression: Chuẩn nén HDF5 (mặc định 'gzip').
        compression_opts: Mức độ nén (1-9, mặc định 4).
        show_progress: Hiển thị thanh tiến trình tqdm.

    Trả về:
        Dictionary tổng kết kết quả trích xuất.
    """
    global _INTERRUPTED
    if cfg is None:
        cfg = TrainConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    h5_file_path = output_path / f"{split}.h5"

    seed_everything(cfg.seed)
    register_signal_handlers()

    logger.info(f"=" * 60)
    logger.info(f"BẮT ĐẦU TRÍCH XUẤT SPLIT: [{split.upper()}]")
    logger.info(f"Tệp đầu ra     : {h5_file_path}")
    logger.info(f"Dataset root   : {cfg.dataset_dir}")
    logger.info(f"Time interval  : {cfg.time_interval}s")
    logger.info(f"Window sec     : {cfg.window_sec}s")
    logger.info(f"Letterbox size : {cfg.letterbox_size}")
    logger.info(f"Chế độ Resume  : {'BẬT' if resume else 'TẮT'}")
    logger.info(f"=" * 60)

    # Khởi tạo GCNDataset
    dataset = GCNDataset(
        dataset_dir=cfg.dataset_dir,
        split=split,
        time_interval=cfg.time_interval,
        window_sec=cfg.window_sec,
        manifest_filename=cfg.manifest_filename,
        landmark_model_path=cfg.landmark_model_path,
        letterbox_size=cfg.letterbox_size,
    )

    total_samples = len(dataset)
    if total_samples == 0:
        logger.warning(f"Không có mẫu nào trong split '{split}'. Bỏ qua.")
        return {
            "split": split,
            "total": 0,
            "completed": 0,
            "new_extracted": 0,
            "failed": 0,
            "path": str(h5_file_path),
        }

    # Quản lý mở file HDF5
    file_mode = "a" if (resume and h5_file_path.is_file()) else "w"
    h5f = h5py.File(h5_file_path, mode=file_mode)

    completed_keys: Set[str] = set()

    try:
        # Nhóm /samples
        if "samples" not in h5f:
            samples_grp = h5f.create_group("samples")
        else:
            samples_grp = h5f["samples"]

        # Kiểm tra và khôi phục
        if file_mode == "a":
            valid_keys, corrupted_keys = scan_existing_samples(samples_grp)
            completed_keys.update(valid_keys)

            if corrupted_keys:
                logger.warning(
                    f"Phát hiện {len(corrupted_keys)} nhóm mẫu ghi dở/lỗi ở phiên trước. Đang dọn dẹp..."
                )
                for bad_key in corrupted_keys:
                    del samples_grp[bad_key]
                h5f.flush()

            logger.info(
                f"Trạng thái khôi phục: Đã có sẵn {len(completed_keys)}/{total_samples} mẫu hoàn tất."
            )
        else:
            # Ghi thuộc tính gốc (root attributes)
            h5f.attrs["split"] = split
            h5f.attrs["time_interval"] = float(cfg.time_interval)
            h5f.attrs["window_sec"] = float(cfg.window_sec) if cfg.window_sec is not None else -1.0
            h5f.attrs["letterbox_size"] = int(cfg.letterbox_size)
            h5f.attrs["dataset_dir"] = str(cfg.dataset_dir)
            h5f.attrs["manifest_filename"] = str(cfg.manifest_filename)
            h5f.attrs["total_samples"] = int(total_samples)
            h5f.attrs["created_at"] = datetime.now().isoformat()

        # Tiến hành trích xuất
        new_extracted_count = 0
        failed_samples: List[Dict[str, Any]] = []

        pbar = tqdm(
            range(total_samples),
            desc=f"Extracting [{split}]",
            initial=len(completed_keys),
            total=total_samples,
            disable=not show_progress,
            dynamic_ncols=True,
        )

        for idx in range(total_samples):
            if _INTERRUPTED:
                logger.warning(f"Đã tạm dừng trích xuất split '{split}' do nhận tín hiệu ngắt.")
                break

            sample_meta = dataset.samples[idx]
            video_id = str(sample_meta.get("video_id", f"sample_{idx}"))
            key = make_h5_key(video_id, idx)

            # Nếu mẫu đã hoàn tất hợp lệ, bỏ qua
            if key in completed_keys:
                continue

            # Trích xuất dữ liệu qua GCNDataset
            try:
                landmarks_tensor, label_tensor, meta = dataset[idx]
            except Exception as exc:
                err_msg = str(exc)
                logger.error(
                    f"Lỗi khi trích xuất mẫu idx={idx} (video_id='{video_id}', path='{sample_meta.get('path')}'): {err_msg}"
                )
                failed_samples.append({
                    "idx": idx,
                    "video_id": video_id,
                    "path": str(sample_meta.get("path")),
                    "error": err_msg,
                })
                pbar.update(1)
                continue

            # Chuyển đổi sang numpy float32: shape (2, T, 68)
            lm_np = landmarks_tensor.detach().cpu().numpy().astype(np.float32)

            # Nếu group đã tồn tại do chưa hoàn chỉnh, xóa đi trước khi tạo mới
            if key in samples_grp:
                del samples_grp[key]

            sample_grp = samples_grp.create_group(key)
            sample_grp.create_dataset(
                "landmarks",
                data=lm_np,
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_opts,
                chunks=True,
            )

            # Lưu metadata chi tiết vào thuộc tính của sample group
            sample_grp.attrs["label"] = int(label_tensor.item())
            sample_grp.attrs["video_id"] = str(meta.get("video_id", video_id))
            sample_grp.attrs["path"] = str(meta.get("path", sample_meta.get("path", "")))
            sample_grp.attrs["fps"] = float(meta.get("fps", 0.0))
            sample_grp.attrs["num_sampled_frames"] = int(meta.get("num_sampled_frames", lm_np.shape[1]))
            sample_grp.attrs["duration_sec"] = float(meta.get("duration_sec", 0.0))
            sample_grp.attrs["time_interval"] = float(meta.get("time_interval", cfg.time_interval))
            sample_grp.attrs["sample_idx"] = int(idx)

            completed_keys.add(key)
            new_extracted_count += 1
            pbar.update(1)

            # Flush định kỳ để đảm bảo dữ liệu ghi vào đĩa
            if new_extracted_count % flush_every == 0:
                h5f.flush()

        pbar.close()

        # Cập nhật thuộc tính tổng quan của file
        h5f.attrs["extracted_samples"] = int(len(samples_grp))
        h5f.attrs["updated_at"] = datetime.now().isoformat()
        h5f.flush()

        # Lưu danh sách mẫu lỗi nếu có
        if failed_samples:
            fail_log_path = output_path / f"failed_samples_{split}.json"
            try:
                with open(fail_log_path, "w", encoding="utf-8") as f:
                    json.dump(failed_samples, f, indent=4, ensure_ascii=False)
                logger.warning(
                    f"Có {len(failed_samples)} mẫu gặp lỗi! Chi tiết lỗi đã lưu tại: {fail_log_path}"
                )
            except Exception as e:
                logger.error(f"Không thể lưu tệp failed_samples: {e}")

        logger.info(
            f"Hoàn thành split '{split}': Tổng {len(completed_keys)}/{total_samples} mẫu "
            f"(Trích xuất mới: {new_extracted_count}, Lỗi: {len(failed_samples)})"
        )

        return {
            "split": split,
            "total": total_samples,
            "completed": len(completed_keys),
            "new_extracted": new_extracted_count,
            "failed": len(failed_samples),
            "path": str(h5_file_path),
        }

    finally:
        try:
            h5f.close()
        except Exception:
            pass


def extract_all_splits(
    splits: Tuple[str, ...] | List[str] = ("train", "val", "test"),
    output_dir: Optional[Path | str] = None,
    cfg: Optional[TrainConfig] = None,
    resume: bool = True,
    flush_every: int = 20,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    Trích xuất tuần tự các split và lưu thành các tệp .h5 tương ứng.
    """
    if cfg is None:
        cfg = TrainConfig()

    if output_dir is None:
        # Mặc định lưu tại E:\LSTM\data_processed\landmarks_h5 hoặc ./landmarks_h5
        candidate_dir = Path(cfg.dataset_dir) / "landmarks_h5"
        if candidate_dir.parent.is_dir():
            output_dir = candidate_dir
        else:
            output_dir = Path(__file__).resolve().parent / "landmarks_h5"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    start_time = time.time()

    for split in splits:
        if _INTERRUPTED:
            logger.warning("Đã dừng tiến trình trước khi chuyển sang split tiếp theo.")
            break

        res = extract_split_to_h5(
            split=split,
            output_dir=output_dir,
            cfg=cfg,
            resume=resume,
            flush_every=flush_every,
            compression=compression,
            compression_opts=compression_opts,
        )
        results[split] = res

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"TỔNG KẾT QUÁ TRÌNH TRÍCH XUẤT (Thời gian: {elapsed:.1f}s)")
    for sp, info in results.items():
        logger.info(
            f" - {sp:5s}: Hoàn thành {info['completed']}/{info['total']} mẫu "
            f"(Mới: {info['new_extracted']}, Lỗi: {info['failed']}) -> {info['path']}"
        )
    logger.info("=" * 60 + "\n")

    return results


def verify_h5_file(h5_path: Path | str, num_inspect: int = 3) -> Dict[str, Any]:
    """
    Kiểm tra tính toàn vẹn của tệp .h5 và in ra báo cáo thống kê.
    """
    p = Path(h5_path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tại: {p}")

    report: Dict[str, Any] = {
        "path": str(p),
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "attributes": {},
        "num_samples": 0,
        "sample_inspection": [],
        "labels_distribution": {},
    }

    print("\n" + "=" * 65)
    print(f"BÁO CÁO KIỂM TRA FILE HDF5: {p.name} ({report['file_size_mb']} MB)")
    print("=" * 65)

    with h5py.File(p, "r") as h5f:
        # Đọc root attributes
        for k, v in h5f.attrs.items():
            report["attributes"][k] = v
            print(f" - Attr: {k:18s} = {v}")

        if "samples" not in h5f:
            print("[CẢNH BÁO] Không tìm thấy group '/samples' trong tệp!")
            return report

        samples_grp = h5f["samples"]
        all_keys = list(samples_grp.keys())
        report["num_samples"] = len(all_keys)
        print(f"\nTổng số mẫu ghi nhận trong /samples: {len(all_keys)}")

        # Thống kê nhãn và kiểm tra mẫu
        label_counts: Dict[int, int] = {}
        for key in all_keys:
            grp = samples_grp[key]
            lbl = grp.attrs.get("label", -1)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        report["labels_distribution"] = label_counts
        print("Phân phối nhãn:", label_counts)

        # Kiểm tra chi tiết một vài mẫu đầu tiên
        print(f"\nKiểm tra chi tiết {min(num_inspect, len(all_keys))} mẫu đầu tiên:")
        for i, key in enumerate(all_keys[:num_inspect]):
            grp = samples_grp[key]
            ds = grp["landmarks"]
            sample_info = {
                "key": key,
                "landmarks_shape": ds.shape,
                "dtype": str(ds.dtype),
                "label": int(grp.attrs.get("label", -1)),
                "video_id": grp.attrs.get("video_id", ""),
                "fps": grp.attrs.get("fps", 0.0),
                "num_sampled_frames": grp.attrs.get("num_sampled_frames", 0),
                "duration_sec": grp.attrs.get("duration_sec", 0.0),
            }
            report["sample_inspection"].append(sample_info)
            print(
                f"  [{i + 1}] Key: {key:25s} | Shape: {ds.shape} | Label: {sample_info['label']} "
                f"| Frames: {sample_info['num_sampled_frames']} | Duration: {sample_info['duration_sec']:.1f}s"
            )

    print("=" * 65 + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ xử lý tham số dòng lệnh."""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Trích xuất trước điểm mốc khuôn mặt (Landmarks) bằng GCNDataset và lưu ra tệp HDF5 (.h5)."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=cfg.dataset_dir,
        help=f"Đường dẫn thư mục dữ liệu video gốc (mặc định: {cfg.dataset_dir})",
    )
    parser.add_argument(
        "--manifest-filename",
        type=str,
        default=cfg.manifest_filename,
        help=f"Tên tệp CSV manifest (mặc định: {cfg.manifest_filename})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(cfg.dataset_dir) / "landmarks_h5"),
        help="Thư mục lưu trữ các tệp train.h5, val.h5, test.h5 (mặc định: <dataset_dir>/landmarks_h5)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Danh sách các split cần trích xuất (mặc định: train val test)",
    )
    parser.add_argument(
        "--time-interval",
        type=float,
        default=cfg.time_interval,
        help=f"Khoảng thời gian lấy mẫu frame (giây) (mặc định: {cfg.time_interval})",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=cfg.window_sec,
        help=f"Thời lượng cửa sổ video cần trích xuất (giây) (mặc định: {cfg.window_sec})",
    )
    parser.add_argument(
        "--letterbox-size",
        type=int,
        default=cfg.letterbox_size,
        help=f"Kích thước letterbox chuẩn hóa khung hình (mặc định: {cfg.letterbox_size})",
    )
    parser.add_argument(
        "--landmark-model-path",
        type=str,
        default=None,
        help="Đường dẫn tệp mô hình MediaPipe face_landmarker.task (mặc định: tự tìm)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Tắt cơ chế khôi phục (ghi đè lại từ đầu tệp HDF5)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Tần suất flush bộ nhớ đệm xuống đĩa (số lượng mẫu, mặc định: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.seed,
        help=f"Seed ngẫu nhiên (mặc định: {cfg.seed})",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Đường dẫn tệp ghi log (mặc định: ghi ra console)",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Kiểm tra nội dung tệp .h5 được chỉ định thay vì trích xuất",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    # Chế độ kiểm tra tệp HDF5
    if args.verify:
        verify_h5_file(args.verify)
        return

    # Khởi tạo TrainConfig tùy biến theo CLI
    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        manifest_filename=args.manifest_filename,
        time_interval=args.time_interval,
        window_sec=args.window_sec,
        letterbox_size=args.letterbox_size,
        landmark_model_path=args.landmark_model_path,
        seed=args.seed,
    )

    extract_all_splits(
        splits=args.splits,
        output_dir=args.output_dir,
        cfg=cfg,
        resume=not args.no_resume,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
