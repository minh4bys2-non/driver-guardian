import csv

import cv2
import numpy as np
import pytest

from ai.LSTM.src.dataloader import RawVideoFramesDataset, build_raw_video_dataloaders


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 24))
    assert writer.isOpened()
    for i in range(18):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 2] = 100 + i * 5
        writer.write(frame)
    writer.release()
    return path


def manifest(root, rows):
    with (root / "dataset_merged_split.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("processed_path", "binary_label", "clip_id", "split",
                                                "subject_id", "source_video_id"))
        writer.writeheader()
        writer.writerows(rows)


def row(video, split="train", subject="A", source="source", label=1):
    return dict(processed_path=video.name, binary_label=label, clip_id=f"clip_{split}",
                split=split, subject_id=subject, source_video_id=source)


@pytest.mark.parametrize("missing", ["processed_path", "binary_label", "clip_id", "split"])
def test_manifest_missing_columns_reports_schema_before_reading_video(tmp_path, missing, monkeypatch):
    columns = [name for name in ("processed_path", "binary_label", "clip_id", "split") if name != missing]
    path = tmp_path / "dataset_merged_split.csv"
    path.write_text(",".join(columns) + "\n")
    def unexpected_capture(*args):
        pytest.fail("Video decoding must not start with an invalid manifest")
    monkeypatch.setattr(cv2, "VideoCapture", unexpected_capture)
    with pytest.raises(ValueError, match="missing columns") as error:
        RawVideoFramesDataset(tmp_path)
    assert missing in str(error.value) and str(path) in str(error.value)
    assert f"found {columns}" in str(error.value)


def test_manifest_relative_path_rgb_and_train_only_augmentation(tmp_path, video):
    manifest(tmp_path, [row(video, split=split) for split in ("train", "val", "test")])
    class Augmenter:
        calls = 0
        def apply_sequence(self, frames, seed):
            self.calls += 1
            return frames
    augmenter = Augmenter()
    for split in ("train", "val", "test"):
        ds = RawVideoFramesDataset(tmp_path, split, window_sec=2, target_fps=3, img_size=32, augmenter=augmenter)
        frames, label, length, meta = ds[0]
        assert frames[0, 0, 16, 16] > frames[0, 2, 16, 16] + 50
        assert label.item() == 1 and length.item() == 6
        assert meta["video_path"] == str(video) and meta["subject_id"] == "A"
        assert 0 <= meta["start_sec"] <= 1 and meta["end_sec"] <= 3
    assert augmenter.calls == 1


@pytest.mark.parametrize("reason", ["subject", "source", "path"])
def test_rejects_cross_split_leakage(tmp_path, video, reason):
    rows = []
    for split, subject in zip(("train", "val", "test"), "ABC"):
        target = tmp_path / f"{split}.mp4"
        target.write_bytes(video.read_bytes())
        rows.append(row(target if reason != "path" else video, split,
                        subject="A" if reason == "subject" else "",
                        source="shared" if reason == "source" else split))
    manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="[Ll]eakage"):
        build_raw_video_dataloaders(tmp_path, window_sec=2, target_fps=3, num_workers=0, use_augmentation=False)


@pytest.mark.parametrize("kwargs", [{"window_sec": 0.01, "target_fps": 1},
                                    {"img_size": 0}, {"target_fps": float("nan")}])
def test_rejects_invalid_sampling_before_reading(tmp_path, kwargs):
    with pytest.raises(ValueError, match="window|FPS|img_size|frames"):
        RawVideoFramesDataset(tmp_path, **kwargs)


@pytest.mark.parametrize("label", [-1, 2])
def test_rejects_nonbinary_manifest_labels(tmp_path, video, label):
    manifest(tmp_path, [row(video, label=label)])
    with pytest.raises(ValueError, match="label"):
        RawVideoFramesDataset(tmp_path, window_sec=2, target_fps=3)


def test_filters_short_low_fps_and_corrupt_videos(tmp_path, video):
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"not a video")
    manifest(tmp_path, [row(video), row(corrupt)])
    ds = RawVideoFramesDataset(tmp_path, window_sec=2, target_fps=3)
    assert len(ds) == 1
    for kwargs in ({"window_sec": 4}, {"target_fps": 7, "window_sec": 2}):
        with pytest.raises(ValueError, match="No videos"):
            RawVideoFramesDataset(tmp_path, **kwargs)


def test_overreported_frame_count_retries_and_caches_decoded_length(tmp_path, video, monkeypatch):
    import torch

    capture = cv2.VideoCapture

    class OverreportedCapture:
        def __init__(self, path):
            self.cap = capture(path)

        def get(self, prop):
            return 30 if prop == cv2.CAP_PROP_FRAME_COUNT else self.cap.get(prop)

        def __getattr__(self, name):
            return getattr(self.cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", OverreportedCapture)
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(1.0))
    manifest(tmp_path, [row(video)])
    ds = RawVideoFramesDataset(tmp_path, window_sec=2, target_fps=3, img_size=32)
    for _ in range(2):
        frames, _, length, meta = ds[0]
        assert frames.shape == (6, 3, 32, 32) and length.item() == 6
        assert meta["duration_s"] == 3 and meta["start_sec"] == 1
        assert meta["end_sec"] == 3
    assert ds._frame_counts[video] == 18


def test_build_loaders_accepts_disjoint_manifest(tmp_path, video):
    rows = []
    for split, subject in zip(("train", "val", "test"), "ABC"):
        target = tmp_path / f"{split}.mp4"
        target.write_bytes(video.read_bytes())
        rows.append(row(target, split, subject, source=split))
    manifest(tmp_path, rows)
    loaders = build_raw_video_dataloaders(tmp_path, batch_size=1, num_workers=0, img_size=32,
                                          window_sec=2, target_fps=3, use_augmentation=False, pin_memory=False)
    for loader, split in zip(loaders, ("train", "val", "test")):
        frames, labels, lengths, metas = next(iter(loader))
        assert frames.shape == (1, 6, 3, 32, 32)
        assert labels.tolist() == [1] and lengths.tolist() == [6]
        assert metas[0]["split"] == split


def test_training_with_real_video_loaders(tmp_path, video):
    from ai.LSTM.src.config import TrainConfig
    from ai.LSTM.src.training import train
    from ai.LSTM.src.checkpoint import load_model
    rows = []
    for split, subject in zip(("train", "val", "test"), "ABC"):
        target = tmp_path / f"{split}.mp4"
        target.write_bytes(video.read_bytes())
        rows.append(row(target, split, subject, source=split))
    manifest(tmp_path, rows)
    cfg = TrainConfig(dataset_dir=str(tmp_path), save_dir=str(tmp_path / "run"),
                      backbone_mode="scratch", backbone_checkpoint=None,
                      device="cpu", amp=False, num_workers=0, epochs=1, warmup_epochs=0,
                      backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                      input_dim=8, hidden_dim=8, image_size=64, window_sec=2,
                      target_fps=1, batch_size=1, chunk_size=2, use_augmentation=True)
    _, metrics = train(cfg)
    restored, checkpoint = load_model(tmp_path / "run/best.pt")
    assert metrics["n_samples"] == 1
    assert checkpoint["epoch"] == 1 and checkpoint["config"]["target_fps"] == 1
    assert (tmp_path / "run/train.log").is_file()

    from dataclasses import replace
    import torch
    from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
    from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import NeuralNetwork
    network = NeuralNetwork(tmp_path / "run/best.pt", device="cpu", config=replace(
        DEFAULT_CONFIG, neural_size=cfg.image_size, neural_sec=cfg.window_sec,
        neural_fps=cfg.target_fps))
    frames = RawVideoFramesDataset(tmp_path, "test", img_size=cfg.image_size,
                                  window_sec=cfg.window_sec, target_fps=cfg.target_fps)[0][0]
    with torch.inference_mode():
        expected = restored(frames[None]).softmax(-1)[0, 1].item()
    assert network(frames) == pytest.approx(expected, abs=1e-6)


def test_training_and_datasystem_sample_the_same_50_rgb_frames(tmp_path):
    from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import read_video
    from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import letterbox

    folder = tmp_path / "val/0_alert"
    folder.mkdir(parents=True)
    path = folder / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    assert writer.isOpened()
    try:
        for i in range(300):
            writer.write(np.full((24, 32, 3), (i % 256, 30, 200), np.uint8))
    finally:
        writer.release()
    frames, _, length, _ = RawVideoFramesDataset(tmp_path, "val", img_size=224)[0]
    sampled = np.stack([letterbox(frame.transpose(1, 2, 0), 224).transpose(2, 0, 1)
                        for i, (frame, _) in enumerate(read_video(str(path), 30)) if i % 6 == 0])
    assert length.item() == 50 and sampled.shape == (50, 3, 224, 224)
    np.testing.assert_array_equal(frames.numpy(), sampled)


def server_manifest(root, rows):
    with (root / 'dataset_merged_split.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=('video_id', 'dataset', 'split', 'label', 'label_name',
                                                'rel_path', 'orig_path', 'subject_id'))
        writer.writeheader()
        writer.writerows(rows)


def test_server_manifest_loads_processed_clip_and_preserves_metadata(tmp_path, video):
    target = tmp_path / 'test/0_alert/clip.mp4'
    target.parent.mkdir(parents=True)
    target.write_bytes(video.read_bytes())
    server_manifest(tmp_path, [dict(video_id='clip0', dataset='UL-DD', split='test', label=0,
                                   label_name='0_alert', rel_path='test\\0_alert\\clip.mp4',
                                   orig_path='UL-DD\\Video_Data\\N\\A\\N_R3D_A.mp4', subject_id='subject_N')])
    ds = RawVideoFramesDataset(tmp_path, 'test', window_sec=2, target_fps=3, img_size=32)
    frames, label, length, meta = ds[0]
    assert frames.shape == (6, 3, 32, 32)
    assert label.item() == 0 and length.item() == 6
    assert meta['video_id'] == 'clip0' and meta['video_path'] == str(target)
    assert meta['subject_id'] == 'subject_N'
    assert meta['source_video_id'] == 'UL-DD/Video_Data/N/A/N_R3D_A.mp4'


@pytest.mark.parametrize('shared_source', [False, True])
def test_server_manifest_split_leakage_uses_original_video(tmp_path, video, shared_source):
    rows = []
    for split in ('train', 'val', 'test'):
        target = tmp_path / f'{split}.mp4'
        target.write_bytes(video.read_bytes())
        rows.append(dict(video_id=split, split=split, label=1, rel_path=target.name,
                         orig_path='N/A/source.mp4' if shared_source else f'N/A/{split}.mp4'))
    server_manifest(tmp_path, rows)
    options = dict(window_sec=2, target_fps=3, img_size=32, num_workers=0, use_augmentation=False)
    if shared_source:
        with pytest.raises(ValueError, match='source_video_id'):
            build_raw_video_dataloaders(tmp_path, **options)
    else:
        loaders = build_raw_video_dataloaders(tmp_path, **options)
        assert [len(loader.dataset) for loader in loaders] == [1, 1, 1]
