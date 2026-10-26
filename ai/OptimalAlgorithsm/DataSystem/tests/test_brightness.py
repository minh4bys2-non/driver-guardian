from dataclasses import replace

import cv2
import numpy as np
import pytest

from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import balance_brightness, read_video
from ai.OptimalAlgorithsm.DataSystem.normalization import PhysicalNormalizer


def test_dark_grayscale_is_lifted_without_changing_shape_or_dtype():
    frame = np.full((24, 32, 3), 25, np.uint8)
    result = balance_brightness(frame)
    assert result.shape == frame.shape and result.dtype == np.uint8
    assert result.mean() == pytest.approx(70, abs=1)
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
    np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])
    assert np.all(frame == 25)


@pytest.mark.parametrize('value', [0, 70, 128, 255])
def test_black_and_sufficiently_bright_frames_are_unchanged(value):
    frame = np.full((8, 8, 3), value, np.uint8)
    np.testing.assert_array_equal(balance_brightness(frame), frame)


def test_luminance_curve_preserves_order_and_limits_shadow_gain():
    levels = np.arange(61, dtype=np.uint8)
    frame = np.repeat(levels[None, :, None], 3, axis=2)
    result = balance_brightness(frame)
    assert np.all(np.diff(result[0, :, 0].astype(int)) >= 0)
    assert result[0, 0, 0] == 0
    assert np.all(result[0, :, 0] >= levels)
    very_dark = np.full((8, 8, 3), 1, np.uint8)
    assert balance_brightness(very_dark).max() <= 16


def test_color_chroma_is_preserved_within_rounding_error():
    frame = np.full((8, 8, 3), (10, 20, 40), np.uint8)
    result = balance_brightness(frame)
    before = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    after = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    assert after[:, :, 0].mean() > before[:, :, 0].mean()
    np.testing.assert_allclose(after[:, :, 1:], before[:, :, 1:], atol=1)


def test_disabled_correction_preserves_every_pixel():
    frame = np.random.default_rng(42).integers(0, 60, (16, 32, 3), dtype=np.uint8)
    np.testing.assert_array_equal(balance_brightness(frame, target_luma=0), frame)


@pytest.mark.parametrize('target', [-1, 256, float('nan'), float('inf')])
def test_invalid_target_is_rejected_before_opening_video(target):
    with pytest.raises(ValueError, match='target_luma'):
        list(read_video('missing.mp4', target_luma=target))


def test_real_dark_video_preserves_timestamps_and_rgb(tmp_path):
    path = str(tmp_path / 'dark.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(5):
            writer.write(np.full((24, 32, 3), (10, 20, 40), np.uint8))
    finally:
        writer.release()
    original = list(read_video(path, 10, target_luma=0))
    corrected = list(read_video(path, 10))
    assert len(original) == len(corrected) == 10
    assert [t for _, t in original] == [t for _, t in corrected]
    for (raw, _), (actual, _) in zip(original, corrected):
        expected_bgr = balance_brightness(raw.transpose(1, 2, 0)[:, :, ::-1].copy())
        np.testing.assert_array_equal(actual, expected_bgr[:, :, ::-1].transpose(2, 0, 1))
        assert actual[0].mean() > actual[1].mean() > actual[2].mean()
        assert actual.mean() > raw.mean()


def test_brightness_configuration_changes_dataset_identity():
    assert PhysicalNormalizer(DEFAULT_CONFIG).id != PhysicalNormalizer(
        replace(DEFAULT_CONFIG, target_luma=0)).id


@pytest.mark.parametrize('target', [0, 90])
def test_engine_passes_brightness_configuration_to_reader(tmp_path, monkeypatch, target):
    from unittest.mock import Mock
    from ai.OptimalAlgorithsm.DataSystem import engine

    reader = Mock(return_value=iter([]))
    monkeypatch.setattr(engine, 'read_video', reader)
    monkeypatch.setattr(engine, 'NeuralNetwork', Mock())
    monkeypatch.setattr(engine, 'CameraMetrics', Mock())
    pipeline = engine.Engine(tmp_path, config=replace(DEFAULT_CONFIG, target_luma=target))
    assert list(pipeline.process_video('clip.mp4', 0)) == []
    reader.assert_called_once_with(str(tmp_path / 'clip.mp4'), target_fps=30, target_luma=target,
                                   min_duration_sec=30)
