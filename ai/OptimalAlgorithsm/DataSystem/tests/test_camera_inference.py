import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ai.OptimalAlgorithsm import inference


def test_neural_sampling_uses_time_and_resets_after_gap():
    config = replace(inference.DEFAULT_CONFIG, neural_size=8)
    buffer = inference.NeuralBuffer(config)
    for i in range(301):
        buffer.update(np.full((8, 8, 3), i % 256, np.uint8), i / 30)
    assert len(buffer.frames) == 50
    assert [int(frame[0, 0, 0]) for frame in buffer.frames] == [i % 256 for i in range(6, 301, 6)]
    buffer.update(np.full((8, 8, 3), 123, np.uint8), 11)
    assert len(buffer.frames) == 1
    assert np.all(buffer.frames[0] == 123)


def test_neural_sampling_does_not_use_future_frame():
    buffer = inference.NeuralBuffer(replace(inference.DEFAULT_CONFIG, neural_size=8))
    buffer.update(np.zeros((8, 8, 3), np.uint8), 0)
    buffer.update(np.ones((8, 8, 3), np.uint8), .21)
    assert np.all(buffer.frames[-1] == 0)


@pytest.mark.parametrize("method", ["weighted_mean", "noisy_or", "cooccurrence"])
def test_fusion_matches_training(method, tmp_path):
    normalizer = SimpleNamespace(scales=dict.fromkeys(inference.FEATURES, 1.0))
    report = dict(input_schema=dict(features=inference.FEATURES,
                                   feature_encoding="physical_v1",
                                   normalization_scales=normalizer.scales),
                  fitness_method=method,
                  selected_model=dict(model="candidate", candidate=[.8] * 8 + [.5]))
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(report))
    fusion = inference.load_fusion(path, normalizer)
    sample = dict.fromkeys(inference.FEATURES, .7)
    expected = inference.Fitness([[.7] * 8]).predict([.8] * 8 + [.5], method)[0]
    assert inference.classify(sample, fusion) == expected
    report.pop("input_schema")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="schema"):
        inference.load_fusion(path, normalizer)


@pytest.mark.parametrize("opened", [True, False])
def test_camera_failure_releases_resources(monkeypatch, opened):
    capture = Mock()
    capture.isOpened.return_value = opened
    capture.read.return_value = False, None
    camera = Mock()
    monkeypatch.setattr(inference.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(inference, "CameraMetrics", lambda **kwargs: camera)
    monkeypatch.setattr(inference, "Engine", lambda *args, **kwargs: Mock())
    args = replace(inference.CAMERA_CONFIG, checkpoint="test.pt", device="cpu", no_display=True)
    with pytest.raises(RuntimeError, match="camera"):
        inference.run(args)
    capture.release.assert_called_once()
    camera.close.assert_called_once()


def test_all_modes_use_neural_score():
    fusions = inference.load_fusions(None)
    sample = dict.fromkeys(inference.FEATURES, 0.0)
    for method, fusion in fusions.items():
        # Cùng mask/threshold, chỉ thay điểm mạng phải làm thay đổi nhãn.
        fusion["selected_model"]["candidate"] = [0.0] * 7 + [1.0, .5]
        sample["neuralnetwork_score"] = .1
        assert inference.classify(sample, fusion) == 0
        sample["neuralnetwork_score"] = .9
        assert inference.classify(sample, fusion) == 1
        assert fusion["source"] == "manual_unoptimized"


def test_wrong_report_method_is_rejected(monkeypatch):
    monkeypatch.setattr(inference, "load_fusion", lambda *args: {"fitness_method": "noisy_or"})
    with pytest.raises(ValueError, match="chế độ"):
        inference.load_fusions(None, {"weighted_mean": [1.] * 9}, {"weighted_mean": "test.json"})


@pytest.mark.parametrize('method, expected', [('weighted_mean', .6), ('noisy_or', .7552), ('cooccurrence', .64 + .36 * .64 * .32)])
def test_displayed_score_matches_fusion_formula(method, expected):
    sample = dict.fromkeys(inference.FEATURES, 0.)
    sample[inference.FEATURES[0]], sample[inference.FEATURES[-1]] = .4, .8
    fusion = dict(fitness_method=method, selected_model=dict(model='candidate', candidate=[.8] + [0.] * 6 + [.8, .65]))
    detail = inference.fusion_details(sample, fusion)
    assert detail['score'] == pytest.approx(expected)
    assert detail['prediction'] == int(expected >= .65)
    assert detail['threshold'] == .65


def test_empty_mask_does_not_predict_positive_at_zero_threshold():
    fitness = inference.Fitness([[1.] * 8])
    for method in inference.FUSION_CANDIDATES:
        assert fitness.score([0.] * 9, method)[0] == 0
        assert fitness.predict([0.] * 9, method)[0] == 0
