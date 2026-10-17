import json
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from ai.LSTM.src.dataloader import RawVideoFramesDataset
from ai.LSTM.src.model import DrowsyClassifier
from ai.OptimalAlgorithsm.DataSystem import engine
from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import read_video
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, load_jsonl


@pytest.fixture
def checkpoint(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                             input_dim=4, hidden_dim=4, dropout=0).eval()
    path = tmp_path / 'best.pt'
    torch.save(dict(format_version=1, model=model.state_dict(), model_config=model.model_config,
                    config=dict(image_size=64, target_fps=5, window_sec=2), epoch=3), path)
    yield path, model
    torch.set_num_threads(previous)


def write_video(path, fps, seconds=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (32, 24))
    assert writer.isOpened()
    try:
        for i in range(round(fps * seconds)):
            writer.write(np.full((24, 32, 3), (i % 256, 30, 200), np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize('source_fps', [25, 30, 60])
def test_real_checkpoint_video_and_jsonl_match_training(tmp_path, monkeypatch, checkpoint, source_fps):
    path, trained = checkpoint
    video = tmp_path / 'test/1_drowsy/clip.mp4'
    write_video(video, source_fps)
    config = replace(DEFAULT_CONFIG, checkpoint_path=str(path), device='cpu', neural_size=64,
                     window_sec=2, stride_sec=2, neural_sec=2, neural_fps=5, chunk_size=2)
    pipeline = engine.Engine(tmp_path, config=config)
    for name, value in trained.state_dict().items():
        torch.testing.assert_close(pipeline.neural.model.state_dict()[name], value, rtol=0, atol=0)
    assert not pipeline.neural.model.training
    captured = []
    handle = pipeline.neural.model.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    camera = Mock()
    camera.process.return_value = dict(
        eye_ready=True, mouth_ready=True, head_calibrated=True,
        eye_observed_sec=2, mouth_observed_sec=2, head_observed_sec=2,
        head_motion_resolution_hz=.5, head_motion_frequency_hz=.5, head_motion_observed_sec=2,
        perclos_pct=25., blink_rate_per_min=4., yawning_frequency_per_min=1.,
        nodding_frequency_per_min=2., blink_detected=False, nod_detected=False,
    )
    monkeypatch.setattr(engine, 'CameraMetrics', lambda **kwargs: camera)
    manifest, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    manifest.write_text(json.dumps(dict(video_path='test/1_drowsy/clip.mp4', label=1, subject_id='A')))
    assert pipeline.run(manifest, output, strict=True) == 1
    handle.remove()
    camera.close.assert_called_once()
    assert camera.process.call_count == 60
    training_frames = RawVideoFramesDataset(tmp_path, 'test', window_sec=2, target_fps=5, img_size=64)[0][0]
    torch.testing.assert_close(captured[0][0], training_frames, rtol=0, atol=0)
    with torch.inference_mode():
        expected_score = trained(training_frames[None]).softmax(-1)[0, 1].item()
    row = json.loads(output.read_text())
    assert row['neuralnetwork_score'] == pytest.approx(expected_score, abs=1e-6)
    assert row['label'] == 1 and row['subject_id'] == 'A'
    assert (row['start_second'], row['end_second']) == (0, 2)
    assert row['perclos'] == .25
    assert row['raw_features']['blink_duration'] == row['raw_features']['nod_duration'] == 0
    for key in FEATURES:
        assert row[key] == pytest.approx(row['raw_features'][key] / row['normalization_scales'][key])
    X, y, metadata = load_jsonl(output, include_metadata=True)
    np.testing.assert_allclose(X[0], [row[key] for key in FEATURES])
    assert y.tolist() == [1]
    assert metadata['input_schema']['normalization_id'] == pipeline.normalizer.id
    assert pipeline.run(manifest, output, strict=True, resume=True) == 0
    assert len(output.read_text().splitlines()) == 1


def test_fractional_sampling_uses_previous_source_frame(tmp_path):
    # DataSystem is causal; training rounds indices to the nearest source frame.
    video = tmp_path / 'test/0_alert/clip.mp4'
    write_video(video, 24)
    sampled = [frame for i, (frame, _) in enumerate(read_video(str(video), 30)) if i % 6 == 0]
    cap = cv2.VideoCapture(str(video))
    decoded = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    finally:
        cap.release()
    indices = np.floor(np.arange(10) * 24 / 5).astype(int)
    np.testing.assert_array_equal(sampled, np.asarray(decoded)[indices])
    assert np.any(indices != np.rint(np.arange(10) * 24 / 5).astype(int))
