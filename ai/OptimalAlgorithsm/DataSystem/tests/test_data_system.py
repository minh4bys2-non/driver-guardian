import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ai.OptimalAlgorithsm.DataSystem import engine, loadvideo2handle
from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import NeuralNetwork, letterbox


class Capture:
    def __init__(self, times, fps):
        self.times, self.fps, self.index = times, fps, -1
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.index += 1
        return (True, np.full((4, 8, 3), self.index % 256, np.uint8)) if self.index < len(self.times) else (False, None)

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.times[self.index] * 1000

    def release(self):
        self.released = True


class DataSystemTests(unittest.TestCase):
    def resample(self, times, source_fps, target_fps):
        cap = Capture(times, source_fps)
        with patch.object(loadvideo2handle.cv2, 'VideoCapture', return_value=cap):
            result = list(loadvideo2handle.read_video('video', target_fps))
        self.assertTrue(cap.released)
        return result

    def test_equal_fps_preserves_every_frame(self):
        result = self.resample(np.arange(1800) / 30, 30, 30)
        self.assertEqual(len(result), 1800)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result[:4]], [0, 1, 2, 3])
        self.assertEqual(result[-1][1], 1799 / 30)

    def test_upsampling_includes_last_frame_interval(self):
        result = self.resample([0, .5], 2, 4)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 0, 1, 1])

    def test_downsampling_and_missing_timestamps(self):
        for times in ([0, .25, .5, .75], [0, 0, 0, 0]):
            result = self.resample(times, 4, 2)
            self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 2])

    def test_negative_initial_timestamp(self):
        result = self.resample([-.01, .49], 2, 2)
        self.assertEqual([t for _, t in result], [0, .5])

    def test_negative_origin_can_cross_zero(self):
        result = self.resample([-.25, 0, .25], 4, 4)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 1, 2])
        self.assertEqual([t for _, t in result], [0, .25, .5])

    def test_bad_timeline_rejected(self):
        with self.assertRaises(ValueError):
            self.resample([0, .5, .4], 2, 2)
        with self.assertRaises(ValueError):
            self.resample([0, .5, 0], 2, 2)
        with self.assertRaises(ValueError):
            self.resample([], 30, 30)

    def test_float_letterbox_matches_uint8(self):
        frame = np.full((4, 8, 3), 255, np.uint8)
        integer = letterbox(frame, 8)
        floating = letterbox(frame.astype(np.float32) / 255, 8)
        np.testing.assert_allclose(integer / 255, floating, atol=1e-7)
        np.testing.assert_array_equal(integer[2:6], frame)

    def test_engine_rejects_invalid_configuration_and_labels(self):
        with patch.object(engine, 'NeuralNetwork'):
            for options in ({'fps': 29.97}, {'neural_sec': 70}, {'neural_fps': 7}, {'fps': float('nan')}):
                with self.assertRaises(ValueError):
                    engine.Engine('.', **options)
            model = engine.Engine('.')
            for label in (5, 10, True, '1', None):
                with self.assertRaises(ValueError):
                    list(model.process_video('video.mp4', label))

    def test_jsonl_validation_strict_and_continue(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, 'NeuralNetwork'):
            src, dst = Path(tmp) / 'input.jsonl', Path(tmp) / 'output.jsonl'
            src.write_text('[]\n{"video_path": "valid.mp4", "label": 0}\n')
            model = engine.Engine(tmp)
            with patch.object(model, 'process_video', return_value=iter([{'label': 0}])):
                with patch('sys.stderr'):
                    self.assertEqual(model.run(src, dst), 1)
                self.assertEqual(dst.read_text().strip(), '{"label": 0}')
            with self.assertRaises(ValueError):
                model.run(src, dst, strict=True)
            with self.assertRaises(ValueError):
                model.run(src, src)

    def test_neural_preprocessing_and_binary_output(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                self.frames = x
                return (x[:, :1, :1, :1],) * 3

        class Neck(torch.nn.Module):
            def forward(self, *features):
                return features

        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.tensor([[0., 1.]])

            def forward(self, features, seq_lens):
                self.seq_lens = seq_lens
                return self.logits

        cnn = torch.nn.Module()
        cnn.backbone, cnn.neck = Backbone(), Neck()
        classifier = Classifier()
        model = NeuralNetwork(cnn_model=cnn, conv_gru_model=classifier, device='cpu')
        for dtype in (torch.uint8, torch.float16, torch.float32, torch.bfloat16):
            frames = torch.ones((2, 3, 4, 8), dtype=dtype)
            if dtype == torch.uint8:
                frames *= 255
            self.assertAlmostEqual(model(frames), torch.sigmoid(torch.tensor(1.)).item())
            self.assertEqual(tuple(cnn.backbone.frames.shape), (2, 3, 640, 640))
            self.assertAlmostEqual(cnn.backbone.frames[0, 0, 0, 0].item(), 114 / 255, places=6)
            self.assertEqual(classifier.seq_lens.tolist(), [2])
        for value in (float('nan'), -1., 2.):
            with self.assertRaises(ValueError):
                model(torch.full((2, 3, 4, 8), value))
        classifier.logits = torch.zeros((1, 3))
        with self.assertRaises(ValueError):
            model(torch.zeros((2, 3, 4, 8), dtype=torch.uint8))

    def test_window_schedule_neural_sampling_and_quality_gate(self):
        metrics = dict(eye_ready=True, mouth_ready=True, head_calibrated=True,
                       eye_observed_sec=2, mouth_observed_sec=2, head_observed_sec=2,
                       head_motion_resolution_hz=.5, head_motion_frequency_hz=None,
                       perclos_pct=10, blink_rate_per_min=0, yawning_frequency_per_min=0,
                       nodding_frequency_per_min=0, blink_detected=False, nod_detected=False)
        frames = [(np.full((3, 4, 8), i, np.uint8), i / 10) for i in range(30)]
        with patch.object(engine, 'NeuralNetwork') as neural, patch.object(engine, 'CameraMetrics') as camera, \
                patch.object(engine, 'read_video', return_value=iter(frames)):
            neural.return_value.eval.return_value.return_value = .7
            camera.return_value.process.return_value = metrics
            model = engine.Engine('.', fps=10, window_sec=2, stride_sec=1, neural_sec=1, neural_size=8)
            rows = list(model.process_video('video.mp4', 1))
            self.assertEqual([(r['start_second'], r['end_second']) for r in rows], [(0, 2), (1, 3)])
            batch = model.neural.call_args.args[0]
            self.assertEqual(batch.shape, (5, 3, 8, 8))
            self.assertEqual(batch[:, 0, 3, 3].tolist(), [20, 22, 24, 26, 28])
            self.assertEqual(rows[0]['perclos'], .1)
            self.assertEqual(rows[0]['dominant_head_motion_frequency'], 0)
            camera.return_value.close.assert_called_once()
            for missing in ({'eye_ready': False}, {'eye_observed_sec': 0}, {'head_motion_resolution_hz': None}):
                self.assertIsNone(model._features(metrics | missing, [], [], [], 'video', 0, 1))


if __name__ == '__main__':
    unittest.main()
