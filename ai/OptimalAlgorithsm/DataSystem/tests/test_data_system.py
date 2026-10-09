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

    def make_engine(self, **options):
        with patch.object(engine, 'NeuralNetwork'):
            model = engine.Engine('.', **options)
        model.neural.return_value = .7
        return model

    @staticmethod
    def metrics(window=60, **overrides):
        return dict(eye_ready=True, mouth_ready=True, head_calibrated=True,
                    eye_observed_sec=window, mouth_observed_sec=window, head_observed_sec=window,
                    head_motion_resolution_hz=1 / window, head_motion_frequency_hz=.1,
                    head_motion_observed_sec=window, perclos_pct=10, blink_rate_per_min=4,
                    yawning_frequency_per_min=0, nodding_frequency_per_min=0,
                    blink_detected=False, nod_detected=False) | overrides

    def test_risk_mapping_breakpoints_and_interpolation(self):
        model = self.make_engine()
        cases = {
            'blink_frequency': ((0, 1), (4, 1), (9.5, .5), (15, 0), (20, 0), (27.5, .5), (35, 1)),
            'blink_duration': ((0, 0), (400, 0), (600, .5), (800, 1), (2000, 1)),
            'perclos': ((0, 0), (.05, 0), (.1, .5), (.15, 1), (1, 1)),
            'yawn_frequency': ((0, 0), (1, .5), (2, 1), (3, 1)),
            'nod_duration': ((0, 0), (.5, 0), (1.25, .5), (2, 1), (3.5, 1)),
            'nod_frequency': ((0, 0), (1.5, .5), (3, 1), (4, 1)),
            'dominant_head_motion_frequency': ((0, 1), (.025, .5), (.05, 0), (.1, 0),
                                               (.2, 0), (.4, .5), (.6, 1), (1, 1)),
            'cnn_lstm_score': ((0, 0), (.5, .5), (1, 1)),
        }
        for key, points in cases.items():
            for value, expected in points:
                with self.subTest(key=key, value=value):
                    self.assertAlmostEqual(model._norm(key, value), expected)

    def test_risk_mapping_rejects_missing_and_nonfinite_values(self):
        model = self.make_engine()
        for value in (None, float('nan'), float('inf'), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model._norm('blink_frequency', value)

    def test_risk_config_validation_and_copy(self):
        config = {'perclos': {'safe': 0, 'danger': .2}}
        model = self.make_engine(risk_config=config)
        self.assertAlmostEqual(model._norm('perclos', .1), .5)
        config['perclos']['danger'] = .5
        self.assertAlmostEqual(model._norm('perclos', .1), .5)
        invalid = (
            {'perclos': {'safe': .2, 'danger': .1}}, {'perclos': {'danger': 15}},
            {'blink_duration': {'unit': 'sec'}}, {'blink_duration': {'aggregation': 'mean'}},
            {'window_sec': 0}, {'blink_frequency': {'normal': (15, 15)}},
            {'blink_frequency': {'high_danger': float('nan')}},
            {'yawn_frequency': {'min_event_sec': 7.5}}, {'nod_frequency': {'min_event_sec': -1}},
            {'dominant_head_motion_frequency': {'method': 'other'}},
            {'dominant_head_motion_frequency': {'require_reliable_motion': 'yes'}},
            {'unknown': {}}, {'perclos': {'unknown': 1}},
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make_engine(risk_config=config)
        with self.assertRaises(ValueError):
            self.make_engine(window_sec=30)

    def test_aggregation_unit_conversion_and_eye_gate(self):
        model = self.make_engine()
        frames = [np.zeros((3, 8, 8), np.uint8)]
        blink = [(1, 100), (2, 200), (3, 800)]  # P90 = 680 ms, not the mean.
        nod = [(1, 900), (2, 1500)]  # Maximum = 1.5 seconds.
        metrics = self.metrics(yawning_frequency_per_min=1, nodding_frequency_per_min=1.5)
        row = model._features(metrics, blink, nod, frames, 'video', 0, 1)
        for key, expected in dict(blink_duration=.7, nod_duration=.6667, perclos=.5,
                                  blink_frequency=.7, yawn_frequency=.5, nod_frequency=.5,
                                  cnn_lstm_score=.7).items():
            self.assertEqual(row[key], expected)
        self.assertEqual(row['feature_encoding'], 'risk_v2')
        for frequency, perclos, expected in ((4, 5, 0), (4, 10, .5), (4, 15, 1),
                                            (15, 15, 0), (35, 5, 1)):
            row = model._features(self.metrics(blink_rate_per_min=frequency, perclos_pct=perclos),
                                  [], [], frames, 'video', 0, 1)
            self.assertEqual(row['blink_frequency'], expected)
        model = self.make_engine(risk_config={'blink_frequency': {'gate_low_with_eye_metrics': False}})
        row = model._features(self.metrics(perclos_pct=5), [], [], frames, 'video', 0, 1)
        self.assertEqual(row['blink_frequency'], 1)

    def test_motion_reliability_and_configured_window(self):
        from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow
        model = self.make_engine()
        frames = [np.zeros((3, 8, 8), np.uint8)]
        for change in ({'head_motion_frequency_hz': None}, {'head_motion_frequency_hz': 0},
                       {'head_motion_frequency_hz': float('nan')},
                       {'head_motion_resolution_hz': None}, {'head_motion_observed_sec': 59.9}):
            self.assertIsNone(model._features(self.metrics(**change), [], [], frames, 'video', 0, 1))
        for frequency in (.1, .4):
            motion = HeadMotionWindow(60)
            for i in range(1201):
                result = motion.process(5 * np.sin(2 * np.pi * frequency * i / 20), i / 20)
            row = model._features(self.metrics(**result), [], [], frames, 'video', 0, 1)
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row['dominant_head_motion_frequency'], 0 if frequency == .1 else .5, delta=.02)
            lost = motion.process(None, 60.05)
            self.assertIsNone(model._features(self.metrics(**lost), [], [], frames, 'video', 0, 1))
        model = self.make_engine(risk_config={'dominant_head_motion_frequency': {'require_reliable_motion': False}})
        row = model._features(self.metrics(head_motion_frequency_hz=None), [], [], frames, 'video', 0, 1)
        self.assertEqual(row['dominant_head_motion_frequency'], 0)

    def test_minimum_event_durations_reach_physical_detector(self):
        model = self.make_engine()
        with patch.object(engine, 'CameraMetrics') as factory, patch.object(engine, 'read_video', return_value=[]):
            list(model.process_video('video.mp4', 0))
            options = factory.call_args.kwargs
        camera = engine.CameraMetrics(**options)
        self.assertEqual(camera.head_motion.window, 60)
        self.assertEqual(camera.mouth.fsm.minimum, 4000)
        self.assertEqual(camera.nod.minimum, 800)
        for detector, duration in ((camera.mouth.fsm, 3.9), (camera.mouth.fsm, 4.0),
                                   (camera.nod, .7), (camera.nod, .8)):
            detector.reset()
            detector.process(0, 0)
            detector.process(1, .1)
            for i in range(2, round(duration * 10) + 1):
                detector.process(1, i / 10)
            event = detector.process(0, (round(duration * 10) + 1) / 10)
            self.assertEqual(event['event_done'], duration in (4., .8))
        camera.close()

    def test_window_schedule_neural_sampling_and_quality_gate(self):
        metrics = self.metrics(2, head_motion_resolution_hz=.5, head_motion_frequency_hz=.5,
                               blink_rate_per_min=0)
        frames = [(np.full((3, 4, 8), i, np.uint8), i / 10) for i in range(30)]
        with patch.object(engine, 'NeuralNetwork') as neural, patch.object(engine, 'CameraMetrics') as camera, \
                patch.object(engine, 'read_video', return_value=iter(frames)):
            neural.return_value.eval.return_value.return_value = .7
            camera.return_value.process.return_value = metrics
            model = engine.Engine('.', fps=10, stride_sec=1, neural_sec=1, neural_size=8,
                                  risk_config={'window_sec': 2, 'dominant_head_motion_frequency': {'window_sec': 2}})
            rows = list(model.process_video('video.mp4', 1))
            self.assertEqual([(r['start_second'], r['end_second']) for r in rows], [(0, 2), (1, 3)])
            batch = model.neural.call_args.args[0]
            self.assertEqual(batch.shape, (5, 3, 8, 8))
            self.assertEqual(batch[:, 0, 3, 3].tolist(), [20, 22, 24, 26, 28])
            self.assertEqual(rows[0]['perclos'], .5)
            self.assertEqual(rows[0]['blink_frequency'], .5)
            self.assertEqual(rows[0]['feature_encoding'], 'risk_v2')
            self.assertEqual(rows[0]['dominant_head_motion_frequency'], .75)
            camera.return_value.close.assert_called_once()
            for perclos, expected in ((0, 0), (5, 0), (10, .5), (15, 1)):
                sample = model._features(
                    metrics | {'blink_rate_per_min': 4, 'perclos_pct': perclos},
                    [], [], batch, 'video', 0, 1)
                self.assertEqual(sample['blink_frequency'], expected)
            for missing in ({'eye_ready': False}, {'eye_observed_sec': 0}, {'head_motion_resolution_hz': None}):
                self.assertIsNone(model._features(metrics | missing, [], [], [], 'video', 0, 1))


if __name__ == '__main__':
    unittest.main()
