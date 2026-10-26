import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ai.OptimalAlgorithsm.DataSystem import engine
from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import read_video
from ai.OptimalAlgorithsm.DataSystem.physicalbranch import VideoMetrics
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow


class DataRegressionTests(unittest.TestCase):
    def make_engine(self, **kwargs):
        with patch.object(engine, 'NeuralNetwork'):
            model = engine.Engine('.', neural_size=8, **kwargs)
        model.neural.return_value = .5
        return model

    @staticmethod
    def metrics():
        return dict(eye_ready=True, mouth_ready=True, head_calibrated=True,
                    eye_observed_sec=30, mouth_observed_sec=30, head_observed_sec=30,
                    head_motion_resolution_hz=1 / 30, head_motion_frequency_hz=0.,
                    head_motion_observed_sec=30, perclos_pct=0., blink_rate_per_min=0.,
                    yawning_frequency_per_min=0., nodding_frequency_per_min=0.,
                    blink_detected=False, nod_detected=False)

    def test_static_valid_window_is_retained(self):
        model = self.make_engine()
        row = model._features(self.metrics(), [], [], [np.zeros((3, 8, 8), np.uint8)], 'video', 0, 0)
        self.assertEqual(row['dominant_head_motion_frequency'], 0.)
        self.assertEqual(row['neuralnetwork_score'], .5)

    def test_quality_gate_uses_effective_metric_windows(self):
        model = self.make_engine()
        metrics = self.metrics() | dict(windows_sec=dict(perclos=10, pom=20, over_angle=30, head_motion=15),
                                       eye_observed_sec=8, mouth_observed_sec=16,
                                       head_observed_sec=24, head_motion_observed_sec=15)
        frames = [np.zeros((3, 8, 8), np.uint8)]
        self.assertIsNotNone(model._features(metrics, [], [], frames, 'video', 0, 0))
        for key in ('eye_observed_sec', 'mouth_observed_sec', 'head_observed_sec', 'head_motion_observed_sec'):
            with self.subTest(key=key), contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(model._features(metrics | {key: metrics[key] - .01}, [], [], frames, 'video', 0, 0))

    def test_full_fft_policy_explains_first_window_and_loss_skips(self):
        model = self.make_engine(fps=10)
        motion = HeadMotionWindow(30)
        def process(frame, timestamp):
            # Head calibration takes one second; a single lost pose at 65s resets FFT.
            pitch = None if timestamp < 1 or timestamp == 65 else 0.
            return self.metrics() | motion.process(pitch, timestamp)
        frames = ((np.zeros((3, 4, 8), np.uint8), i / 10) for i in range(1200))
        with patch.object(engine, 'CameraMetrics') as camera, patch.object(engine, 'read_video', return_value=frames), contextlib.redirect_stderr(io.StringIO()):
            camera.return_value.process.side_effect = process
            sampled = list(model.process_video('video.mp4', 0))
        self.assertEqual([row['start_second'] for row in sampled], [30., 90.])
        camera.return_value.close.assert_called_once()

    def test_manifest_subject_survives_extraction_and_resume(self):
        model = self.make_engine()
        row = model._features(self.metrics(), [], [], [np.zeros((3, 8, 8), np.uint8)], 'video.mp4', 0, 0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            src, dst = Path(tmp) / 'input.jsonl', Path(tmp) / 'output.jsonl'
            src.write_text(json.dumps(dict(video_path='video.mp4', label=0, subject_id='driver-1')))
            with patch.object(model, 'process_video', return_value=iter([row])):
                self.assertEqual(model.run(src, dst, strict=True), 1)
            self.assertEqual(json.loads(dst.read_text())['subject_id'], 'driver-1')
            second = row | dict(start_second=30, end_second=60)
            src.write_text(json.dumps(dict(video_path='video.mp4', label=0)))
            with patch.object(model, 'process_video', return_value=iter([second])):
                self.assertEqual(model.run(src, dst, strict=True, resume=True), 1)
            self.assertEqual(json.loads(dst.read_text().splitlines()[1])['subject_id'], 'driver-1')
            src.write_text(json.dumps(dict(video_path='video.mp4', label=0, subject_id='driver-2')))
            with self.assertRaisesRegex(ValueError, 'Conflicting subject'):
                model.run(src, dst, strict=True, resume=True)

    def test_invalid_frames_and_processing_errors_close_camera(self):
        model = self.make_engine()
        for frame in (np.zeros((3, 4, 8)), np.zeros((4, 8, 3), np.uint8), np.zeros((3, 0, 8), np.uint8)):
            with patch.object(engine, 'CameraMetrics') as camera, patch.object(engine, 'read_video', return_value=[(frame, 0)]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(ValueError):
                    list(model.process_video('video.mp4', 0))
                camera.return_value.close.assert_called_once()
        with patch.object(engine, 'CameraMetrics') as camera, patch.object(engine, 'read_video', return_value=[(np.zeros((3, 4, 8), np.uint8), 0)]), contextlib.redirect_stderr(io.StringIO()):
            camera.return_value.process.side_effect = RuntimeError('tracking failed')
            with self.assertRaisesRegex(RuntimeError, 'tracking failed'):
                list(model.process_video('video.mp4', 0))
            camera.return_value.close.assert_called_once()

    def test_real_video_resampling_preserves_timeline_and_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(10):
                    writer.write(np.full((24, 32, 3), (10, 50, 200), np.uint8))
            finally:
                writer.release()
            for fps in (5, 10, 20):
                frames = list(read_video(path, fps))
                self.assertEqual(len(frames), fps)
                np.testing.assert_allclose([t for _, t in frames], np.arange(fps) / fps)
                frame = frames[0][0]
                self.assertEqual(frame.shape, (3, 24, 32))
                np.testing.assert_allclose(frame.mean(axis=(1, 2)), [200, 50, 10], atol=3)

    def test_video_metrics_resets_between_videos_and_converts_rgb(self):
        with patch('ai.OptimalAlgorithsm.DataSystem.physicalbranch.CameraMetrics') as camera:
            camera.return_value.process.return_value = {}
            metrics = VideoMetrics()
            frame = np.zeros((3, 4, 8), np.uint8)
            frame[0] = 100
            for _ in range(2):
                metrics.process([(frame, 0.)])
            self.assertEqual(camera.return_value.reset.call_count, 2)
            self.assertEqual(camera.return_value.close.call_count, 2)
            bgr, timestamp = camera.return_value.process.call_args.args
            self.assertEqual(bgr[0, 0].tolist(), [0, 0, 100])
            self.assertEqual(timestamp, 0.)
