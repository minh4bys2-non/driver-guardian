import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ai.OptimalAlgorithsm.DataSystem import engine, loadvideo2handle
from ai.OptimalAlgorithsm.DataSystem.config import DataSystemConfig, DEFAULT_CONFIG, MODEL_IMAGE_SIZE
from ai.OptimalAlgorithsm.DataSystem.normalization import PhysicalNormalizer, FEATURE_ENCODING
from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import NeuralNetwork, letterbox


class Capture:
    def __init__(self, times, fps):
        self.times, self.fps, self.index = times, fps, -1
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.index += 1
        return (
            (True, np.full((4, 8, 3), self.index % 256, np.uint8))
            if self.index < len(self.times)
            else (False, None)
        )

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.times[self.index] * 1000

    def release(self):
        self.released = True


class DataSystemTests(unittest.TestCase):
    def test_default_image_size_is_224(self):
        self.assertEqual(MODEL_IMAGE_SIZE, 224)
        self.assertEqual(DataSystemConfig().neural_size, 224)
        self.assertEqual(DEFAULT_CONFIG.neural_size, 224)
        self.assertEqual(self.make_engine().neural_size, 224)

    def test_default_letterbox_preserves_aspect_ratio_channels_and_dtype(self):
        for height, width in ((100, 200), (200, 100), (224, 224)):
            for dtype in (np.uint8, np.float32):
                for chw in (False, True):
                    with self.subTest(shape=(height, width), dtype=dtype, chw=chw):
                        color = np.array([51, 102, 204], dtype=dtype)
                        if dtype == np.float32:
                            color /= 255
                        frame = np.broadcast_to(color, (height, width, 3)).copy()
                        result = letterbox(frame.transpose(2, 0, 1) if chw else frame)
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(result.shape, (3, 224, 224) if chw else (224, 224, 3))
                        result = result.transpose(1, 2, 0) if chw else result
                        expected = np.full((224, 224, 3), 114 if dtype == np.uint8 else 114 / 255, dtype=dtype)
                        if width > height:
                            expected[56:168] = color
                        elif height > width:
                            expected[:, 56:168] = color
                        else:
                            expected[:] = color
                        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_engine_passes_default_224_rgb_frames_to_neural(self):
        model = self.make_engine(fps=5, window_sec=2, stride_sec=2, neural_sec=1)
        frames = ((np.full((3, 4, 8), i, np.uint8), i / 5) for i in range(10))
        with patch.object(engine, "CameraMetrics") as camera, patch.object(
            engine, "read_video", return_value=frames
        ):
            camera.return_value.process.return_value = self.metrics(
                window=2, head_motion_resolution_hz=.5
            )
            rows = list(model.process_video("video.mp4", 0))
        self.assertEqual(len(rows), 1)
        model.neural.assert_called_once()
        batch = model.neural.call_args.args[0]
        self.assertEqual(batch.shape, (5, 3, 224, 224))
        self.assertEqual(batch.dtype, np.uint8)
        np.testing.assert_array_equal(batch[:, 0, 112, 112], [5, 6, 7, 8, 9])
        self.assertTrue(np.all(batch[:, :, :56] == 114))
        self.assertTrue(np.all(batch[:, :, 168:] == 114))
        self.assertEqual(rows[0]["neuralnetwork_score"], .7)
        camera.return_value.close.assert_called_once()

    def test_30fps_timeline_samples_last_ten_seconds_at_5fps(self):
        model = self.make_engine()
        frames = ((np.full((3, 4, 8), (i // 6) % 256, np.uint8), i / 30)
                  for i in range(1800))
        with patch.object(engine, "CameraMetrics") as camera, patch.object(
            engine, "read_video", return_value=frames
        ):
            camera.return_value.process.return_value = self.metrics()
            rows = list(model.process_video("video.mp4", 0))
        self.assertEqual(len(rows), 1)
        model.neural.assert_called_once()
        batch = model.neural.call_args.args[0]
        self.assertEqual(batch.shape, (50, 3, 224, 224))
        np.testing.assert_array_equal(batch[:, 0, 112, 112], np.arange(250, 300) % 256)

    def test_inherited_config_reaches_camera_and_neural_pipeline(self):
        config = DataSystemConfig(
            fps=20, window_sec=30, stride_sec=30, neural_sec=4, neural_fps=5,
            chunk_size=8, init_duration_sec=2, calibration_frames=12,
            nod_pitch_deg=16, nod_release_deg=9, nod_direction=1,
            yawn_duration_ms=(2000, 6000), nod_duration_ms=(500, 2500),
            blink_duration_ms=(120, 1500), min_variance=.002,
            head_motion_min_amplitude_deg=.3, windows={"head_motion": 20},
        )
        with patch.object(engine, "NeuralNetwork") as neural:
            model = engine.Engine(".", config=config)
        self.assertEqual(model.window, 600)
        self.assertEqual(model.stride, 600)
        self.assertEqual(model.neural_count, 20)
        self.assertEqual(neural.call_args.kwargs['chunk_size'], 8)
        self.assertEqual(neural.call_args.kwargs['config'], model.config)
        with patch.object(engine, "CameraMetrics") as factory, patch.object(
            engine, "read_video", return_value=[]
        ):
            list(model.process_video("video.mp4", 0))
        camera = engine.CameraMetrics(**factory.call_args.kwargs)
        self.assertEqual(camera.eye.init_frames, 40)
        self.assertEqual(camera.eye.fsm.minimum, 120)
        self.assertEqual(camera.eye.hmm.min_variance, .002)
        self.assertEqual(camera.mouth.fsm.minimum, 2000)
        self.assertEqual(camera.mouth.fsm.maximum, 6000)
        self.assertEqual(camera.nod.minimum, 500)
        self.assertEqual(camera.nod.maximum, 2500)
        self.assertEqual(camera.nod_pitch, 16)
        self.assertEqual(camera.nod_release, 9)
        self.assertEqual(camera.nod_direction, 1)
        self.assertEqual(camera.head_motion.window, 20)
        self.assertEqual(camera.head_motion.min_amplitude, .3)
        self.assertEqual(camera.calibration_frames, 12)
        self.assertEqual(config.windows, {"head_motion": 20})
        self.assertEqual(DEFAULT_CONFIG.windows, {})
        self.assertEqual(model.normalizer.scales['blink_duration'], 1500)
        self.assertEqual(model.normalizer.scales['nod_duration'], 2.5)

    def resample(self, times, source_fps, target_fps):
        cap = Capture(times, source_fps)
        with patch.object(loadvideo2handle.cv2, "VideoCapture", return_value=cap):
            result = list(loadvideo2handle.read_video("video", target_fps, target_luma=0))
        self.assertTrue(cap.released)
        return result

    def test_equal_fps_preserves_every_frame(self):
        result = self.resample(np.arange(1800) / 30, 30, 30)
        self.assertEqual(len(result), 1800)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result[:4]], [0, 1, 2, 3])
        self.assertEqual(result[-1][1], 1799 / 30)

    def test_upsampling_includes_last_frame_interval(self):
        result = self.resample([0, 0.5], 2, 4)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 0, 1, 1])

    def test_downsampling_and_missing_timestamps(self):
        for times in ([0, 0.25, 0.5, 0.75], [0, 0, 0, 0]):
            result = self.resample(times, 4, 2)
            self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 2])

    def test_negative_initial_timestamp(self):
        result = self.resample([-0.01, 0.49], 2, 2)
        self.assertEqual([t for _, t in result], [0, 0.5])

    def test_negative_origin_can_cross_zero(self):
        result = self.resample([-0.25, 0, 0.25], 4, 4)
        self.assertEqual([int(f[0, 0, 0]) for f, _ in result], [0, 1, 2])
        self.assertEqual([t for _, t in result], [0, 0.25, 0.5])

    def test_bad_timeline_rejected(self):
        with self.assertRaises(ValueError):
            self.resample([0, 0.5, 0.4], 2, 2)
        with self.assertRaises(ValueError):
            self.resample([0, 0.5, 0], 2, 2)
        with self.assertRaises(ValueError):
            self.resample([], 30, 30)

    def test_float_letterbox_matches_uint8(self):
        frame = np.full((4, 8, 3), 255, np.uint8)
        integer = letterbox(frame, 8)
        floating = letterbox(frame.astype(np.float32) / 255, 8)
        np.testing.assert_allclose(integer / 255, floating, atol=1e-7)
        np.testing.assert_array_equal(integer[2:6], frame)

    def test_engine_rejects_invalid_configuration_and_labels(self):
        with patch.object(engine, "NeuralNetwork"):
            for options in (
                {"fps": 29.97},
                {"neural_sec": 70},
                {"neural_fps": 7},
                {"fps": float("nan")},
            ):
                with self.assertRaises(ValueError):
                    engine.Engine(".", **options)
            model = engine.Engine(".")
            for label in (5, 10, True, "1", None):
                with self.assertRaises(ValueError):
                    list(model.process_video("video.mp4", label))

    def test_jsonl_validation_strict_and_continue(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            engine, "NeuralNetwork"
        ):
            src, dst = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            src.write_text('[]\n{"video_path": "valid.mp4", "label": 0}\n')
            model = engine.Engine(tmp)
            with patch.object(
                model, "process_video", return_value=iter([{"label": 0}])
            ):
                with patch("sys.stderr"):
                    self.assertEqual(model.run(src, dst), 1)
                self.assertEqual(dst.read_text().strip(), '{"label": 0}')
            with self.assertRaises(ValueError):
                model.run(src, dst, strict=True)
            with self.assertRaises(ValueError):
                model.run(src, src)

    def test_resume_preserves_samples_and_rejects_broken_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            src.write_text('{"video_path":"video.mp4","label":0}\n')
            model = self.make_engine()
            row = dict.fromkeys(model.normalizer.scales, 0.0)
            row.update(
                source_video="video.mp4",
                label=0,
                start_second=0,
                end_second=60,
                feature_encoding=FEATURE_ENCODING,
                normalization_id=model.normalizer.id,
                normalization_scales=model.normalizer.scales,
            )
            second = row | {"start_second": 60, "end_second": 120}
            original = json.dumps(row)
            dst.write_text(original)
            with patch.object(
                model, "process_video", side_effect=lambda *a, **kw: iter([row, second])
            ):
                self.assertEqual(model.run(src, dst, resume=True), 1)
                self.assertEqual(model.run(src, dst, resume=True), 0)
            self.assertEqual(
                [json.loads(s) for s in dst.read_text().splitlines()], [row, second]
            )
            for bad in (
                '{"broken":',
                json.dumps(row | {"feature_encoding": "risk_v2"}),
                json.dumps(row | {"normalization_id": "old"}),
                json.dumps(row | {"normalization_scales": {}}),
                json.dumps(row | {"perclos": True}),
                json.dumps(row | {"perclos": [0.0]}),
                json.dumps(row | {"start_second": False}),
                json.dumps(row | {"normalization_scales": model.normalizer.scales | {"perclos": True}}),
                json.dumps(row),
                json.dumps(row | {"label": 1}),
                json.dumps(row | {"start_second": 10, "end_second": 70}),
            ):
                dst.write_text(original + "\n" + bad)
                with self.assertRaises(ValueError):
                    model.run(src, dst, resume=True)
                self.assertEqual(dst.read_text(), original + "\n" + bad)

    def test_bad_manifest_types_continue_and_duplicate_videos_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input.jsonl", Path(tmp) / "output.jsonl"
            rows = [
                {"video_path": [], "label": 0},
                {"video_path": "video.mp4", "label": []},
                {"video_path": "video.mp4", "label": True},
                {"video_path": "video.mp4", "label": 0},
                {"video_path": "./video.mp4", "label": 0},
                {"video_path": "video.mp4", "label": 1},
            ]
            src.write_text("\n".join(json.dumps(r) for r in rows))
            for resume in (False, True):
                model = self.make_engine()
                if dst.exists():
                    dst.write_text("")
                with patch.object(
                    model, "process_video", return_value=iter([])
                ) as process, patch("sys.stderr"):
                    self.assertEqual(model.run(src, dst, resume=resume), 0)
                    process.assert_called_once()
                with self.assertRaises(ValueError):
                    model.run(src, dst, resume=resume, strict=True)

    def test_nonfinite_coverage_cannot_pass_quality_gate(self):
        model = self.make_engine()
        frames = [np.zeros((3, 8, 8), np.uint8)]
        for key in (
            "eye_observed_sec",
            "mouth_observed_sec",
            "head_observed_sec",
            "head_motion_observed_sec",
        ):
            for value in (None, float("nan"), float("inf")):
                with self.subTest(key=key, value=value):
                    self.assertIsNone(
                        model._features(
                            self.metrics(**{key: value}), [], [], frames, "video", 0, 1
                        )
                    )
        model.neural.assert_not_called()

    def test_default_stride_produces_nonoverlapping_windows(self):
        model = self.make_engine(neural_size=8)
        self.assertEqual(model.stride / model.fps, 60)
        frames = ((np.zeros((3, 4, 8), np.uint8), i / 30) for i in range(3600))
        with patch.object(engine, "CameraMetrics") as camera, patch.object(
            engine, "read_video", return_value=frames
        ):
            camera.return_value.process.return_value = self.metrics()
            rows = list(model.process_video("video.mp4", 1))
        self.assertEqual(
            [(r["start_second"], r["end_second"]) for r in rows], [(0, 60), (60, 120)]
        )

    def make_engine(self, **options):
        with patch.object(engine, "NeuralNetwork"):
            model = engine.Engine(".", **options)
        model.neural.return_value = 0.7
        return model

    @staticmethod
    def metrics(window=60, **overrides):
        return (
            dict(
                eye_ready=True,
                mouth_ready=True,
                head_calibrated=True,
                eye_observed_sec=window,
                mouth_observed_sec=window,
                head_observed_sec=window,
                head_motion_resolution_hz=1 / window,
                head_motion_frequency_hz=0.1,
                head_motion_observed_sec=window,
                perclos_pct=10,
                blink_rate_per_min=4,
                yawning_frequency_per_min=0,
                nodding_frequency_per_min=0,
                blink_detected=False,
                nod_detected=False,
            )
            | overrides
        )

    def test_normalization_preserves_magnitude_and_endpoints(self):
        normalizer = PhysicalNormalizer(DEFAULT_CONFIG)
        for fraction in (0, .1, .5, 1):
            raw = {key: upper * fraction for key, upper in normalizer.scales.items()}
            for value in normalizer.transform(raw).values():
                self.assertAlmostEqual(value, fraction)
        raw = dict.fromkeys(normalizer.scales, 0.)
        for value in (None, True, float('nan'), float('inf'), -1,
                      normalizer.scales['blink_frequency'] + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalizer.transform(raw | {'blink_frequency': value})

    def test_normalization_uses_effective_detector_options_and_windows(self):
        cfg = replace(DEFAULT_CONFIG, windows={'blink': 10, 'yawn': 15},
                      eye_options={'min_duration_ms': 200, 'max_duration_ms': 900},
                      mouth_options={'min_duration_ms': 500, 'max_duration_ms': 2000})
        normalizer = PhysicalNormalizer(cfg)
        self.assertEqual(normalizer.scales['blink_duration'], 900)
        self.assertEqual(normalizer.scales['dominant_head_motion_frequency'], 15)
        self.assertNotEqual(normalizer.id, PhysicalNormalizer(DEFAULT_CONFIG).id)
        self.assertEqual(normalizer.id, PhysicalNormalizer(cfg).id)
        for mode, window, minimum, key in (
            ('eye', 10, 200, 'blink_frequency'), ('mouth', 15, 500, 'yawn_frequency')
        ):
            from ai.PhysicalBranch.temporal_metrics import StateMachine
            fsm = StateMachine(mode, window_size_sec=window, min_duration_ms=minimum, config=cfg)
            fsm.process(0, 0)
            duration_frames = round(minimum * cfg.fps / 1000)
            for i in range(1, round((window + 2) * cfg.fps)):
                state = int(i % (duration_frames + 1) != 0)
                result = fsm.process(state, i / cfg.fps)
                self.assertLessEqual(result['rate_per_minute'], normalizer.scales[key])

    def test_aggregation_units_and_no_cross_feature_gate(self):
        model = self.make_engine()
        frames = [np.zeros((3, 8, 8), np.uint8)]
        row = model._features(self.metrics(yawning_frequency_per_min=1, nodding_frequency_per_min=1.5),
                              [(1, 100), (2, 200), (3, 800)], [(1, 900), (2, 1500)], frames, 'video', 0, 1)
        self.assertAlmostEqual(row['blink_duration'], 680 / 2000)
        self.assertAlmostEqual(row['nod_duration'], 1.5 / 3.5)
        self.assertEqual(row['perclos'], .1)
        self.assertEqual(row['neuralnetwork_score'], .7)
        self.assertEqual(row['raw_features']['blink_duration'], 680)
        self.assertEqual(row['feature_encoding'], FEATURE_ENCODING)
        for perclos in (0, 5, 10, 15, 100):
            other = model._features(self.metrics(perclos_pct=perclos), [], [], frames, 'video', 0, 1)
            self.assertEqual(row['blink_frequency'], other['blink_frequency'])
            self.assertEqual(other['perclos'], perclos / 100)

    def test_fft_bound_covers_fractional_frame_windows(self):
        from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow

        cfg = replace(DEFAULT_CONFIG, windows={'head_motion': .165})
        normalizer = PhysicalNormalizer(cfg)
        motion = HeadMotionWindow(.165, config=cfg)
        for i in range(30):
            result = motion.process(5 * (-1) ** i, i / cfg.fps)
            frequency = result['head_motion_frequency_hz']
            if frequency is not None:
                self.assertLessEqual(frequency, normalizer.scales['dominant_head_motion_frequency'])
        self.assertIsNotNone(result['head_motion_frequency_hz'])

    def test_motion_reliability_and_configured_window(self):
        from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow

        model = self.make_engine()
        frames = [np.zeros((3, 8, 8), np.uint8)]
        for change in (
            {"head_motion_frequency_hz": None},
            {"head_motion_frequency_hz": -0.1},
            {"head_motion_frequency_hz": float("nan")},
            {"head_motion_resolution_hz": None},
            {"head_motion_observed_sec": 59.9},
        ):
            self.assertIsNone(
                model._features(self.metrics(**change), [], [], frames, "video", 0, 1)
            )
        for frequency in (0.1, 0.4):
            motion = HeadMotionWindow(60)
            for i in range(1201):
                result = motion.process(
                    5 * np.sin(2 * np.pi * frequency * i / 20), i / 20
                )
            row = model._features(self.metrics(**result), [], [], frames, "video", 0, 1)
            self.assertIsNotNone(row)
            self.assertAlmostEqual(
                row["dominant_head_motion_frequency"],
                frequency / (model.fps / 2),
                delta=0.001,
            )
            lost = motion.process(None, 60.05)
            self.assertIsNone(
                model._features(self.metrics(**lost), [], [], frames, "video", 0, 1)
            )
    def test_minimum_event_durations_reach_physical_detector(self):
        model = self.make_engine()
        with patch.object(engine, "CameraMetrics") as factory, patch.object(
            engine, "read_video", return_value=[]
        ):
            list(model.process_video("video.mp4", 0))
            options = factory.call_args.kwargs
        camera = engine.CameraMetrics(**options)
        self.assertEqual(camera.head_motion.window, 60)
        self.assertEqual(camera.mouth.fsm.minimum, 4000)
        self.assertEqual(camera.nod.minimum, 800)
        for detector, duration in (
            (camera.mouth.fsm, 3.9),
            (camera.mouth.fsm, 4.0),
            (camera.nod, 0.7),
            (camera.nod, 0.8),
        ):
            detector.reset()
            detector.process(0, 0)
            detector.process(1, 0.1)
            for i in range(2, round(duration * 10) + 1):
                detector.process(1, i / 10)
            event = detector.process(0, (round(duration * 10) + 1) / 10)
            self.assertEqual(event["event_done"], duration in (4.0, 0.8))
        camera.close()

    def test_window_schedule_neural_sampling_and_quality_gate(self):
        metrics = self.metrics(
            2,
            head_motion_resolution_hz=0.5,
            head_motion_frequency_hz=0.5,
            blink_rate_per_min=0,
        )
        frames = [(np.full((3, 4, 8), i, np.uint8), i / 10) for i in range(30)]
        with patch.object(engine, "NeuralNetwork") as neural, patch.object(
            engine, "CameraMetrics"
        ) as camera, patch.object(engine, "read_video", return_value=iter(frames)):
            neural.return_value.eval.return_value.return_value = 0.7
            camera.return_value.process.return_value = metrics
            model = engine.Engine(
                ".",
                fps=10,
                stride_sec=1,
                neural_sec=1,
                neural_size=8,
                window_sec=2,
            )
            rows = list(model.process_video("video.mp4", 1))
            self.assertEqual(
                [(r["start_second"], r["end_second"]) for r in rows], [(0, 2), (1, 3)]
            )
            batch = model.neural.call_args.args[0]
            self.assertEqual(batch.shape, (5, 3, 8, 8))
            self.assertEqual(batch[:, 0, 3, 3].tolist(), [20, 22, 24, 26, 28])
            self.assertEqual(rows[0]["perclos"], 0.1)
            self.assertEqual(rows[0]["blink_frequency"], 0)
            self.assertEqual(rows[0]["feature_encoding"], FEATURE_ENCODING)
            self.assertEqual(rows[0]["dominant_head_motion_frequency"], 0.1)
            camera.return_value.close.assert_called_once()
            model.neural.reset_mock()
            with patch.object(engine, "read_video", return_value=iter(frames)):
                resumed = list(
                    model.process_video("video.mp4", 1, saved_windows={(0, 2)})
                )
            self.assertEqual(
                [(r["start_second"], r["end_second"]) for r in resumed], [(1, 3)]
            )
            self.assertEqual(camera.return_value.process.call_count, 60)
            model.neural.assert_called_once()
            for perclos in (0, 5, 10, 15):
                sample = model._features(
                    metrics | {"blink_rate_per_min": 4, "perclos_pct": perclos},
                    [],
                    [],
                    batch,
                    "video",
                    0,
                    1,
                )
                self.assertEqual(sample["blink_frequency"], 4 / model.normalizer.scales["blink_frequency"])
            for missing in (
                {"eye_ready": False},
                {"eye_observed_sec": 0},
                {"head_motion_resolution_hz": None},
            ):
                self.assertIsNone(
                    model._features(metrics | missing, [], [], [], "video", 0, 1)
                )


if __name__ == "__main__":
    unittest.main()
