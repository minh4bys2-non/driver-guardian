import json
import unittest
from dataclasses import fields
from unittest.mock import patch

import cv2
import numpy as np

from ai.PhysicalBranch.adaptive_hmm_fsm import AdaptiveHMM_FSM
from ai.PhysicalBranch.brand import normalize_metrics
from ai.PhysicalBranch.camera_metrics import CameraMetrics
from ai.PhysicalBranch.head_pose_estimation import HeadPoseEstimator
from ai.PhysicalBranch.interface import AIResult
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow, StateMachine, WindowRatio


class TemporalTests(unittest.TestCase):
    def test_complete_cycles_and_window_expiry(self):
        fsm = StateMachine('eye', window_size_sec=2)
        fsm.process(1, 0)
        self.assertFalse(fsm.process(0, .125)['event_done'])
        fsm.process(1, .25)
        event = fsm.process(0, .5)
        self.assertTrue(event['event_done'])
        self.assertEqual(event['last_valid_duration_ms'], 250)
        self.assertEqual(event['event_count'], 1)
        self.assertEqual(event['rate_per_minute'], 30)
        self.assertEqual(fsm.process(0, 2.5)['event_count'], 0)

    def test_missing_and_long_events_are_not_counted(self):
        for interruption in ('missing', 'gap', 'prolonged'):
            with self.subTest(interruption=interruption):
                fsm = StateMachine('pitch', min_duration_ms=100, max_duration_ms=500)
                fsm.process(0, 0)
                fsm.process(1, .125)
                if interruption == 'missing':
                    fsm.process(None, .25)
                    fsm.process(1, .375)
                    end = .5
                elif interruption == 'gap':
                    fsm.process(1, 1)
                    end = 1.125
                else:
                    for t in (.25, .375, .5, .625, .75):
                        fsm.process(1, t)
                    end = .875
                self.assertFalse(fsm.process(0, end)['event_done'])
                self.assertIsNone(fsm.last_valid_duration)

    def test_ratio_clips_window_and_excludes_gaps(self):
        ratio = WindowRatio(.3)
        ratio.process(True, 0)
        ratio.process(False, .2)
        percentage, observed = ratio.process(False, .4)
        self.assertAlmostEqual(percentage, 100 / 3)
        self.assertAlmostEqual(observed, .3)
        self.assertEqual(ratio.process(None, .8), (None, 0))
        self.assertEqual(ratio.process(True, 1), (None, 0))
        self.assertEqual(ratio.process(True, 2), (None, 0))

    def test_separate_event_and_ratio_windows_and_p80_boundary(self):
        eye = AdaptiveHMM_FSM('eye', fps=4, init_duration_sec=1,
                              window_size_sec=10, ratio_window_sec=2)
        for i in range(4):
            eye.process(.3, i * .125)
        threshold = eye.normal - .8 * (eye.normal - eye.event_reference)
        eye.process(threshold, .5)
        out = eye.process(threshold, .625)
        self.assertGreater(out['perclos'], 0)
        self.assertEqual(eye.fsm.window, 10)
        self.assertEqual(eye.ratio.window, 2)

    def test_invalid_timestamps_and_windows(self):
        for cls in (WindowRatio, HeadMotionWindow):
            for value in (0, -1, float('inf'), float('nan'), True):
                with self.subTest(cls=cls, value=value), self.assertRaises(ValueError):
                    cls(value)
        for metric in (StateMachine(), WindowRatio(), HeadMotionWindow()):
            metric.process(0, 1)
            for timestamp in (1, .5, float('nan'), float('inf')):
                with self.subTest(metric=metric, timestamp=timestamp), self.assertRaises(ValueError):
                    metric.process(0, timestamp)


class MotionTests(unittest.TestCase):
    def test_fft_with_irregular_timestamps(self):
        motion = HeadMotionWindow(20)
        rng = np.random.default_rng(17)
        for t in np.r_[0, np.cumsum(rng.uniform(.08, .12, 220))]:
            out = motion.process(10 + 5 * np.sin(2 * np.pi * .3 * t), t)
            if t < 20:
                self.assertIsNone(out['head_motion_frequency_hz'])
        self.assertAlmostEqual(out['head_motion_frequency_hz'], .3, delta=.02)
        self.assertAlmostEqual(out['pitch_mean_deg'], 10, delta=.1)
        self.assertAlmostEqual(out['head_motion_resolution_hz'], .05, delta=.001)
        self.assertEqual(out['head_motion_observed_sec'], 20)

    def test_slow_motion_needs_long_window(self):
        motion = HeadMotionWindow(80)
        for t in np.linspace(0, 80, 801):
            out = motion.process(4 * np.sin(2 * np.pi * .025 * t), t)
        self.assertAlmostEqual(out['head_motion_frequency_hz'], .025, delta=.001)

    def test_constant_signal_and_discontinuities(self):
        motion = HeadMotionWindow(1)
        for t in np.linspace(0, 1, 11):
            out = motion.process(15, t)
        self.assertIsNone(out['head_motion_frequency_hz'])
        self.assertAlmostEqual(out['pitch_mean_deg'], 15)
        self.assertIsNone(motion.process(None, 1.1)['pitch_mean_deg'])
        self.assertEqual(motion.process(10, 1.2)['head_motion_observed_sec'], 0)
        self.assertEqual(motion.process(10, 2)['head_motion_observed_sec'], 0)


class PipelineTests(unittest.TestCase):
    def test_eye_mouth_pipeline_with_independent_windows(self):
        monitor = CameraMetrics(fps=10, window_sec=4, calibration_frames=1,
                                windows={'yawn': 2, 'perclos': 1},
                                eye_options={'init_duration_sec': .4},
                                mouth_options={'init_duration_sec': .4, 'min_duration_ms': 200})
        with patch.object(HeadPoseEstimator, 'estimate', return_value=(0, 0, 0)):
            for i in range(16):
                points = np.zeros((468, 2))
                ear = .06 if 6 <= i < 9 else .3
                mar = .65 if 6 <= i < 13 else .1
                for indices, ratio in ((monitor.LEFT_EYE, ear), (monitor.RIGHT_EYE, ear),
                                       (monitor.MOUTH, mar)):
                    h = 5 * ratio
                    points[indices] = [(0, 0), (2, -h), (8, -h), (10, 0), (8, h), (2, h)]
                result = monitor.process_landmarks(points, (640, 480), i / 10)
        self.assertEqual(result['blink_count'], 1)
        self.assertEqual(result['yawn_count'], 1)
        self.assertAlmostEqual(result['blink_duration_ms'], 300)
        self.assertAlmostEqual(result['yawn_duration_ms'], 700)
        self.assertEqual(result['blink_rate_per_min'], 15)
        self.assertEqual(result['yawning_frequency_per_min'], 30)
        self.assertAlmostEqual(result['perclos_pct'], 30)
        monitor.set_windows(blink=2)
        with patch.object(HeadPoseEstimator, 'estimate', return_value=(0, 0, 0)):
            result = monitor.process_landmarks(points, (640, 480), 1.6)
        self.assertTrue(result['eye_ready'])
        self.assertEqual(result['blink_count'], 0)
        self.assertEqual(result['yawn_count'], 1)
        self.assertEqual(result['windows_sec']['blink'], 2)

    def test_run_callback_can_change_windows_and_resources_close(self):
        from unittest.mock import Mock

        capture, tracker = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        tracker.process.return_value = None
        monitor = CameraMetrics()
        monitor.tracker = tracker
        received = []

        def callback(metrics):
            received.append(metrics)
            monitor.set_windows(perclos=15)

        with patch('cv2.VideoCapture', return_value=capture), patch('builtins.print'):
            count = monitor.run(display=False, use_video_time=True, max_frames=3, on_metrics=callback)
        self.assertEqual(count, 3)
        self.assertEqual([m['timestamp_sec'] for m in received], [0, 1 / 30, 2 / 30])
        self.assertEqual([m['windows_sec']['perclos'] for m in received], [60, 15, 15])
        capture.release.assert_called_once()
        tracker.close.assert_called_once()

    def test_window_changes_preserve_calibration_and_models(self):
        monitor = CameraMetrics(window_sec=30, windows={'perclos': 60, 'head_motion': 20})
        for i in range(monitor.eye.init_frames):
            monitor.eye.process(.3, i / 30)
        monitor.neutral = np.array([1., 2., 3.])
        means = monitor.eye.hmm.means.copy()
        monitor.eye.fsm.events.append(4)
        monitor.mouth.fsm.events.append(4)
        monitor.nodding = True
        monitor.set_windows(blink=15, nod=10)
        self.assertEqual(monitor.windows_sec['perclos'], 60)
        self.assertEqual(monitor.windows_sec['blink'], 15)
        self.assertFalse(monitor.eye.fsm.events)
        self.assertEqual(list(monitor.mouth.fsm.events), [4])
        self.assertFalse(monitor.nodding)
        self.assertTrue(monitor.eye.initialized)
        np.testing.assert_array_equal(monitor.eye.hmm.means, means)
        np.testing.assert_array_equal(monitor.neutral, [1, 2, 3])
        saved = monitor.windows_sec
        for options in ({'perclos': 0}, {'window_sec': float('inf')}, {'bad': 20},
                        {'blink': 3, 'nod': float('nan')}):
            with self.assertRaises(ValueError):
                monitor.set_windows(**options)
            self.assertEqual(monitor.windows_sec, saved)
        monitor.set_windows(40, head_motion=80)
        self.assertEqual(monitor.window_sec, 40)
        self.assertEqual(monitor.windows_sec, dict.fromkeys(saved, 40) | {'head_motion': 80})

    def test_relative_pitch_both_directions_and_event_duration(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                monitor = CameraMetrics(calibration_frames=1, nod_duration_ms=(200, 2000))
                with patch.object(HeadPoseEstimator, 'estimate') as estimate:
                    results = []
                    for i, delta in enumerate((0, 15, 16, 10, 8, 0)):
                        estimate.return_value = (30 + sign * delta, 5, -2)
                        results.append(monitor.process_landmarks(np.zeros((468, 2)), (640, 480), i * .125))
                self.assertEqual(results[1]['pitch_amplitude_deg'], 15)
                self.assertEqual(results[1]['pitch_deg'], sign * 15)
                self.assertTrue(results[3]['nodding'])
                self.assertTrue(results[4]['nod_detected'])
                self.assertEqual(results[-1]['nod_count'], 1)
                self.assertEqual(results[-1]['nod_event_duration_ms'], 375)

    def test_dto_and_missing_face(self):
        monitor = CameraMetrics(window_sec=20, windows={'nod': 10})
        result = monitor.process_landmarks(None, (640, 480), 0)
        self.assertFalse(result['face_detected'])
        self.assertIsNone(result['head_motion_frequency_hz'])
        dto = AIResult.from_metrics(result | {'fps': 30})
        self.assertEqual(json.loads(dto.to_json())['windows_sec']['nod'], 10)
        self.assertFalse(set(result) - {f.name for f in fields(AIResult)})
        with self.assertRaises(ValueError):
            AIResult(0, ear=float('nan')).to_json()
        canvas = monitor.draw(np.zeros((480, 640, 3), dtype=np.uint8), result)
        self.assertEqual(canvas.shape[1], 1100)

    def test_pose_projection_and_resize_reset(self):
        estimator = HeadPoseEstimator(640, 480)
        rotation = np.array([.2, -.1, .05])
        points, _ = cv2.projectPoints(estimator.MODEL, rotation, np.array([0., 0., 1500.]),
                                     estimator.camera_matrix, estimator.distortion)
        expected = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
        np.testing.assert_allclose(estimator.estimate(points.reshape(6, 2)), expected, atol=1e-5)
        with self.assertRaises(ValueError):
            estimator.estimate(np.zeros((6, 2)))
        self.assertIsNone(estimator.rotation)
        monitor = CameraMetrics(calibration_frames=1)
        landmarks = np.zeros((468, 2))
        landmarks[estimator.LANDMARKS] = points.reshape(6, 2)
        self.assertTrue(monitor.process_landmarks(landmarks, (640, 480), 0)['head_ready'])
        result = monitor.process_landmarks(None, (1280, 720), .1)
        self.assertFalse(result['head_calibrated'])
        self.assertEqual(result['head_motion_observed_sec'], 0)

    def test_reference_scores_are_configurable(self):
        result = normalize_metrics({'blink_rate_per_min': 3, 'mar': .73, 'pitch_amplitude_deg': 14,
                                    'head_motion_frequency_hz': .025, 'perclos_pct': 10},
                                   scales={'perclos_pct': ((0, 0), (20, 1))})
        self.assertIsNone(result['blink_rate_per_min'])
        self.assertIsNone(result['mar'])
        self.assertIsNone(result['head_motion_frequency_hz'])
        self.assertEqual(result['pitch_amplitude_deg'], .5)
        self.assertEqual(result['perclos_pct'], .5)
        self.assertNotIn('pitch_deg', result)
        for invalid in ({'pom_pct': 101}, {'ear': True}, {'head_motion_frequency_hz': -1}):
            with self.assertRaises(ValueError):
                normalize_metrics(invalid)


if __name__ == '__main__':
    unittest.main()
