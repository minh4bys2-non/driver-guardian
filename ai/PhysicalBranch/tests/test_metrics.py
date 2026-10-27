import json
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import cv2
import numpy as np

from ai.PhysicalBranch.adaptive_hmm_fsm import AdaptiveHMM, AdaptiveHMM_FSM
from ai.PhysicalBranch.brand import normalize_metrics
from ai.PhysicalBranch.camera_metrics import CameraMetrics
from ai.PhysicalBranch.config import DEFAULT_CONFIG, PhysicalConfig
from ai.PhysicalBranch.head_pose_estimation import HeadPoseEstimator
from ai.PhysicalBranch.interface import AIResult
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow, StateMachine, WindowRatio


class ConfigTests(unittest.TestCase):
    def test_config_reaches_detectors_pose_and_independent_windows(self):
        config = PhysicalConfig(
            fps=10, window_sec=12, max_gap_sec=.15, init_duration_sec=.4,
            blink_duration_ms=(200, 900), yawn_duration_ms=(1000, 3000),
            nod_duration_ms=(300, 1200), head_motion_min_amplitude_deg=.4,
            min_variance=.002, em_iterations=3, learning_rate=.02,
            windows={"perclos": 8}, mouth_options={"adapt_interval": 20},
            distortion_coeffs=(.01, 0, 0, 0),
            camera_matrix=((500, 0, 320), (0, 500, 240), (0, 0, 1)),
        )
        camera = CameraMetrics(config=config, windows={"head_motion": 10})
        self.assertEqual(camera.eye.init_frames, 4)
        self.assertEqual(camera.eye.hmm.min_variance, .002)
        self.assertEqual(camera.eye.hmm.em_iterations, 3)
        self.assertEqual(camera.eye.hmm.learning_rate, .02)
        self.assertEqual(camera.mouth.hmm.adapt_interval, 20)
        self.assertEqual(camera.eye.fsm.minimum, 200)
        self.assertEqual(camera.mouth.fsm.maximum, 3000)
        self.assertEqual(camera.nod.minimum, 300)
        self.assertEqual(camera.nod.max_gap, .15)
        self.assertEqual(camera.eye.ratio.window, 8)
        self.assertEqual(camera.head_motion.window, 10)
        self.assertEqual(camera.head_motion.min_amplitude, .4)
        camera.process_landmarks(None, (640, 480), 0)
        np.testing.assert_array_equal(camera.estimator.distortion, config.distortion_coeffs)
        np.testing.assert_array_equal(camera.estimator.camera_matrix, config.camera_matrix)
        explicit = np.diag([600, 600, 1])
        np.testing.assert_array_equal(
            HeadPoseEstimator(640, 480, camera_matrix=explicit, config=config).camera_matrix, explicit)
        camera.reset()
        self.assertEqual(camera.eye.hmm.min_variance, .002)
        self.assertEqual(config.windows, {"perclos": 8})
        self.assertEqual(DEFAULT_CONFIG.windows, {})
        self.assertEqual(CameraMetrics().eye.fsm.minimum, 100)

    def test_config_changes_p80_and_fallback_without_mutating_defaults(self):
        config = replace(DEFAULT_CONFIG, fps=10, init_duration_sec=.4,
                         eye_closed_ratio=.1, perclos_closed_fraction=.9,
                         resting_prior=(.98, .02))
        eye = AdaptiveHMM_FSM(config=config)
        for _ in range(2):
            for i in range(4):
                result = eye.process(.3, i / 10)
            self.assertAlmostEqual(eye.event_reference, .03)
            self.assertAlmostEqual(result['p80_threshold'], .057)
            np.testing.assert_array_equal(eye.hmm.posterior, [.98, .02])
            eye.reset()
        self.assertEqual(DEFAULT_CONFIG.perclos_closed_fraction, .8)

    def test_run_uses_config_and_allows_explicit_none_overrides(self):
        config = replace(DEFAULT_CONFIG, camera_source='test.mp4', display=False,
                         use_video_time=True, max_frames=1, logdir='runs/test', output_interval_sec=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch('cv2.VideoCapture') as capture, patch('cv2.imshow') as show, \
                patch('ai.PhysicalBranch.camera_metrics.TensorBoardLogger') as logger, \
                patch('builtins.print') as output:
            cap = capture.return_value
            cap.isOpened.return_value = True
            cap.get.return_value = 30
            for overrides, expected in (({}, 1), ({'max_frames': None, 'logdir': None}, 3)):
                cap.read.side_effect = [(True, frame)] * 3 + [(False, None)]
                camera = CameraMetrics(config=config)
                received = []
                with patch.object(camera, 'process', side_effect=lambda frame, t: {'timestamp_sec': t}):
                    self.assertEqual(camera.run(on_metrics=received.append, **overrides), expected)
                self.assertEqual([m['timestamp_sec'] for m in received], [i / 30 for i in range(expected)])
            capture.assert_called_with('test.mp4')
            logger.assert_called_once_with('runs/test')
            logger.return_value.close.assert_called_once()
            self.assertEqual(output.call_count, 2)
            show.assert_not_called()

    def test_hmm_adaptation_requires_configured_state_weight(self):
        hmm = AdaptiveHMM(config=replace(DEFAULT_CONFIG, adapt_interval=4, min_adapt_state_weight=5))
        hmm.fit_initial([.1, .1, .3, .3])
        means = hmm.means.copy()
        for value in (.1, .3, .1, .3):
            self.assertFalse(hmm.update_online(value))
        np.testing.assert_array_equal(hmm.means, means)
        self.assertEqual(len(hmm.buffer), 0)


class TemporalTests(unittest.TestCase):
    def test_default_yawn_duration_matches_data_pipeline(self):
        for duration, expected in ((3.9, False), (4.0, True), (7.5, True), (7.6, False)):
            with self.subTest(duration=duration):
                fsm = StateMachine('mouth')
                fsm.process(0, 0)
                for i in range(round(duration * 10)):
                    fsm.process(1, (i + 1) / 10)
                result = fsm.process(0, duration + .1)
                self.assertEqual(result['event_done'], expected)

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


class DetectorTests(unittest.TestCase):
    def test_default_warmup_allows_eighty_percent_coverage_after_first_window(self):
        detector = AdaptiveHMM_FSM()
        for i in range(1800):
            result = detector.process(.3, i / 30)
            if i == 899:
                self.assertLess(result['observed_seconds'], 30 * .8)
        self.assertEqual(detector.init_frames, 300)
        self.assertGreaterEqual(result['observed_seconds'], 30 * .8)
        self.assertEqual(result['perclos'], 0)

    def test_implicit_timeline_clears_hmm_at_gaps(self):
        implicit = AdaptiveHMM_FSM(fps=2, init_duration_sec=2)
        explicit = AdaptiveHMM_FSM(fps=2, init_duration_sec=2)
        for i, value in enumerate((.3, .3, .3, .3, .06, .3)):
            self.assertEqual(implicit.process(value), explicit.process(value, i / 2))
        self.assertEqual(len(implicit.hmm.buffer), 1)

    def test_invalid_ratios_do_not_calibrate_or_complete_events(self):
        for mode in ('eye', 'mouth'):
            with self.subTest(mode=mode):
                detector = AdaptiveHMM_FSM(mode, fps=10, init_duration_sec=.4)
                for i in range(4):
                    self.assertTrue(detector.process(-.1, i / 10)['signal_missing'])
                self.assertFalse(detector.initialized)
                self.assertFalse(detector.samples)
                for i in range(4, 8):
                    detector.process(.3, i / 10)
                self.assertTrue(detector.initialized)
                self.assertIsNone(detector.process(-.1, .8)['state'])
                self.assertIsNone(detector.hmm.posterior)

    def test_invalid_initialization_is_rejected(self):
        for name in ('fps', 'init_duration_sec'):
            for value in (0, -1, float('nan'), float('inf'), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AdaptiveHMM_FSM(**{name: value})
        for options in ({'adapt_interval': 2.5}, {'adapt_interval': True},
                        {'em_iterations': 1.5}, {'em_iterations': True},
                        {'min_variance': float('nan')}, {'min_variance': float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AdaptiveHMM(**options)

    def test_invalid_online_observation_preserves_model(self):
        detector = AdaptiveHMM_FSM(fps=10, init_duration_sec=.4)
        for _ in range(4):
            detector.process(.3)
        saved = list(detector.hmm.buffer)
        means = detector.hmm.means.copy()
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                detector.hmm.update_online(value)
            self.assertEqual(list(detector.hmm.buffer), saved)
            np.testing.assert_array_equal(detector.hmm.means, means)


class MotionTests(unittest.TestCase):
    def test_default_fft_waits_for_thirty_seconds(self):
        motion = HeadMotionWindow()
        for t in np.linspace(0, 30, 301):
            result = motion.process(5 * np.sin(2 * np.pi * .1 * t), t)
            if t < 30:
                self.assertIsNone(result['head_motion_frequency_hz'])
        self.assertAlmostEqual(result['head_motion_frequency_hz'], .1, delta=.002)

    def test_mean_amplitude_integrates_zero_crossing(self):
        motion = HeadMotionWindow(.1)
        motion.process(-10, 0)
        result = motion.process(10, .2)
        self.assertAlmostEqual(result['pitch_mean_deg'], 5)
        self.assertAlmostEqual(result['pitch_mean_amplitude_deg'], 5)
        motion = HeadMotionWindow(1)
        motion.process(-10, 0)
        result = motion.process(10, .2)
        self.assertAlmostEqual(result['pitch_mean_deg'], 0)
        self.assertAlmostEqual(result['pitch_mean_amplitude_deg'], 5)

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
        self.assertEqual(out['head_motion_frequency_hz'], 0.0)
        self.assertAlmostEqual(out['pitch_mean_deg'], 15)
        self.assertIsNone(motion.process(None, 1.1)['pitch_mean_deg'])
        self.assertEqual(motion.process(10, 1.2)['head_motion_observed_sec'], 0)
        self.assertEqual(motion.process(10, 2)['head_motion_observed_sec'], 0)


class PipelineTests(unittest.TestCase):
    def test_invalid_camera_configuration_fails_before_tracking(self):
        options = [
            {'calibration_frames': value} for value in (0, 1.5, True, float('nan'), float('inf'))
        ]
        options += [{name: value} for name in ('detection_confidence', 'tracking_confidence')
                    for value in (-.1, 1.1, float('nan'), float('inf'))]
        options += [{'nod_duration_ms': value} for value in ((), (800,), (800, 3500, 4000))]
        options.append({'nod_direction': True})
        for config in options:
            with self.subTest(config=config), self.assertRaises(ValueError):
                CameraMetrics(**config)

    def test_invalid_run_configuration_does_not_open_camera(self):
        with patch('cv2.VideoCapture') as capture:
            for config in ({'output_interval_sec': float('nan')},
                           {'output_interval_sec': float('inf')},
                           {'max_frames': 1.5}, {'max_frames': True}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    CameraMetrics().run(**config)
            capture.assert_not_called()

    def test_invalid_camera_geometry_is_rejected(self):
        for value in (float('nan'), float('inf'), 0):
            with self.assertRaises(ValueError):
                HeadPoseEstimator(value, 480)
        for matrix in (np.zeros((3, 3)), np.diag([-1, 1, 1]), np.diag([1, 1, 2])):
            with self.assertRaises(ValueError):
                HeadPoseEstimator(640, 480, camera_matrix=matrix)
        for distortion in (np.zeros(3), np.zeros((2, 2)), np.full(4, float('nan'))):
            with self.assertRaises(ValueError):
                HeadPoseEstimator(640, 480, distortion_coeffs=distortion)
        for shape in ((5,), (1, 5), (5, 1)):
            estimator = HeadPoseEstimator(640, 480, distortion_coeffs=np.zeros(shape))
            self.assertEqual(estimator.distortion.shape, shape)

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
        self.assertEqual([m['windows_sec']['perclos'] for m in received], [30, 15, 15])
        capture.release.assert_called_once()
        tracker.close.assert_called_once()

    def test_quit_counts_processed_frame_and_closes_resources(self):
        from unittest.mock import Mock

        capture, tracker = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        tracker.process.return_value = None
        monitor = CameraMetrics()
        monitor.tracker = tracker
        with patch('cv2.VideoCapture', return_value=capture), patch('builtins.print'), \
                patch('cv2.imshow'), patch('cv2.waitKey', return_value=ord('q')), \
                patch('cv2.destroyAllWindows') as destroy:
            self.assertEqual(monitor.run(use_video_time=True), 1)
        capture.release.assert_called_once()
        tracker.close.assert_called_once()
        destroy.assert_called_once()

    def test_calibration_wraps_angles_and_restarts_after_gap(self):
        monitor = CameraMetrics(calibration_frames=2)
        points = np.zeros((468, 2))
        with patch.object(HeadPoseEstimator, 'estimate') as estimate:
            estimate.return_value = (30, 0, 0)
            monitor.process_landmarks(points, (640, 480), 0)
            estimate.return_value = (179, 0, 0)
            result = monitor.process_landmarks(points, (640, 480), 1)
            self.assertFalse(result['head_calibrated'])
            estimate.return_value = (-179, 0, 0)
            result = monitor.process_landmarks(points, (640, 480), 1.1)
        self.assertTrue(result['head_calibrated'])
        self.assertAlmostEqual(result['pitch_deg'], 1)
        self.assertFalse(result['nodding'])

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

    def test_face_tracker_tasks_api_methods_and_output_format(self):
        from ai.PhysicalBranch.camera_metrics import FaceTracker
        from unittest.mock import MagicMock, patch

        mock_detector = MagicMock()
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.25
        mock_result = MagicMock()
        mock_result.face_landmarks = [[mock_landmark] * 478]
        mock_detector.detect.return_value = mock_result

        with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=mock_detector), \
             patch("mediapipe.Image"):
            tracker = FaceTracker()
            self.assertEqual(tracker._mode, "tasks")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            landmarks = tracker.process(frame)
            self.assertIsNotNone(landmarks)
            self.assertEqual(landmarks.shape, (478, 2))
            np.testing.assert_allclose(landmarks[0], [320.0, 120.0])

            # Test no face detected
            mock_result.face_landmarks = []
            self.assertIsNone(tracker.process(frame))

            # Test close
            tracker.close()
            mock_detector.close.assert_called_once()
            self.assertIsNone(tracker.detector)

    def test_face_tracker_missing_model_raises_filenotfound(self):
        from ai.PhysicalBranch.camera_metrics import FaceTracker
        from ai.PhysicalBranch.config import PhysicalConfig
        cfg = PhysicalConfig(face_landmarker_model_path="non_existent_model.task")
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(FileNotFoundError):
                FaceTracker(config=cfg)


if __name__ == '__main__':
    unittest.main()
