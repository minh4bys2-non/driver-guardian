import itertools
import unittest

import cv2
import numpy as np

from ai.PhysicalBranch.adaptive_hmm_fsm import AdaptiveHMM
from ai.PhysicalBranch.head_pose_estimation import HeadPoseEstimator
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow, StateMachine, WindowRatio


class PhysicalRegressionTests(unittest.TestCase):
    def test_forward_backward_matches_enumeration_of_all_state_paths(self):
        model = AdaptiveHMM()
        model.pi = np.array([.7, .3])
        model.A = np.array([[.8, .2], [.1, .9]])
        model.means = np.array([.3, .06])
        model.vars = np.array([.01, .003])
        x = np.array([.3, .1, .05, .25])
        emission = np.exp(model._log_emission_prob(x))
        expected_gamma = np.zeros((4, 2))
        expected_xi = np.zeros((3, 2, 2))
        total = 0
        for states in itertools.product(range(2), repeat=len(x)):
            probability = model.pi[states[0]] * emission[0, states[0]]
            for t in range(1, len(x)):
                probability *= model.A[states[t - 1], states[t]] * emission[t, states[t]]
            total += probability
            for t, state in enumerate(states):
                expected_gamma[t, state] += probability
                if t:
                    expected_xi[t - 1, states[t - 1], state] += probability
        gamma, xi = model._forward_backward(x)
        np.testing.assert_allclose(gamma, expected_gamma / total, atol=1e-12)
        np.testing.assert_allclose(xi, expected_xi / total, atol=1e-12)

    def test_hmm_fit_and_adaptation_preserve_probability_invariants(self):
        rng = np.random.default_rng(23)
        x = np.r_[rng.normal(.3, .01, 40), rng.normal(.06, .01, 40)]
        for direction in ('low', 'high'):
            model = AdaptiveHMM(positive_state=direction, adapt_interval=80).fit_initial(x)
            self.assertEqual(model.means[1] < model.means[0], direction == 'low')
            for value in x:
                _, posterior = model.predict(value)
                self.assertAlmostEqual(posterior.sum(), 1)
                self.assertTrue((posterior >= 0).all())
                model.update_online(value)
            np.testing.assert_allclose(model.A.sum(axis=1), 1)
            self.assertAlmostEqual(model.pi.sum(), 1)
            self.assertTrue((model.vars >= model.min_variance).all())

    def test_pose_recovers_combined_rotations_with_calibrated_intrinsics(self):
        matrix = np.array([[900., 0, 320], [0, 850, 240], [0, 0, 1]])
        estimator = HeadPoseEstimator(640, 480, camera_matrix=matrix)
        for angles in ((0, 0, 0), (20, -25, 15), (-30, 35, -20)):
            x, y, z = np.deg2rad(angles)
            rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
            ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
            rotation, _ = cv2.Rodrigues(rz @ ry @ rx)
            points, _ = cv2.projectPoints(estimator.MODEL, rotation, np.array([10., 20., 2000.]), matrix, estimator.distortion)
            np.testing.assert_allclose(estimator.estimate(points.reshape(6, 2)), angles, atol=1e-5)

    def test_ratio_matches_independent_interval_integral(self):
        rng = np.random.default_rng(4)
        ratio = WindowRatio(1.7, .25)
        samples = []
        time = 0
        for _ in range(200):
            time += rng.uniform(.01, .4)
            state = rng.choice([None, False, True])
            samples.append((time, state))
            percent, observed = ratio.process(state, time)
            total = active = 0
            for (a, left), (b, right) in zip(samples, samples[1:]):
                if left is not None and right is not None and b - a <= .25:
                    duration = max(0, b - max(a, time - 1.7))
                    total += duration
                    active += duration * bool(left)
            self.assertAlmostEqual(observed, total)
            if total:
                self.assertAlmostEqual(percent, 100 * active / total)
            else:
                self.assertIsNone(percent)

    def test_event_duration_boundaries_at_multiple_frame_rates(self):
        for fps in (10, 20, 30):
            for duration, expected in ((.1, True), (2., True), (2.1, False)):
                detector = StateMachine(fps=fps)
                detector.process(0, 0)
                for i in range(1, round(duration * fps) + 1):
                    detector.process(1, i / fps)
                out = detector.process(0, (round(duration * fps) + 1) / fps)
                self.assertEqual(out['event_done'], expected)

    def test_static_is_zero_only_after_full_valid_window_and_recovers_after_loss(self):
        motion = HeadMotionWindow(1)
        for i in range(11):
            out = motion.process(12., i / 10)
            self.assertEqual(out['head_motion_frequency_hz'], 0. if i == 10 else None)
        self.assertIsNone(motion.process(None, 1.1)['head_motion_frequency_hz'])
        for i in range(12, 23):
            out = motion.process(12., i / 10)
            self.assertEqual(out['head_motion_frequency_hz'], 0. if i == 22 else None)

    def test_fft_recovers_known_frequencies_at_multiple_sampling_rates(self):
        for fps in (10, 20, 30):
            for frequency in (.25, 1., 2.):
                motion = HeadMotionWindow(4)
                for i in range(4 * fps + 1):
                    out = motion.process(3 * np.sin(2 * np.pi * frequency * i / fps), i / fps)
                self.assertAlmostEqual(out['head_motion_frequency_hz'], frequency, delta=out['head_motion_resolution_hz'])
