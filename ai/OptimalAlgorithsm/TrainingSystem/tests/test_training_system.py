import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness, load_jsonl
from ai.OptimalAlgorithsm.TrainingSystem.cmdpsofs import (
    CMDPSOFS, CMDPSOFSConfig, ParetoArchive, RunLogger, crowding_distance, dominates,
)
from ai.OptimalAlgorithsm.TrainingSystem.IBGWO4 import IBGWO4, IBGWO4Config, IBGWO4Logger, pareto_order
from ai.OptimalAlgorithsm.TrainingSystem.MOHHO import MOHHO, MOHHOConfig, MOHHOLogger
from ai.OptimalAlgorithsm.TrainingSystem.NSPSOFS import NSPSOFS, NSPSOFSConfig, NSPSOFSLogger

ALGORITHMS = (
    (CMDPSOFS, CMDPSOFSConfig, RunLogger),
    (IBGWO4, IBGWO4Config, IBGWO4Logger),
    (MOHHO, MOHHOConfig, MOHHOLogger),
    (NSPSOFS, NSPSOFSConfig, NSPSOFSLogger),
)


class FitnessTests(unittest.TestCase):
    def test_prediction_formulas_and_selection_boundary(self):
        X = np.zeros((3, 8))
        X[:, :2] = [[.2, .8], [.8, .2], [1, 1]]
        fitness = Fitness(X)
        candidate = np.zeros(9)
        candidate[[0, 1, 8]] = [.5, 1, .6]
        np.testing.assert_array_equal(fitness.predict(candidate), [1, 0, 1])
        np.testing.assert_array_equal(fitness.predict(candidate, 'noisy_or'), [1, 0, 1])
        candidate[8] = .7
        np.testing.assert_array_equal(fitness.predict(candidate), [0, 0, 1])
        np.testing.assert_array_equal(fitness.predict(candidate, 'noisy_or'), [1, 0, 1])
        candidate[:8] = .499
        np.testing.assert_array_equal(fitness.predict(candidate), [0, 0, 0])

    def test_known_confusion_matrix(self):
        candidate = np.zeros(9)
        candidate[:2] = .5
        actual = Fitness.evaluate(candidate, [1, 0, 1, 1, 0], [1, 1, 0, 1, 0])
        np.testing.assert_allclose(actual, [1/3, 1/3, 2/8])
        np.testing.assert_array_equal(Fitness.evaluate(candidate, [0, 0], [0, 0]), [1, 1, .25])

    def test_invalid_inputs_are_rejected_before_casting(self):
        for X in ([], np.zeros((2, 7)), np.full((2, 8), np.nan), np.full((2, 8), 1.1)):
            with self.subTest(X=X), self.assertRaises(ValueError):
                Fitness(X)
        fitness = Fitness(np.zeros((2, 8)))
        for value in (np.nan, np.inf, -1, 2):
            candidate = np.ones(9)
            candidate[8] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fitness.predict(candidate)
        for labels in ([1.9, .9], [256, 0], [1], [[1, 0]], [], [np.nan, 0]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                Fitness.evaluate(np.ones(9), [1, 0], labels)
        with self.assertRaises(ValueError):
            fitness.predict(np.ones(9), 'predict')
        with self.assertRaises(ValueError):
            Fitness.evaluate(np.ones(8), [0], [0])

    def test_jsonl_schema_and_line_numbers(self):
        valid = dict.fromkeys(FEATURES, .25) | {'label': 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            path.write_text('\n' + json.dumps(valid) + '\n')
            X, y = load_jsonl(path)
            self.assertEqual(X.shape, (1, 8))
            self.assertEqual(y.tolist(), [1])
            for row in ([], {}, valid | {'label': 1.9}, valid | {'label': True},
                        valid | {'label': 256}, valid | {FEATURES[0]: float('nan')},
                        valid | {FEATURES[0]: 1.1}, valid | {FEATURES[0]: [.5]}):
                path.write_text('\n' + json.dumps(row) + '\n')
                with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'line 2'):
                    load_jsonl(path)
            path.write_text('\n')
            with self.assertRaisesRegex(ValueError, 'empty'):
                load_jsonl(path)

    def test_old_and_risk_encodings_cannot_be_mixed(self):
        legacy = dict.fromkeys(FEATURES, .25) | {'label': 1}
        risk = legacy | {'feature_encoding': 'risk_v2'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            path.write_text(json.dumps(risk) + '\n')
            self.assertEqual(load_jsonl(path)[1].tolist(), [1])
            for records in ([legacy, risk], [risk, legacy], [risk, legacy | {'feature_encoding': 'risk_v1'}], [risk | {'feature_encoding': 'unknown'}]):
                path.write_text('\n'.join(json.dumps(row) for row in records))
                with self.subTest(records=records), self.assertRaises(ValueError):
                    load_jsonl(path)


class ParetoTests(unittest.TestCase):
    def test_dominance_and_front_order(self):
        self.assertTrue(dominates(np.array([0, 1, 0]), np.ones(3)))
        self.assertFalse(dominates(np.ones(3), np.ones(3)))
        objectives = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0], [.5, .5, 0], [2, 2, 2]])
        order = pareto_order(objectives)
        self.assertEqual(set(order[:3]), {1, 2, 3})
        self.assertEqual(order[-2:].tolist(), [0, 4])

    def test_constant_objective_does_not_change_crowding(self):
        np.testing.assert_array_equal(crowding_distance(np.ones((4, 3))), np.zeros(4))
        varying = np.array([[.2, .8], [0, 1], [1, 0], [.8, .2]])
        np.testing.assert_array_equal(
            crowding_distance(varying), crowding_distance(np.c_[varying, np.ones(4)]))

    def test_archive_keeps_only_nondominated_solutions(self):
        rng = np.random.default_rng(9)
        positions, objectives = rng.random((20, 9)), rng.random((20, 3))
        archive = ParetoArchive(20)
        archive.update(positions[:10], objectives[:10])
        archive.update(positions, objectives)
        objectives = objectives.astype(np.float32)
        expected = [i for i in range(20) if not any(dominates(o, objectives[i]) for o in objectives)]
        self.assertEqual(len(archive), len(expected))
        for o in archive.objectives:
            self.assertFalse(any(dominates(other, o) for other in objectives))
        smaller = ParetoArchive(3)
        smaller.update(positions, objectives)
        self.assertLessEqual(len(smaller), 3)
        leader = smaller.select_leader(rng)
        self.assertTrue(any(np.array_equal(leader, p) for p in smaller.positions))
        leader[:] = -1
        self.assertTrue((smaller.positions >= 0).all())


class OptimizerTests(unittest.TestCase):
    def test_configuration_and_label_validation(self):
        for optimizer, config, logger in ALGORITHMS:
            for options in ({'population_size': 0}, {'generations': 0}, {'archive_size': 0},
                            {'population_size': 3.5}, {'dimensions': 8}, {'dimensions': 9.0}):
                with self.subTest(algorithm=optimizer.__name__, options=options), self.assertRaises(ValueError):
                    config(**options)
            with self.assertRaises(ValueError):
                optimizer(Fitness(np.zeros((2, 8))), [1.9, .9], 'weighted_mean', config(), None)
            with self.assertRaises(ValueError):
                optimizer(Fitness(np.zeros((2, 8))), [0, 1], 'evaluate', config(), None)
        for options in ({'inertia': float('nan')}, {'velocity_limit': -1}, {'nonuniform_b': 0}):
            with self.assertRaises(ValueError):
                CMDPSOFSConfig(**options)
        with self.assertRaises(ValueError):
            ParetoArchive(-1)

    def test_all_algorithms_both_methods_and_reproducibility(self):
        rng = np.random.default_rng(21)
        X = rng.random((32, 8)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 1).astype(np.int8)
        fitness = Fitness(X)
        for optimizer_type, config_type, logger_type in ALGORITHMS:
            for method in ('weighted_mean', 'noisy_or'):
                results = []
                for repeat in range(2):
                    with self.subTest(algorithm=optimizer_type.__name__, method=method, repeat=repeat), \
                            tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                        cfg = config_type(population_size=6, generations=4, archive_size=8)
                        logger = logger_type(tmp)
                        logger.save_config(cfg, Path('data.jsonl'), method, len(y))
                        optimizer = optimizer_type(fitness, y, method, cfg, logger)
                        with patch.object(logger, 'save_plots'):
                            archive = optimizer.run()
                        self.assertTrue(0 < len(archive) <= 8)
                        self.assertEqual(len(logger.history), 4)
                        self.assertTrue(np.isfinite(archive.positions).all())
                        self.assertTrue(((archive.positions >= 0) & (archive.positions <= 1)).all())
                        for position, objectives in zip(archive.positions, archive.objectives):
                            np.testing.assert_allclose(objectives, optimizer._evaluate(position))
                            self.assertFalse(any(dominates(other, objectives) for other in archive.objectives))
                        rows = [json.loads(line) for line in logger.final_archive.read_text().splitlines()]
                        self.assertEqual(len(rows), len(archive))
                        self.assertTrue(all(r['num_features'] == len(r['selected_features']) for r in rows))
                        results.append((archive.positions.copy(), archive.objectives.copy()))
                for a, b in zip(results[0], results[1]):
                    np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
