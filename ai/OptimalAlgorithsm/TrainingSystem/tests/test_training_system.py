import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import (
    FEATURES, Fitness, classification_metrics, load_jsonl, selection_key,
)
from ai.OptimalAlgorithsm.TrainingSystem.engine import run_experiment, score_candidates, split_dataset
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
    def test_metrics_expose_constant_predictions(self):
        metrics = classification_metrics([0, 0, 1, 1, 1], [1, 1, 1, 1, 1])
        self.assertEqual((metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']), (3, 0, 2, 0))
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['precision'], .6)
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual(metrics['false_positive_rate'], 1)
        self.assertTrue(metrics['constant_prediction'])

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

    def test_cooccurrence_single_joint_and_weak_signals(self):
        X = np.zeros((6, 8))
        X[0, 0] = .8
        X[1, 0] = .6
        X[2, :2] = .6
        X[3, :] = .2
        X[5, :2] = 1
        candidate = np.ones(9)
        candidate[8] = .7
        fitness = Fitness(X)
        # Scores: .8, .6, .744, .232, 0, 1.
        np.testing.assert_array_equal(fitness.predict(candidate, 'cooccurrence'), [1, 0, 1, 0, 0, 1])
        candidate[1:8] = .49
        np.testing.assert_array_equal(fitness.cooccurrence(candidate), [1, 0, 0, 0, 0, 1])
        candidate[0] = .5
        np.testing.assert_array_equal(fitness.cooccurrence(candidate), np.zeros(6))
        candidate[:8] = 0
        candidate[8] = 0
        np.testing.assert_array_equal(fitness.cooccurrence(candidate), np.zeros(6))

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

    def test_all_algorithms_all_methods_and_reproducibility(self):
        rng = np.random.default_rng(21)
        X = rng.random((32, 8)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 1).astype(np.int8)
        fitness = Fitness(X)
        for optimizer_type, config_type, logger_type in ALGORITHMS:
            for method in ('weighted_mean', 'noisy_or', 'cooccurrence'):
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
                        expected = 42 if optimizer_type is IBGWO4 else 24
                        if optimizer_type is MOHHO:
                            self.assertTrue(24 <= optimizer.evaluations <= 42)
                        else:
                            self.assertEqual(optimizer.evaluations, expected)
                        last = logger.history[-1]
                        self.assertEqual(last['evaluations'], optimizer.evaluations)
                        representative = last['representative']
                        self.assertEqual(representative['metrics'], classification_metrics(
                            y, fitness.predict(representative['candidate'], method)))
                        self.assertNotIn('best_recall', last)
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


class ExperimentTests(unittest.TestCase):
    @staticmethod
    def records():
        return [
            dict.fromkeys(FEATURES, .1 + .8 * label) | {
                'label': label, 'subject_id': f'person-{subject}',
                'source_video': f'videos/{subject}/{label}.mp4',
            }
            for subject in range(10) for label in (0, 1) for _ in range(2)
        ]

    def test_subject_splits_are_disjoint_complete_and_reproducible(self):
        records = self.records()
        y = np.array([r['label'] for r in records])
        splits, groups = split_dataset(records, y)
        repeated, _ = split_dataset(records, y)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(y)))
        group_sets = [set(groups[idx]) for idx in splits.values()]
        self.assertEqual([len(g) for g in group_sets], [6, 2, 2])
        for i in range(3):
            for j in range(i):
                self.assertFalse(group_sets[i] & group_sets[j])
        for name, indices in splits.items():
            np.testing.assert_array_equal(indices, repeated[name])
        # UTA subject identity does not depend on the fold/part prefix.
        uta = [dict(r, source_video=f"Fold{1 + i % 2}_part1/{i // 4:02d}/0.mp4")
               for i, r in enumerate(records)]
        for row in uta:
            del row['subject_id']
        _, uta_groups = split_dataset(uta, y)
        self.assertEqual(len(set(uta_groups)), 10)

    def test_missing_ambiguous_or_unusable_groups_fail(self):
        records = self.records()
        y = np.array([r['label'] for r in records])
        for bad in ({'source_video': None}, {'subject_id': ''}, {'subject_id': True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                split_dataset([records[0] | bad] + records[1:], y)
        with self.assertRaisesRegex(ValueError, 'subject_id'):
            split_dataset([{k: v for k, v in row.items() if k != 'subject_id'} for row in records], y)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            split_dataset([records[0] | {'subject_id': 'other'}] + records[1:], y)
        with self.assertRaisesRegex(ValueError, 'five groups'):
            split_dataset(records[:4], y[:4])
        with self.assertRaisesRegex(ValueError, 'both labels'):
            split_dataset(records, np.zeros(len(y), dtype=np.int8))

    def test_subject_aliases_and_path_aliases_cannot_leak(self):
        records = self.records()
        for i, row in enumerate(records):
            row['subject_id'] = str(i // 4).zfill(2) if i % 2 else i // 4
        y = [r['label'] for r in records]
        _, groups = split_dataset(records, y)
        self.assertEqual(len(set(groups)), 10)
        records[1]['source_video'] = 'videos/0/unused/../0.mp4'
        records[1]['subject_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            split_dataset(records, y)
        records = self.records()
        records[0].update(source_video='Fold1_part1/01/0.mp4', subject_id='02')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            split_dataset(records, y)

    def test_validation_selection_rejects_trivial_high_recall(self):
        X = np.zeros((5, 8), dtype=np.float32)
        X[:, 0] = [0, 0, 1, 1, 1]
        y = np.array([0, 0, 1, 1, 1])
        always_positive = np.zeros(9, dtype=np.float32)
        always_positive[0] = 1
        separating = always_positive.copy()
        separating[8] = .5
        results = score_candidates([always_positive, separating], Fitness(X), y, 'noisy_or')
        selected = max(results, key=selection_key)
        self.assertEqual(selected['candidate'], separating.tolist())
        self.assertEqual(selected['metrics']['balanced_accuracy'], 1)
        results = score_candidates([always_positive], Fitness(X), y, 'noisy_or')
        self.assertEqual(max(results, key=selection_key)['model'], 'constant_1')

    def test_all_entrypoint_pipelines_report_held_out_metrics(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / 'data.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in records))
            for optimizer_type, config_type, logger_type in ALGORITHMS:
                with self.subTest(algorithm=optimizer_type.__name__), patch.object(logger_type, 'save_plots'):
                    report = run_experiment(optimizer_type, logger_type,
                        config_type(population_size=6, generations=4, archive_size=8), path, tmp)
                run_dir = Path(report['output_dir'])
                manifest = json.loads((run_dir / 'split.json').read_text())
                self.assertEqual(len(manifest['sha256']), 64)
                X, y = load_jsonl(path)
                test = manifest['splits']['test']['sample_indices']
                selected = report['selected_model']
                pred = (Fitness(X[test]).predict(selected['candidate'], 'noisy_or')
                        if selected['model'] == 'candidate' else np.full(len(test), selected['constant_label']))
                self.assertEqual(report['test_metrics'], classification_metrics(y[test], pred))
                self.assertTrue((run_dir / 'evaluation.json').is_file())
                validation_rows = [json.loads(s) for s in (run_dir / 'validation_archive.jsonl').read_text().splitlines()]
                self.assertEqual(report['validation_metrics'], max(validation_rows, key=selection_key)['metrics'])
                self.assertTrue(all('test_metrics' not in r for r in validation_rows))

    def test_test_labels_do_not_influence_selection(self):
        records = self.records()
        y = np.array([r['label'] for r in records])
        splits, _ = split_dataset(records, y)
        reports = []
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(RunLogger, 'save_plots'):
            for flip_test in (False, True):
                changed = [dict(row) for row in records]
                if flip_test:
                    for i in splits['test']:
                        changed[i]['label'] = 1 - changed[i]['label']
                path = Path(tmp) / 'data.jsonl'
                path.write_text('\n'.join(json.dumps(row) for row in changed))
                reports.append(run_experiment(CMDPSOFS, RunLogger,
                    CMDPSOFSConfig(population_size=6, generations=4, archive_size=8), path, tmp))
        self.assertEqual(reports[0]['selected_model'], reports[1]['selected_model'])
        self.assertEqual(reports[0]['validation_metrics'], reports[1]['validation_metrics'])
        self.assertNotEqual(reports[0]['test_metrics'], reports[1]['test_metrics'])


if __name__ == '__main__':
    unittest.main()
