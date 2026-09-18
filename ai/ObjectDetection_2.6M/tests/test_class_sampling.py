import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from finetune.finetune_config import FineTuneConfig
from finetune.finetune_engine import run_finetuning
from src.config import TrainConfig
from test_detection_pipeline import create_dataset
from train.dataloader_ import (
    annotation_target, build_dataloaders,
    load_class_sampling, oversample_image_ids,
)
from train.engine import run_training
from utils.artifacts import checkpoint_dir


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def settings(root, **kwargs):
    return dict(labels_root=str(root / 'labels'), images_root_dir=str(root / 'images'),
                index_cache_dir=str(root / 'cache'), ckpt_dir=str(root / 'checkpoints'),
                log_dir=str(root / 'logs'), tb_log_dir='', img_size=64, batch_size=2,
                num_workers=0, pin_memory=False, rebuild_index=True, device='cpu', amp=False,
                epochs=1, nc=2, backbone_w=(8, 16, 32, 64, 128), reg_max=4,
                val_interval_steps=1, save_ckpt_interval_steps=1, ckpt_keep_last=0) | kwargs


class ClassSamplingTests(unittest.TestCase):
    def test_sampling_file_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sampling.jsonl'
            self.assertEqual(load_class_sampling(path, {101: 0}), {101: 1.0})
            self.assertEqual(load_class_sampling("", {101: 0}), {101: 1.0})
            write_jsonl(path, [{'id': 101, 'probability': 350}, {'id': 205, 'probability': 50}])
            self.assertEqual(load_class_sampling(path, {101: 0, 205: 1}), {101: 3.5, 205: 1.})
            for records in ([], [{'id': 999, 'probability': 200}],
                            [{'id': 101, 'probability': 200}] * 2,
                            [{'id': '101', 'probability': 200}],
                            [{'id': 101, 'probability': -1}],
                            [{'id': 101, 'probability': float('nan')}],
                            [{'id': 101, 'probability': float('inf')}],
                            [{'id': 101, 'probability': True}], [{'id': 101}]):
                with self.subTest(records=records), self.assertRaises(ValueError):
                    write_jsonl(path, records)
                    load_class_sampling(path, {101: 0})
            path.write_text('{invalid}\n')
            with self.assertRaisesRegex(ValueError, 'dòng 1'):
                load_class_sampling(path, {101: 0})

    def test_missing_sampling_file_keeps_pretrain_and_finetune_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root, [{'id': 101, 'name': 'common'}, {'id': 205, 'name': 'rare'}])
            for config in (TrainConfig, FineTuneConfig):
                with self.subTest(config=config.__name__):
                    kwargs = {'epochs': 2, 'head_only_epochs': 1} if config is FineTuneConfig else {}
                    cfg = config(**settings(root, class_sampling_path=str(root / 'missing.jsonl'), **kwargs))
                    train, val, _, _ = build_dataloaders(cfg)
                    self.assertTrue(cfg.train_class_sampling)
                    self.assertEqual(train.dataset.image_ids, [0, 1, 2])
                    self.assertEqual(val.dataset.image_ids, [0, 1, 2])
                    self.assertEqual(sum(len(images) for images, _ in train), 3)

    def test_percentage_format(self):
        probabilities = [103, 140, 100, 118, 112, 109, 181, 100, 100,
                         100, 100, 102, 100, 100, 100, 127, 100, 104]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'class_sampling.jsonl'
            write_jsonl(path, [{'id': i, 'probability': p} for i, p in enumerate(probabilities)])
            weights = load_class_sampling(path, dict(enumerate(range(18))))
            self.assertEqual(weights, {i: p / 100 for i, p in enumerate(probabilities)})
            categories = {i: {i % 18} for i in range(18000)}
            sampled = oversample_image_ids(list(categories), categories, weights)
            counts = Counter(i % 18 for i in sampled)
            for category_id, probability in enumerate(probabilities):
                if probability == 100:
                    self.assertEqual(counts[category_id], 1000)
                else:
                    self.assertLess(abs(counts[category_id] - probability * 10), 60)

    def test_multiclass_uses_max_and_fractional_rounding_is_seeded(self):
        categories = {0: {101, 205}, 1: {101}, 2: set()}
        sampled = oversample_image_ids([0, 1, 2], categories, {101: 2., 205: 4.})
        self.assertEqual(Counter(sampled), {0: 4, 1: 2, 2: 1})
        categories = {i: {101} for i in range(1000)}
        a = oversample_image_ids(list(categories), categories, {101: 1.5}, seed=7)
        b = oversample_image_ids(list(categories), categories, {101: 1.5}, seed=7)
        c = oversample_image_ids(list(categories), categories, {101: 1.5}, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertLess(abs(len(a) - 1500), 60)

    def test_filtering_matches_targets_and_validation_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root, [{'id': 101, 'name': 'common'}, {'id': 205, 'name': 'rare'}])
            path = root / 'weights.jsonl'
            write_jsonl(path, [{'id': 205, 'probability': 300}])
            annotations = [{'image_id': 0, 'category_id': 101, 'bbox': [0, 0, 16, 16]},
                           {'image_id': 1, 'category_id': 205, 'bbox': [0, 0, 16, 16]}] * 2
            for extra in ({'iscrowd': 1}, {'isfake': 1}, {'bbox': [80, 0, 10, 10]},
                          {'bbox': [0, 0, -1, 1]}, {'bbox': [0, 0, float('nan'), 1]},
                          {'category_id': 999}):
                annotations.append({'image_id': 2, 'category_id': 205, 'bbox': [0, 0, 16, 16]} | extra)
            write_jsonl(root / 'labels/train/annotations.jsonl', annotations)
            cfg = TrainConfig(**settings(root, class_sampling_path=str(path)))
            train, val, _, _ = build_dataloaders(cfg)
            self.assertEqual(Counter(train.dataset.image_ids), {0: 1, 1: 3})
            self.assertEqual(val.dataset.image_ids, [0, 1, 2])
            self.assertEqual(train.dataset._load_item(3)[1]['labels'].tolist(), [1, 1])
            cfg.include_images_without_annotations = True
            train, _, _, _ = build_dataloaders(cfg)
            self.assertEqual(Counter(train.dataset.image_ids), {0: 1, 1: 3, 2: 1})
            self.assertEqual(train.dataset._load_item(4)[1]['boxes'].shape, (0, 4))
            cfg.train_class_sampling = False
            train, _, _, _ = build_dataloaders(cfg)
            self.assertEqual(train.dataset.image_ids, [0, 1, 2])
        self.assertEqual(annotation_target({'category_id': 101, 'bbox': [-4, -4, 20, 20]},
                                           {101: 0}, 10, 10, True, True), ([0., 0., 10, 10], 0))

    def test_repeated_augmentation_and_resume_across_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root, [{'id': 101, 'name': 'common'}, {'id': 205, 'name': 'rare'}])
            path = root / 'weights.jsonl'
            write_jsonl(path, [{'id': 205, 'probability': 400}])
            cfg = TrainConfig(**settings(root, class_sampling_path=str(path), shuffle=False,
                                         randomBrightnessContrast=1.0))
            train, _, _, _ = build_dataloaders(cfg)
            train.batch_sampler.epoch = 2
            keys = [key for batch in train.batch_sampler for key in batch]
            repeated = [key for key in keys if train.dataset.image_ids[key[1]] == 1]
            self.assertEqual(len({key[0] for key in repeated}), 4)
            self.assertFalse(torch.equal(train.dataset[repeated[0]][0], train.dataset[repeated[1]][0]))
            full = list(train)
            cfg.num_workers, cfg.persistent_workers, cfg.prefetch_factor = 2, True, 2
            resumed, _, _, _ = build_dataloaders(cfg)
            resumed.batch_sampler.epoch = 2
            resumed.batch_sampler.start_batch = 1
            for actual, expected in zip(list(resumed), full[1:]):
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                for target, reference in zip(actual[1], expected[1]):
                    for key in target:
                        torch.testing.assert_close(target[key], reference[key], rtol=0, atol=0)

    def test_sampled_training_finetuning_and_changed_file_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root, [{'id': 101, 'name': 'common'}, {'id': 205, 'name': 'rare'}])
            path = root / 'sampling.jsonl'
            write_jsonl(path, [{'id': 205, 'probability': 200}])
            cfg = TrainConfig(**settings(root, class_sampling_path=str(path)))
            self.assertTrue(math.isfinite(run_training(cfg)))
            source = checkpoint_dir(cfg) / 'last.pt'
            expected = torch.load(source, weights_only=True)
            cfg.resume = str(checkpoint_dir(cfg) / 'ckpt_step00000001.pt')
            self.assertTrue(math.isfinite(run_training(cfg)))
            actual = torch.load(source, weights_only=True)
            for key in expected['model']:
                torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)

            target = root / 'target'
            create_dataset(target, [{'id': i, 'name': f'class_{i}'} for i in (4, 8, 16)])
            target_path = target / 'sampling.jsonl'
            write_jsonl(target_path, [{'id': 8, 'probability': 300}])
            fine = FineTuneConfig(**settings(target, nc=3, epochs=2, head_only_epochs=1,
                                             class_sampling_path=str(target_path)),
                                  tfl_pretrained_pth=str(source))
            self.assertTrue(math.isfinite(run_finetuning(fine)))
            expected = torch.load(checkpoint_dir(fine) / 'last.pt', weights_only=True)
            self.assertEqual(expected['global_step'], 6)
            fine.resume = str(checkpoint_dir(fine) / 'ckpt_step00000004.pt')
            self.assertTrue(math.isfinite(run_finetuning(fine)))
            actual = torch.load(checkpoint_dir(fine) / 'last.pt', weights_only=True)
            for section in ('model', 'ema'):
                for key in expected[section]:
                    torch.testing.assert_close(actual[section][key], expected[section][key], rtol=0, atol=0)
            write_jsonl(target_path, [{'id': 8, 'probability': 400}])
            with self.assertRaisesRegex(ValueError, 'Sampling đã thay đổi'):
                run_finetuning(fine)


if __name__ == '__main__':
    unittest.main()
