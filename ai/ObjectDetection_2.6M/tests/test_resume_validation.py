import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.mAPEvaluation import MetricAccumulator
from src.config import TrainConfig
from train.dataloader_ import DetectionAugmenter, EpochBatchSampler, ObjectDetectionDataset, collate_fn
from train.ema import ModelEMA
from train.engine import run_training, train_one_epoch, validate
from utils.checkpoint import load, save
from utils.seed import set_seed
from utils.artifacts import MODEL_FIELDS, build_metadata, checkpoint_dir


def signed_config(**kwargs):
    kwargs.setdefault("resume", "")
    cfg = TrainConfig(**kwargs)
    cfg.checkpoint_metadata = build_metadata(cfg, [{"id": i, "name": f"class_{i}"} for i in range(cfg.nc)])
    return cfg


class Samples(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, key):
        seed, index = key
        x = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(seed))
        return x, {"boxes": torch.tensor([[0., 0., 2., 2.]]), "labels": torch.tensor([0])}


class AugmentedSamples(ObjectDetectionDataset):
    def __init__(self):
        self.image_ids = list(range(7))
        self.augmenter = DetectionAugmenter(TrainConfig())
        self.max_load_retries = 1
        self.error_log_path = None
        self.split_name = 'train'

    def _load_item(self, index):
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        image, boxes, labels = self.augmenter(image, [[8., 8., 48., 48.]], [0])
        return torch.from_numpy(image.copy()), {
            'boxes': torch.tensor(boxes).reshape(-1, 4), 'labels': torch.tensor(labels)}


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        for key in MODEL_FIELDS:
            setattr(self, key, getattr(TrainConfig(), key))

    def forward(self, images):
        return self.weight * (images.mean() + torch.rand(()) + random.random() + np.random.rand())


class SimpleLoss:
    def __call__(self, preds, targets):
        loss = preds.square()
        return loss, {"loss": loss.item()}


def make_loader(workers=0):
    ds = Samples()
    return DataLoader(ds, batch_sampler=EpochBatchSampler(ds, 2, True, False, 42),
                      collate_fn=collate_fn, num_workers=workers,
                      persistent_workers=workers > 0, timeout=10 if workers else 0,
                      generator=torch.Generator().manual_seed(42))


class ResumeValidationTests(unittest.TestCase):
    def test_detector_training_validation_and_resume(self):
        class TinySamples(Samples):
            def __len__(self):
                return 3

            def __getitem__(self, key):
                seed, _ = key
                image = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(seed))
                return image, {'boxes': torch.tensor([[8., 8., 48., 48.]]), 'labels': torch.tensor([0])}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = signed_config(nc=1, epochs=1, device='cpu', ckpt_dir=tmp, tb_log_dir=tmp + '/tb',
                              val_interval_steps=2, save_ckpt_interval_steps=1,
                              log_interval=1, log_hist_interval=1)
            ds = TinySamples()

            def loaders(_):
                train = DataLoader(ds, batch_sampler=EpochBatchSampler(ds, 2, True, False, cfg.seed),
                                   collate_fn=collate_fn, generator=torch.Generator().manual_seed(cfg.seed))
                val = DataLoader([ds[(i, i)] for i in range(len(ds))], batch_size=2,
                                 collate_fn=collate_fn, generator=torch.Generator().manual_seed(cfg.seed))
                return train, val, [{"id": 0, "name": "class_0"}]

            with patch('train.engine.setup_logging'), patch('train.engine.get_dataloader', side_effect=loaders):
                expected_loss = run_training(cfg)
                expected = torch.load(checkpoint_dir(cfg) / 'last.pt', weights_only=True)
                cfg.resume = str(checkpoint_dir(cfg) / 'ckpt_step00000001.pt')
                actual_loss = run_training(cfg)
                actual = torch.load(checkpoint_dir(cfg) / 'last.pt', weights_only=True)
            self.assertEqual(actual_loss, expected_loss)
            self.assertEqual((actual['epoch'], actual['next_batch'], actual['global_step']), (1, 0, 2))
            for key, value in expected['model'].items():
                torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)

    def test_augmentation_resume_with_different_workers(self):
        ds = AugmentedSamples()
        sampler = EpochBatchSampler(ds, 2, True, False, 42)
        sampler.epoch = 3
        full = list(DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn))
        sampler.start_batch = 2
        resumed = list(DataLoader(AugmentedSamples(), batch_sampler=sampler, collate_fn=collate_fn,
                                  num_workers=2, persistent_workers=True, timeout=10))
        self.assertEqual(len(resumed), len(full) - 2)
        for (images, targets), (expected_images, expected_targets) in zip(resumed, full[2:]):
            torch.testing.assert_close(images, expected_images, rtol=0, atol=0)
            for target, expected in zip(targets, expected_targets):
                for key in target:
                    torch.testing.assert_close(target[key], expected[key], rtol=0, atol=0)

    def test_resume_matches_uninterrupted_with_prefetch(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as tmp:
                cfg = signed_config(ckpt_dir=tmp, save_ckpt_interval_steps=2, ckpt_keep_last=0)
                set_seed(42)
                model = StochasticModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
                ema = ModelEMA(model)
                train_one_epoch(model, SimpleLoss(), make_loader(workers), None,
                                optimizer, scheduler, None, ema, 'cpu', cfg, 0)
                expected = model.weight.detach().clone()
                expected_rng = (random.random(), np.random.rand(), torch.rand(()))

                resumed = StochasticModel()
                opt = torch.optim.AdamW(resumed.parameters(), lr=1.0)
                sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
                resumed_ema = ModelEMA(resumed)
                epoch, batch, step, best = load(checkpoint_dir(cfg) / 'ckpt_step00000002.pt',
                                               resumed, opt, sched, resumed_ema)
                self.assertEqual((epoch, batch, step, resumed_ema.updates), (0, 2, 2, 2))
                result = train_one_epoch(resumed, SimpleLoss(), make_loader(workers), None,
                                         opt, sched, None, resumed_ema, 'cpu', cfg, epoch,
                                         global_step=step, best_map=best, start_batch=batch)
                self.assertEqual(result[1], 4)
                torch.testing.assert_close(resumed.weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(resumed_ema.ema.weight, ema.ema.weight, rtol=0, atol=0)
                self.assertEqual(resumed_ema.updates, ema.updates)
                self.assertEqual(sched.state_dict(), scheduler.state_dict())
                self.assertEqual(random.random(), expected_rng[0])
                self.assertEqual(np.random.rand(), expected_rng[1])
                torch.testing.assert_close(torch.rand(()), expected_rng[2], rtol=0, atol=0)

    def test_scaler_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = StochasticModel()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, 1)
            scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=1)
            scaler.scale(model.weight.square()).backward()
            scaler.step(opt)
            scaler.update()
            path = Path(tmp) / 'state.pt'
            save(path, model, opt, sched, None, 3, 9, 1.0, signed_config(), scaler=scaler, next_batch=2)
            restored = torch.amp.GradScaler('cpu')
            self.assertEqual(load(path, model, scaler=restored), (3, 2, 9, 1.0))
            self.assertEqual(restored.state_dict(), scaler.state_dict())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_scaler_and_rng_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = StochasticModel().cuda()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, 1)
            scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=1)
            scaler.scale(model.weight.square()).backward()
            scaler.step(opt)
            scaler.update()
            path = Path(tmp) / 'cuda.pt'
            save(path, model, opt, sched, None, 0, 1, 1., signed_config(), scaler=scaler, next_batch=1)
            expected = torch.rand(4, device='cuda')
            restored = torch.amp.GradScaler('cuda')
            load(path, model, scaler=restored, map_location='cuda')
            torch.testing.assert_close(torch.rand(4, device='cuda'), expected, rtol=0, atol=0)
            self.assertEqual(restored.state_dict(), scaler.state_dict())

    def test_completed_epoch_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = signed_config(epochs=2, ckpt_dir=tmp, tb_log_dir='', resume='unused', use_ema=False, device='cpu')
            loader = make_loader()
            with patch('train.engine.setup_logging'), \
                 patch('train.engine.get_dataloader', return_value=(loader, None, cfg.checkpoint_metadata["categories"])), \
                 patch('train.engine.get_model', return_value=StochasticModel()), \
                 patch('train.engine.load_checkpoint', return_value=(0, len(loader), 4, 1.0)), \
                 patch('train.engine.train_one_epoch', return_value=(1., 8, 1.0)) as train:
                run_training(cfg)
            self.assertEqual(train.call_count, 1)
            self.assertEqual(train.call_args.args[10], 1)

    def test_validation_averages_and_metrics(self):
        class Detector(torch.nn.Module):
            def forward(self, images):
                return {"o2o": {"cls": torch.full((len(images), 1, 1), 10.),
                                "box": torch.tensor([[[0., 0., 2., 2.]]]).expand(len(images), -1, -1)}}

        criterion = Mock(nc=1, side_effect=[(None, {"loss": 2., "o2m/cls": 4.}),
                                           (None, {"loss": 8., "o2m/cls": 10.})])
        target = {"boxes": torch.tensor([[0., 0., 2., 2.]]), "labels": torch.tensor([0])}
        loader = [(torch.zeros(2, 3, 4, 4), [target, target]),
                  (torch.zeros(1, 3, 4, 4), [target])]
        logger = Mock()
        val_loss, val_scores = validate(Detector(), criterion, loader, 'cpu', logger, 7)
        self.assertEqual(val_loss, 5.)
        logger.log_losses.assert_called_once_with({"loss": 5., "o2m/cls": 7.}, step=7, phase='val')
        scalars = logger.log_scalars.call_args.args[0]
        for key in ('map_50', 'map_50_95', 'precision', 'recall'):
            self.assertAlmostEqual(scalars[f'val/{key}'], 1.)
        with self.assertRaisesRegex(ValueError, 'rỗng'):
            validate(Detector(), criterion, [], 'cpu')

    def test_metrics_match_remaining_ground_truth(self):
        metric = MetricAccumulator(nc=1, iou_thresholds=[0.5])
        targets = [{"boxes": torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.]]),
                    "labels": torch.tensor([0, 0])}]
        preds = {"o2o": {"cls": torch.tensor([[[5.], [10.]]]),
                         "box": torch.tensor([[[0., 0., 10., 10.], [0., 0., 10., 10.]]])}}
        metric.update(preds, targets)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.)
        self.assertEqual(result['recall'], 1.)


if __name__ == '__main__':
    unittest.main()
