import tempfile
import unittest
from pathlib import Path

import torch

from finetune.finetune_config import FineTuneConfig
from finetune.finetune_engine import get_finetune_model, get_finetune_optimizer, set_finetune_stage
from src.config import TrainConfig
from src.model import NMSFreeDetector
from utils.artifacts import build_metadata
from utils.checkpoint import save_only_model


class FineTuneTests(unittest.TestCase):
    def test_config_inherits_training_defaults(self):
        cfg = FineTuneConfig()
        self.assertIsInstance(cfg, TrainConfig)
        self.assertEqual(cfg.batch_size, TrainConfig().batch_size)
        self.assertLess(cfg.lr0, TrainConfig().lr0)
        self.assertTrue(TrainConfig().train_class_sampling)
        self.assertTrue(cfg.train_class_sampling)
        self.assertNotEqual(cfg.class_sampling_path, TrainConfig().class_sampling_path)

    def test_pretrained_head_and_frozen_batchnorm(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = NMSFreeDetector(nc=2)
            path = Path(tmp) / 'pretrained.pt'
            cfg = FineTuneConfig(nc=2, tfl_pretrained_pth=str(path))
            cfg.checkpoint_metadata = build_metadata(cfg, [{"id": 0, "name": "first"}, {"id": 1, "name": "second"}])
            save_only_model(path, source, cfg=cfg)
            model = get_finetune_model(cfg)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, source.state_dict()[key])
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertFalse(model.neck.training)
            self.assertTrue(model.head.training)
            before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
            model(torch.rand(2, 3, 64, 64))
            for key, value in model.backbone.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

            cfg.nc = 3
            cfg.checkpoint_metadata = build_metadata(cfg, [{"id": i, "name": str(i)} for i in range(3)])
            transferred = get_finetune_model(cfg)
            self.assertEqual(transferred.head.nc, 3)
            for name in ('backbone', 'neck'):
                for key, value in getattr(source, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(transferred, name).state_dict()[key], rtol=0, atol=0)
            expected_bias = NMSFreeDetector(nc=3).head.heads[0].out_cls_o2m.bias
            torch.testing.assert_close(transferred.head.heads[0].out_cls_o2m.bias, expected_bias)

            cfg.nc = 2
            cfg.checkpoint_metadata = build_metadata(cfg, [{"id": 0, "name": "different"}, {"id": 1, "name": "second"}])
            remapped = get_finetune_model(cfg)
            self.assertFalse(torch.equal(remapped.head.heads[0].out_cls_o2m.weight,
                                         source.head.heads[0].out_cls_o2m.weight))

            cfg.strides = (4, 8, 16)
            cfg.checkpoint_metadata = build_metadata(cfg, cfg.checkpoint_metadata['categories'])
            with self.assertRaisesRegex(ValueError, "chỉ cho phép đổi nc"):
                get_finetune_model(cfg)

    def test_stages_and_optimizer_membership(self):
        cfg = FineTuneConfig(nc=2, resume='resume.pt')
        model = get_finetune_model(cfg)
        optimizer = get_finetune_optimizer(model, cfg)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters()})
        for epoch in (0, cfg.head_only_epochs - 1, cfg.head_only_epochs):
            set_finetune_stage(model, cfg, epoch)
            model.eval()
            model.train()
            for module in (model.backbone, model.neck):
                self.assertEqual(module.training, epoch >= cfg.head_only_epochs)
                self.assertTrue(all(p.requires_grad == (epoch >= cfg.head_only_epochs)
                                    for p in module.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.head.heads.parameters()))
            weight = model.head.dfl.conv.weight
            self.assertFalse(weight.requires_grad)
            before = weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            logits = torch.randn(1, 4 * cfg.reg_max, 8, requires_grad=True)
            model.head.dfl(logits).sum().backward()
            self.assertIsNotNone(logits.grad)
            self.assertIsNone(weight.grad)
            optimizer.step()
            torch.testing.assert_close(weight, before, rtol=0, atol=0)

    def test_invalid_schedule(self):
        for kwargs in ({'head_only_epochs': 0}, {'head_only_epochs': 30},
                       {'head_only_epochs': 1.5}, {'trunk_lr_factor': 0},
                       {'trunk_lr_factor': 1}, {'lr0': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FineTuneConfig(**kwargs)

    def test_resume_does_not_require_pretrained_file(self):
        model = get_finetune_model(FineTuneConfig(nc=2, resume='resume.pt', tfl_pretrained_pth=''))
        self.assertEqual(model.nc, 2)


if __name__ == '__main__':
    unittest.main()
