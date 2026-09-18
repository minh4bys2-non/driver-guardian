import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import TrainConfig
from src.model import NMSFreeDetector
from utils.artifacts import architecture, build_metadata, checkpoint_dir, signature
from utils.checkpoint import load_only_model, save_only_model
from train.engine import save_periodic_checkpoint


class CheckpointArtifactTests(unittest.TestCase):
    def test_signature_covers_every_model_field(self):
        cfg = TrainConfig()
        original = signature(architecture(cfg))
        changes = {'nc': 79, 'reg_max': 8, 'backbone_w': (16, 32, 64, 128, 128),
                   'backbone_n': (1, 1, 2, 1), 'neck_n': 2, 'strides': (4, 8, 16)}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.assertIn(key, asdict(cfg))
                self.assertNotEqual(signature(architecture(TrainConfig(**{key: value}))), original)
        self.assertEqual(signature(architecture(TrainConfig(lr0=0.1, epochs=2))), original)

    def test_metadata_roundtrip_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(nc=2, ckpt_dir=tmp)
            cfg.checkpoint_metadata = build_metadata(cfg, [{'id': 8, 'name': 'car'}, {'id': 3, 'name': 'person'}])
            model = NMSFreeDetector(nc=2)
            path = checkpoint_dir(cfg) / 'best.pt'
            save_only_model(path, model, cfg=cfg)
            categories_file = path.parent / 'categories.json'
            self.assertEqual(json.loads(categories_file.read_text()), [
                {'index': 0, 'id': 3, 'name': 'person'}, {'index': 1, 'id': 8, 'name': 'car'}])
            restored = NMSFreeDetector(nc=2)
            load_only_model(path, restored, cfg=cfg)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

            categories_file.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_only_model(path, restored, cfg=cfg)
            categories_file.write_text(json.dumps(cfg.checkpoint_metadata['categories']))
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['metadata']['architecture']['strides'] = [4, 8, 16]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_only_model(path, restored, cfg=cfg)

    def test_unsigned_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.pt'
            model = NMSFreeDetector()
            torch.save({'model': model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'thiếu chữ ký'):
                load_only_model(path, model)

    def test_category_count_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'không khớp'):
            build_metadata(TrainConfig(), [{'id': 0, 'name': 'person'}])
        with self.assertRaisesRegex(ValueError, 'trùng id'):
            build_metadata(TrainConfig(nc=2), [{'id': 0, 'name': 'a'}, {'id': 0, 'name': 'b'}])

    def test_checkpoint_retention_preserves_shared_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(nc=1, ckpt_dir=tmp, ckpt_keep_last=1)
            cfg.checkpoint_metadata = build_metadata(cfg, [{'id': 7, 'name': 'person'}])
            model = NMSFreeDetector(nc=1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            save_periodic_checkpoint(cfg, model, optimizer, scheduler, None, 0, 1, 1.)
            timestamps = {p.name: p.stat().st_mtime_ns for p in checkpoint_dir(cfg).glob('*.json')}
            save_periodic_checkpoint(cfg, model, optimizer, scheduler, None, 0, 2, 1.)
            directory = checkpoint_dir(cfg)
            self.assertFalse((directory / 'ckpt_step00000001.pt').exists())
            self.assertFalse((directory / 'ckpt_step00000001').exists())
            self.assertEqual({p.name for p in directory.iterdir()},
                             {'ckpt_step00000002.pt', 'last.pt', 'architecture.json', 'categories.json'})
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in directory.glob('*.json')}, timestamps)
            for name in ('last.pt', 'ckpt_step00000002.pt'):
                load_only_model(directory / name, model, cfg=cfg)
            (directory / 'architecture.json').unlink()
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_only_model(directory / 'last.pt', model, cfg=cfg)


if __name__ == '__main__':
    unittest.main()
