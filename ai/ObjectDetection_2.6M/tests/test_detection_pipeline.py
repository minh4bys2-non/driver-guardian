import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from finetune.finetune_config import FineTuneConfig
from finetune.finetune_engine import run_finetuning
from src.config import TrainConfig
from train.engine import run_training
from train.dataloader_ import build_dataloaders
from utils.artifacts import checkpoint_dir, validate_metadata
from src.model import NMSFreeDetector


def create_dataset(root, categories=None):
    categories = categories or [{'id': i, 'name': f'class_{i}'} for i in range(80)]
    for split in ('train', 'val'):
        labels, images = root / 'labels' / split, root / 'images' / split
        labels.mkdir(parents=True)
        images.mkdir(parents=True)
        records = {'categories.jsonl': categories, 'images_info.jsonl': [],
                   'annotations.jsonl': [], f'images_{split}.jsonl': []}
        for i in range(3):
            name = f'{i}.png'
            image = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            assert cv2.imwrite(str(images / name), image)
            records['images_info.jsonl'].append({'id': i, 'file_name': name, 'width': 64, 'height': 64})
            records['annotations.jsonl'].append({'image_id': i, 'category_id': categories[i % len(categories)]['id'],
                                                  'bbox': [8., 8., 40., 40.]})
            records[f'images_{split}.jsonl'].append({'image_name': name, 'path': name})
        for name, values in records.items():
            (labels / name).write_text(''.join(json.dumps(value) + '\n' for value in values))


class DetectionPipelineTests(unittest.TestCase):
    def exercise_pipeline(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_dataset(root)
            sampling = root / "class_sampling.jsonl"
            sampling.write_text('{"id": 1, "probability": 200}\n')
            settings = dict(labels_root=str(root / 'labels'), images_root_dir=str(root / 'images'),
                            index_cache_dir=str(root / 'cache'), ckpt_dir=str(root / 'checkpoints'),
                            log_dir=str(root / 'logs'), tb_log_dir=str(root / 'tb'),
                            img_size=64, batch_size=2, num_workers=2, epochs=6 if device == 'cuda' else 1,
                            device=device, ckpt_keep_last=0,
                            class_sampling_path=str(sampling), val_interval_steps=1,
                            save_ckpt_interval_steps=1, log_interval=1, log_hist_interval=0)
            cfg = TrainConfig(**settings)
            train, val, _, _ = build_dataloaders(cfg)
            self.assertEqual(train.dataset.image_ids.count(1), 2)
            self.assertEqual((len(train.dataset), len(val.dataset)), (4, 3))
            self.assertTrue(math.isfinite(run_training(cfg)))
            source = checkpoint_dir(cfg) / 'last.pt'
            target_root = root / 'new_dataset'
            categories = [{'id': 101, 'name': 'bottle'}, {'id': 205, 'name': 'crate'}, {'id': 307, 'name': 'screw'}]
            create_dataset(target_root, categories)
            target_sampling = target_root / "class_sampling.jsonl"
            target_sampling.write_text('{"id": 205, "probability": 200}\n')
            fine_settings = settings | dict(labels_root=str(target_root / 'labels'),
                                            images_root_dir=str(target_root / 'images'),
                                            index_cache_dir=str(target_root / 'cache'), nc=3,
                                            class_sampling_path=str(target_sampling),
                                            epochs=14 if device == 'cuda' else 2,
                                            head_only_epochs=10 if device == 'cuda' else 1)
            fine = FineTuneConfig(**fine_settings, tfl_pretrained_pth=str(source))
            train, val, _, _ = build_dataloaders(fine)
            self.assertEqual(train.dataset.image_ids.count(1), 2)
            self.assertEqual((len(train.dataset), len(val.dataset)), (4, 3))
            self.assertTrue(math.isfinite(run_finetuning(fine)))
            final = torch.load(checkpoint_dir(fine) / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual((final['epoch'], final['next_batch'], final['global_step']), (fine.epochs, 0, fine.epochs * 2))
            self.assertEqual(final['metadata'], fine.checkpoint_metadata)
            self.assertNotEqual(final['metadata']['architecture_sha256'], cfg.checkpoint_metadata['architecture_sha256'])
            self.assertNotEqual(final['metadata']['categories_sha256'], cfg.checkpoint_metadata['categories_sha256'])
            self.assertEqual([c['id'] for c in final['metadata']['categories']], [101, 205, 307])
            self.assertEqual(final['model']['head.heads.0.out_cls_o2m.weight'].shape[0], 3)
            pretrained = torch.load(source, map_location='cpu', weights_only=True)
            self.assertGreaterEqual(pretrained['ema_updates'], 2)
            self.assertGreaterEqual(final['ema_updates'], 2)
            self.assertIn('step', next(iter(final['optimizer']['state'].values())))
            first_step = checkpoint_dir(fine) / 'ckpt_step00000001.pt'
            initial = torch.load(first_step, map_location='cpu', weights_only=True)
            self.assertFalse(torch.equal(final['model']['head.heads.0.out_cls_o2m.weight'],
                                         initial['model']['head.heads.0.out_cls_o2m.weight']))
            boundary = fine.head_only_epochs * 2
            frozen = torch.load(checkpoint_dir(fine) / f'ckpt_step{boundary:08d}.pt', map_location='cpu', weights_only=True)
            self.assertFalse(torch.equal(frozen['model']['head.heads.0.out_cls_o2m.weight'],
                                         initial['model']['head.heads.0.out_cls_o2m.weight']))
            for prefix in ('backbone.', 'neck.'):
                for key, value in frozen['model'].items():
                    if key.startswith(prefix):
                        torch.testing.assert_close(value, pretrained['ema'][key], rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(value, frozen['model'][key])
                                    for key, value in final['model'].items() if key.startswith(prefix) and key.endswith('weight')))
            groups = final['optimizer']['param_groups']
            self.assertAlmostEqual(groups[2]['lr'] / groups[0]['lr'], fine.trunk_lr_factor)
            self.assertEqual(len(list((root / 'checkpoints').rglob('architecture.json'))), 2)
            self.assertEqual(len(list((root / 'checkpoints').rglob('categories.json'))), 2)
            for path in (root / 'checkpoints').rglob('*.pt'):
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                expected = fine.checkpoint_metadata if path.parent == checkpoint_dir(fine) else cfg.checkpoint_metadata
                model = NMSFreeDetector(**expected['architecture'])
                validate_metadata(path, checkpoint, model, expected)
            for step in (boundary - 1, boundary, boundary + 1):
                fine.resume = str(checkpoint_dir(fine) / f'ckpt_step{step:08d}.pt')
                self.assertTrue(math.isfinite(run_finetuning(fine)))
                resumed = torch.load(checkpoint_dir(fine) / 'last.pt', map_location='cpu', weights_only=True)
                self.assertEqual(resumed['global_step'], fine.epochs * 2)
                for section in ('model', 'ema'):
                    for key, value in final[section].items():
                        torch.testing.assert_close(value, resumed[section][key], rtol=0, atol=0)
            fine.trunk_lr_factor = 0.2
            with self.assertRaisesRegex(ValueError, 'trunk_lr_factor'):
                run_finetuning(fine)

    def test_training_finetuning_resume_and_artifacts_cpu(self):
        self.exercise_pipeline('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_training_finetuning_resume_and_artifacts_cuda_amp(self):
        self.exercise_pipeline('cuda')


if __name__ == '__main__':
    unittest.main()
