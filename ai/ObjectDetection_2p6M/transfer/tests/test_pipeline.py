import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from transfer.config_lmk import FaceLmkConfig, FineTuneConfig, TrainConfig, TrainingStageConfig
from transfer.loss_lmk import FaceLandmarkDetectionLoss, wing_loss


class WingLossTests(unittest.TestCase):
    def test_values_continuity_and_gradients(self):
        error = torch.tensor([0., 0.02, -0.02, 0.1, 0.2], requires_grad=True)
        loss = wing_loss(error, 0.1, 0.02)
        boundary = 0.1 * math.log(6)
        torch.testing.assert_close(loss, torch.tensor([
            0., 0.1 * math.log(2), 0.1 * math.log(2), boundary, 0.1 + boundary,
        ]))
        near = wing_loss(torch.tensor([0.1 - 1e-7, 0.1 + 1e-7]), 0.1, 0.02)
        self.assertLess(abs((near[0] - near[1]).item()), 1e-6)
        loss.sum().backward()
        torch.testing.assert_close(error.grad, torch.tensor([0., 2.5, -2.5, 1., 1.]))

    def test_invalid_parameters(self):
        for value in (0., -1., float('inf'), float('nan')):
            for key in ('lmk_wing_w', 'lmk_wing_epsilon'):
                with self.assertRaises(ValueError):
                    FaceLmkConfig(**{key: value})

    def test_mask_weights_and_face_normalization(self):
        cfg = FaceLmkConfig(num_landmarks=2, eye_landmark_indices=(0,),
                            mouth_landmark_indices=(), nose_tip_landmark_indices=())
        criterion = FaceLandmarkDetectionLoss(cfg)
        raw = torch.tensor([[[0.02], [0.02], [0.2], [0.2]]], requires_grad=True)
        arguments = (
            torch.zeros(1, 2), torch.ones(1, 1), torch.tensor([[[0., 0., 1., 1.]]]),
            torch.ones(1, 1, 1), torch.zeros(1, 1, 2, 2), torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool),
        )
        loss, count = criterion._landmark_loss(raw, *arguments)
        expected = (3 * 0.1 * math.log(2) + 0.1 + 0.1 * math.log(6)) / 4
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(count.item(), 1)
        scaled = list(arguments)
        scaled[1] = arguments[1] * 10
        scaled[2] = arguments[2] * 10
        torch.testing.assert_close(loss, criterion._landmark_loss(raw, *scaled)[0])
        masked = list(arguments)
        masked[5] = torch.zeros(1, 1, dtype=torch.bool)
        zero, count = criterion._landmark_loss(raw, *masked)
        self.assertEqual(zero.item(), 0.)
        self.assertEqual(count.item(), 0)
        zero.backward()
        torch.testing.assert_close(raw.grad, torch.zeros_like(raw))


class PipelineTests(unittest.TestCase):
    def test_small_faces_and_cleanup_after_invalid_schema(self):
        from transfer.dataloader_lmk import FaceLandmarkDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            Image.new('RGB', (640, 640)).save(root / 'images/face.png')
            record = dict(file_name='face.png', image_width=640, image_height=640, faces=[{
                'bounding_box_normalized': dict(xmin=0.1, ymin=0.1, xmax=0.225, ymax=0.225),
                'landmarks_normalized': [dict(x=0.15, y=0.15)] * 478,
            }])
            annotation = root / 'annotations.jsonl'
            annotation.write_text(json.dumps(record) + '\n')
            cfg = FineTuneConfig(index_cache_dir=str(root / 'cache'))
            dataset_cfg = cfg.dataset_config(str(root), train=False)
            dataset = FaceLandmarkDataset(dataset_cfg)
            try:
                sample = dataset[0]
                self.assertEqual(dataset.validation_stats['faces'], 1)
                self.assertEqual(sample['image'].shape, (3, 224, 224))
                torch.testing.assert_close(sample['boxes'][0, 2:] - sample['boxes'][0, :2],
                                           torch.tensor([28., 28.]))
            finally:
                dataset.close()

            record['faces'][0]['bounding_box_normalized']['xmax'] = 0.11
            annotation.write_text(json.dumps(record) + '\n')
            failed = FaceLandmarkDataset.__new__(FaceLandmarkDataset)
            with self.assertRaisesRegex(ValueError, 'bbox quá nhỏ'):
                failed.__init__(dataset_cfg)
            failed.close()

    def test_finetune_has_only_unfrozen_stage(self):
        cfg = FineTuneConfig()
        self.assertEqual(cfg.epochs, 50)
        self.assertEqual(cfg.stage_for_epoch(0), ('finetune', cfg.stage2, 0))
        self.assertEqual(cfg.stage_for_epoch(49), ('finetune', cfg.stage2, 49))
        with self.assertRaisesRegex(ValueError, 'một giai đoạn'):
            FineTuneConfig(stage1=TrainingStageConfig(1, 1e-4, 0., 0., 0.2))

    def test_validation_restores_training_and_saves_batch_position(self):
        from unittest.mock import Mock
        from transfer.train_lmk import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Sequential(torch.nn.BatchNorm1d(2))
        trainer.val_loader = [None]
        trainer.best_val, trainer.global_step = float('inf'), 20000
        trainer.optimizer, trainer.scaler, trainer.ema = None, None, None
        trainer.ckpt = Mock()

        def validate(epoch):
            trainer.model.eval()
            return 1.0

        trainer._validate = validate
        trainer._validate_and_save(0, 20000)
        self.assertTrue(trainer.model.training)
        self.assertTrue(trainer.model[0].training)
        self.assertEqual(trainer.ckpt.save.call_args.args[-4:], (0, 20000, 1.0, 20000))

    def test_train_resume_finetune_and_inference(self):
        import copy
        from types import SimpleNamespace
        from transfer.checkpoint import build_metadata, read_checkpoint
        from transfer.finetune import run_finetuning
        from transfer.inference import FaceLandmarkInferencer
        from transfer.train_lmk import Trainer
        from src.model import NMSFreeDetector
        from utils.artifacts import build_metadata as detection_metadata, write_metadata

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            Image.new('RGB', (480, 480), (120, 100, 90)).save(root / 'images/face.png')
            record = dict(file_name='face.png', image_width=480, image_height=480, faces=[{
                'bounding_box_normalized': dict(xmin=0.1, ymin=0.1, xmax=0.9, ymax=0.9),
                'landmarks_normalized': [dict(x=0.5, y=0.5)] * 478,
            }])
            annotation = root / 'merged_faces.jsonl'
            annotation.write_text((json.dumps(record) + '\n') * 2)
            cfg = TrainConfig(
                face=FaceLmkConfig(head_min_channels=8, lmk_hidden_max_channels=16),
                trunk_backbone_w=(8, 16, 32, 64, 128), trunk_backbone_n=(1, 1, 1, 1), trunk_neck_n=1,
                train_root_dir=str(root), val_root_dir=str(root), jsonl_name='merged_faces.jsonl',
                batch_size=1, num_workers=0,
                persistent_workers=False, device='cpu', amp=False, log_stdout=False,
                index_cache_dir=str(root / 'cache'), ckpt_dir=str(root / 'checkpoints'),
                log_dir=str(root / 'logs'), tb_log_dir=str(root / 'runs'),
                train_horizontal_flip_mode='off', train_geometric_probability=0.5,
                stage1=TrainingStageConfig(1, 1e-3, 0., 0., 0.2),
                stage2=TrainingStageConfig(1, 3e-4, 3e-5, 0., 0.01),
                save_ckpt_interval_steps=1, ckpt_keep_last=10, val_interval_steps=1,
            )
            architecture = dict(nc=1, reg_max=16, backbone_w=cfg.trunk_backbone_w,
                                backbone_n=cfg.trunk_backbone_n, neck_n=1, strides=(8, 16, 32))
            source = NMSFreeDetector(**architecture)
            metadata = detection_metadata(SimpleNamespace(**architecture), [{'id': 0, 'name': 'face'}])
            write_metadata(root / 'source', metadata)
            cfg.trunk_ckpt = str(root / 'source/trunk.pt')
            torch.save({'model': source.state_dict(), 'metadata': metadata}, cfg.trunk_ckpt)
            trainer = Trainer(cfg)
            output = trainer.ckpt.directory
            expected = build_metadata(cfg)
            self.assertEqual(output, root / 'checkpoints' / expected['architecture_sha256'] /
                             expected['categories_sha256'] / cfg.run_name)
            for filename in ('architecture.json', 'categories.json', 'config.json'):
                self.assertTrue((output / filename).is_file())
            self.assertFalse(any(p.requires_grad for p in trainer.model.backbone.parameters()))
            trainer.model.train()
            self.assertTrue(all(not m.training for m in trainer.model.backbone.modules()
                                if isinstance(m, torch.nn.BatchNorm2d)))
            from unittest.mock import patch
            with patch.object(trainer, '_validate', wraps=trainer._validate) as validate:
                trainer.fit()
            # 2 batch/epoch: giữa epoch + cuối epoch, không validate trùng batch cuối.
            self.assertEqual(validate.call_count, 4)
            final = torch.load(output / 'last.pt', weights_only=True)
            self.assertEqual((final['epoch'], final['next_batch'], final['global_step']), (2, 0, 4))
            self.assertTrue(any(trainer.ckpt.log_dir.glob('*.log')))
            self.assertTrue(any(trainer.ckpt.tb_dir.glob('events*')))
            with self.assertRaises(FileExistsError):
                Trainer(cfg)

            # Resume giữa epoch, không cần file pretrained nguồn nữa.
            cfg.resume = str(output / 'ckpt_step00000001.pt')
            cfg.trunk_ckpt = str(root / 'missing.pt')
            resumed = Trainer(cfg)
            self.assertEqual((resumed.start_epoch, resumed.next_batch, resumed.global_step), (0, 1, 1))
            resumed.fit()
            actual = torch.load(output / 'last.pt', weights_only=True)
            for key in final['model']:
                torch.testing.assert_close(final['model'][key], actual['model'][key], rtol=0, atol=0)
            for key in final['ema']:
                torch.testing.assert_close(final['ema'][key], actual['ema'][key], rtol=0, atol=0)
            self.assertEqual(final['ema_updates'], actual['ema_updates'])
            best = str(output / 'best.pt')
            with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                inferencer = FaceLandmarkInferencer(best, device='cpu')
                result = inferencer.model(torch.zeros(1, 3, 224, 224), return_o2m=False)
            self.assertEqual(result['o2o']['lmk'].shape, (1, 1029, 478, 2))
            self.assertTrue(torch.isfinite(result['o2o']['lmk']).all())

            ft = copy.deepcopy(cfg)
            ft.resume, ft.pretrained_ckpt, ft.run_name = '', best, 'landmark_finetune'
            ft.val_root_dir, ft.val_ratio = None, 0.5
            ft.stage1 = TrainingStageConfig(0, 5e-5, 0., 0., 0.05)
            ft.stage2 = TrainingStageConfig(2, 5e-5, 5e-6, 0., 0.05)
            finetuner = Trainer(ft)
            self.assertEqual(finetuner._active_stage_name, 'finetune')
            self.assertTrue(all(p.requires_grad for p in finetuner.model.backbone.parameters()))
            self.assertTrue(all(p.requires_grad for p in finetuner.model.neck.parameters()))
            self.assertEqual(finetuner._set_stage_learning_rates(0, 0), (5e-5, 5e-6))
            finetuner.fit()
            ft_output = output.parent / ft.run_name
            checkpoint = read_checkpoint(ft_output / 'last.pt', ft)
            self.assertEqual(checkpoint['global_step'], 2)
            self.assertEqual(checkpoint['training_plan']['face_loss']['landmark_loss'], 'wing')
            self.assertFalse((ft_output / 'stage1_head_only_final.pt').exists())
            self.assertTrue((ft_output / 'finetune_final.pt').is_file())
            ft.resume = str(ft_output / 'ckpt_step00000001.pt')
            ft.pretrained_ckpt = str(root / 'missing.pt')
            run_finetuning(ft)
            self.assertEqual(read_checkpoint(ft_output / 'last.pt')['global_step'], 2)

            cfg.resume = str(output / 'last.pt')
            changed = copy.deepcopy(cfg)
            changed.face.lmk_wing_w = 0.2
            with self.assertRaisesRegex(ValueError, 'Training plan'):
                Trainer(changed)
            original = annotation.read_text()
            annotation.write_text(original + '\n')
            with self.assertRaisesRegex(ValueError, 'Dataset'):
                Trainer(cfg)
            annotation.write_text(original)
            sidecar = output / 'categories.json'
            original = sidecar.read_text()
            sidecar.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                read_checkpoint(best)
            sidecar.write_text(original)


if __name__ == '__main__':
    unittest.main()
