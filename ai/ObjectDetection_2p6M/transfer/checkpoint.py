"""Metadata SHA-256, bố trí run và checkpoint cho face landmark."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from ai.ObjectDetection_2p6M.utils.artifacts import canonical_json, checkpoint_dir, signature, write_metadata
from ai.ObjectDetection_2p6M.utils.checkpoint import restore_rng, rng_state


def build_metadata(cfg):
    architecture = json.loads(canonical_json(cfg.checkpoint_model_signature()))
    categories = [{'index': 0, 'id': 0, 'name': 'face'}]
    if cfg.face.nc != 1:
        raise ValueError('Dataset landmark hiện chỉ có một class face.')
    return dict(format_version=1, task='face_landmark', architecture=architecture,
                architecture_sha256=signature(architecture), categories=categories,
                categories_sha256=signature(categories))


def read_checkpoint(path, cfg=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('metadata', {})
    if metadata.get('task') != 'face_landmark' or metadata.get('format_version') != 1:
        raise ValueError('Checkpoint thiếu metadata face_landmark; không thể xác minh để nạp.')
    for key in ('architecture', 'categories'):
        if signature(metadata[key]) != metadata[key + '_sha256']:
            raise ValueError(f'SHA-256 {key} không hợp lệ.')
    expected_files = {
        'architecture.json': {'model': metadata['architecture'], 'sha256': metadata['architecture_sha256']},
        'categories.json': metadata['categories'],
    }
    for name, expected in expected_files.items():
        file = Path(path).parent / name
        if not file.is_file() or json.loads(file.read_text()) != expected:
            raise ValueError(f'Thiếu hoặc sai metadata đi kèm: {file}')
    if cfg is not None and metadata != build_metadata(cfg):
        raise ValueError('Kiến trúc hoặc categories không khớp checkpoint.')
    return checkpoint


def config_from_checkpoint(checkpoint):
    from ai.ObjectDetection_2p6M.transfer.config_lmk import FaceLmkConfig, TrainConfig, TrainingStageConfig

    values = dict(checkpoint['cfg'])
    values['face'] = FaceLmkConfig(**values['face'])
    for key in ('stage1', 'stage2'):
        values[key] = TrainingStageConfig(**values[key])
    return TrainConfig(**values)

def resume_config(cfg):
    # Cho phép đổi nơi ghi log, phần cứng đọc dữ liệu và đường dẫn pretrained.
    ignored = {'resume', 'trunk_ckpt', 'pretrained_ckpt', 'require_pretrained_trunk',
               'ckpt_dir', 'log_dir', 'tb_log_dir', 'log_level', 'log_stdout',
               'log_interval', 'log_loss_interval', 'log_gradients', 'run_name',
               'device', 'num_workers', 'prefetch_factor', 'persistent_workers',
               'pin_memory', 'index_cache_dir', 'ckpt_keep_last', 'save_ckpt_interval_steps',
               'val_interval_steps'}
    return {k: v for k, v in asdict(cfg).items()
            if k not in ignored and not k.startswith(('inference_', 'demo_'))}


class CheckpointManager:
    def __init__(self, cfg, data_signature, steps_per_epoch):
        self.cfg = cfg
        self.metadata = build_metadata(cfg)
        self.data_signature = data_signature
        self.steps_per_epoch = steps_per_epoch
        run = SimpleNamespace(ckpt_dir=cfg.ckpt_dir, run_name=cfg.run_name,
                              checkpoint_metadata=self.metadata)
        self.directory = checkpoint_dir(run)
        relative = self.directory.relative_to(Path(cfg.ckpt_dir))
        self.log_dir = Path(cfg.log_dir) / relative
        self.tb_dir = Path(cfg.tb_log_dir) / relative
        if cfg.resume:
            if Path(cfg.resume).resolve().parent != self.directory.resolve():
                raise ValueError('Resume phải giữ cùng architecture/categories/run_name và ckpt_dir.')
        elif self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f'Run đã tồn tại: {self.directory}. Đổi run_name hoặc đặt resume.')

    def prepare(self):
        write_metadata(self.directory, self.metadata)
        config_file = self.directory / 'config.json'
        if not config_file.exists():
            config_file.write_text(json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False) + '\n')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name, model, optimizer, scaler, ema, epoch, global_step, best_val, next_batch=0):
        if build_metadata(self.cfg) != self.metadata:
            raise ValueError('Kiến trúc đã thay đổi sau khi tạo run.')
        checkpoint = {
            'metadata': self.metadata, 'cfg': asdict(self.cfg),
            'training_plan': self.cfg.checkpoint_training_plan(),
            'resume_config': resume_config(self.cfg), 'data_sha256': self.data_signature,
            'steps_per_epoch': self.steps_per_epoch,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'ema': ema.state_dict() if ema else None,
            'ema_updates': ema.updates if ema else 0, 'rng': rng_state(),
            'epoch': epoch, 'next_batch': next_batch, 'global_step': global_step,
            'best_val': best_val,
        }
        path = self.directory / name
        temporary = path.with_suffix('.pt.tmp')
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_periodic(self, model, optimizer, scaler, ema, epoch, global_step, best_val, next_batch=0):
        args = (model, optimizer, scaler, ema, epoch, global_step, best_val, next_batch)
        self.save(f'ckpt_step{global_step:08d}.pt', *args)
        self.save('last.pt', *args)
        files = sorted(self.directory.glob('ckpt_step*.pt'))
        for path in files[:-self.cfg.ckpt_keep_last]:
            path.unlink()

    def load(self, path, model, optimizer, scaler, ema):
        checkpoint = read_checkpoint(path, self.cfg)
        required = ('optimizer', 'scaler', 'rng', 'epoch', 'next_batch', 'global_step',
                    'best_val', 'ema_updates', 'training_plan', 'resume_config')
        if any(key not in checkpoint for key in required):
            raise ValueError('Checkpoint không đủ trạng thái resume.')
        if canonical_json(checkpoint['training_plan']) != canonical_json(self.cfg.checkpoint_training_plan()):
            raise ValueError('Training plan đã thay đổi; hãy fine-tune trong run mới.')
        if canonical_json(checkpoint['resume_config']) != canonical_json(resume_config(self.cfg)):
            raise ValueError('Config dữ liệu/training đã thay đổi; không thể resume.')
        if checkpoint.get('data_sha256') != self.data_signature or checkpoint.get('steps_per_epoch') != self.steps_per_epoch:
            raise ValueError('Dataset/split/số batch đã thay đổi; không thể resume.')
        epoch, next_batch = checkpoint['epoch'], checkpoint['next_batch']
        if not 0 <= epoch <= self.cfg.epochs or not 0 <= next_batch <= self.steps_per_epoch:
            raise ValueError('Vị trí resume không hợp lệ.')
        if epoch == self.cfg.epochs and next_batch:
            raise ValueError('Checkpoint kết thúc training không được có batch đang chờ.')
        if ema is not None and checkpoint.get('ema') is None:
            raise ValueError('Checkpoint thiếu EMA để resume.')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        if ema is not None:
            ema.ema.load_state_dict(checkpoint['ema'], strict=True)
            ema.updates = checkpoint['ema_updates']
        restore_rng(checkpoint['rng'])
        return epoch, next_batch, checkpoint['global_step'], checkpoint['best_val']
