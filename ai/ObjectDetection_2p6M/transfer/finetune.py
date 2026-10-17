"""Chỉnh FineTuneConfig trong config_lmk.py rồi chạy module này."""

from ai.ObjectDetection_2p6M.transfer.config_lmk import FineTuneConfig
from ai.ObjectDetection_2p6M.transfer.train_lmk import Trainer


def run_finetuning(cfg=None):
    cfg = cfg or FineTuneConfig()
    if cfg.stage1.epochs:
        raise ValueError('Fine-tune yêu cầu stage1.epochs=0 để mở băng toàn bộ ngay từ đầu.')
    if not cfg.resume and not cfg.pretrained_ckpt:
        raise ValueError('Điền pretrained_ckpt hoặc resume trong FineTuneConfig.')
    return Trainer(cfg).fit()


if __name__ == '__main__':
    run_finetuning()
