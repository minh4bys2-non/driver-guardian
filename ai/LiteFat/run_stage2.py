import sys
sys.path.insert(0, '/home/riftuser/train/driver-guardian/ai/LiteFat')

from config import TrainConfig
from train import LiteFatTrainer

def main():
    cfg = TrainConfig(
        dataset_dir=r"/home/riftuser/train/dataset/data-processed-v3/data_processed",
        cache_dir=r"/home/riftuser/train/dataset/cache-dataset-processed-v3/LiteFat/cache_pt",
        use_cache=True,
        manifest_filename="dataset_merged_split_v6.csv",
        checkpoint_dir="./checkpoints/LiteFat_v6_stage2_visual",
        log_dir="./checkpoints/LiteFat_v6_stage2_visual/logs",
        use_visual_context=True,
        visual_frame_stride=8,
        static_shape_ratio=0.25,
        drop_single_class_subjects=True,
        eval_window_mode="center",
        time_interval=0.05,
        window_sec=12.0,
        epochs=8,
        freeze_backbone_epochs=2,
        batch_size=16,
        num_workers=2,
        pin_memory=True,
        seed=42,
        lr=2e-4,
        backbone_lr=3e-5,
        pretrained_lr_scale=0.25,
        weight_decay=2e-3,
        grad_clip=1.0,
        use_amp=True,
        save_freq=1,
        early_stop_patience=4,
        ema_decay=0.995,
        pretrained_from="./checkpoints/LiteFat_v6_stage1/best.pt",
        resume=None,
    )
    trainer = LiteFatTrainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
