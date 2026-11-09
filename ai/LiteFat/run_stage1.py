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
        checkpoint_dir="./checkpoints/LiteFat_v6_stage1",
        log_dir="./checkpoints/LiteFat_v6_stage1/logs",
        use_visual_context=False,
        static_shape_ratio=0.25,
        drop_single_class_subjects=True,
        eval_window_mode="center",
        time_interval=0.05,
        window_sec=12.0,
        epochs=25,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        seed=42,
        lr=3e-4,
        weight_decay=2e-3,
        grad_clip=1.0,
        use_amp=True,
        save_freq=1,
        early_stop_patience=8,
        ema_decay=0.995,
        pretrained_from=None,
        resume=None,
    )
    trainer = LiteFatTrainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
