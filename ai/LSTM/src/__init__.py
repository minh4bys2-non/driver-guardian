"""
Source Package cho Driver Guardian AI (ConvGRU Pipeline).
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_LSTM_DIR = _SRC_DIR.parent
_PROJECT_ROOT = _LSTM_DIR.parent.parent
_OD_DIR = _PROJECT_ROOT / "ai" / "ObjectDetection_2p6M"

for _p in [str(_PROJECT_ROOT), str(_OD_DIR), str(_LSTM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .models import (
    SpatialReductionNeck,
    SpatialAttentionPooling,
    TemporalAttentionPooling,
    ConvGRUClassifier,
)
from .loss import DrowsinessLoss, DrowsinessBCELoss, build_loss
from .dataset import (
    RawVideoFramesDataset,
    collate_video_frames,
    build_raw_video_dataloaders,
    ChunkedBackboneNeckExtractor,
)
from .img_preprocess import (
    BaseImageTransform,
    AdaptiveGammaCorrection,
    CLAHETransform,
    BilateralDenoiseTransform,
    ColorBalanceTransform,
    UnsharpMaskTransform,
    MultiScaleRetinexTransform,
    LowLightImagePreprocessor,
)
from .train import (
    Trainer,
    EarlyStopping,
    calculate_metrics,
    seed_everything,
)
from .evaluate import (
    EvalConfig,
    evaluate,
    EvalBenchmarkConfig,
    evaluate_benchmark,
)

__all__ = [
    "SpatialReductionNeck",
    "SpatialAttentionPooling",
    "TemporalAttentionPooling",
    "ConvGRUClassifier",
    "EvalConfig",
    "evaluate",
    "EvalBenchmarkConfig",
    "evaluate_benchmark",
    "DrowsinessLoss",
    "DrowsinessBCELoss",
    "build_loss",
    "RawVideoFramesDataset",
    "collate_video_frames",
    "build_raw_video_dataloaders",
    "ChunkedBackboneNeckExtractor",
    "BaseImageTransform",
    "AdaptiveGammaCorrection",
    "CLAHETransform",
    "BilateralDenoiseTransform",
    "ColorBalanceTransform",
    "UnsharpMaskTransform",
    "MultiScaleRetinexTransform",
    "LowLightImagePreprocessor",
    "Trainer",
    "EarlyStopping",
    "calculate_metrics",
    "seed_everything",
]
