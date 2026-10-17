"""Evaluate validation from config_lmk; run from the project root:

Chỉnh các tham số bên dưới rồi chạy: python -m transfer.evaluation

NME = mean 2D landmark distance / GT normalizer (outer eye corners 33/263,
or sqrt(GT bbox area)). Region NME uses the same normalizer and face weighting.
NME/region NME include matched faces with finite predictions only. Misses and
nonfinite predictions count as infinite error in FR and CED. FR = P(NME > t).
AUC = integral_0^t CED(x) dx / t, computed exactly for the empirical CED.
All JSON metrics are fractions, not percentages. False positives are counted
separately; these landmark metrics do not replace detection precision/recall.
"""

import json
import math
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou, nms

from ai.ObjectDetection_2p6M.transfer.checkpoint import build_metadata, config_from_checkpoint, read_checkpoint
from ai.ObjectDetection_2p6M.transfer.config_lmk import (
    FineTuneConfig, TrainConfig, MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE,
    MEDIAPIPE_LEFT_IRIS, MEDIAPIPE_RIGHT_IRIS, MEDIAPIPE_LIPS,
    MEDIAPIPE_NOSE_TIP, MEDIAPIPE_NUM_LANDMARKS,
)
from ai.ObjectDetection_2p6M.transfer.dataloader_lmk import FaceLandmarkDataset, face_landmark_collate
from ai.ObjectDetection_2p6M.transfer.model_lmk import FaceLmkDetector

# Tham số đánh giá: chỉnh trực tiếp tại đây.
WEIGHTS_PATH = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/checkpoints/231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11/da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9/landmark_finetune2/best.pt"
CONFIG = 'finetune'  # 'train' hoặc 'finetune': lấy validation từ config_lmk.
DEVICE = "cuda"  # None: dùng device trong config, fallback CPU nếu không có CUDA.
BATCH_SIZE = None  # None: dùng batch_size trong config_lmk.
NUM_WORKERS = None  # None: dùng num_workers trong config_lmk.
NORMALIZATION = 'interocular'  # 'interocular' hoặc 'bbox'.
THRESHOLD = 0.1  # Ngưỡng FR và cận trên AUC, dạng fraction.
MATCH_IOU = 0.5
OUTPUT_PATH = '/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/runs/evaluation/evaluation_resultsv2.json'

REGIONS = dict(left_eye=MEDIAPIPE_LEFT_EYE, right_eye=MEDIAPIPE_RIGHT_EYE,
               left_iris=MEDIAPIPE_LEFT_IRIS, right_iris=MEDIAPIPE_RIGHT_IRIS,
               lips=MEDIAPIPE_LIPS, nose_tip=MEDIAPIPE_NOSE_TIP)
REGIONS['other'] = tuple(sorted(set(range(MEDIAPIPE_NUM_LANDMARKS))
                               - {i for indices in REGIONS.values() for i in indices}))


def validation_loader(cfg):
    root = cfg.val_root_dir or (cfg.train_root_dir if cfg.val_ratio else None)
    if not root:
        raise ValueError('Cần val_root_dir hoặc val_ratio > 0 trong config_lmk.')
    dataset = FaceLandmarkDataset(cfg.dataset_config(root, train=False))
    if dataset.num_landmarks != MEDIAPIPE_NUM_LANDMARKS:
        raise ValueError('Region-wise evaluation yêu cầu topology MediaPipe 478.')
    if not cfg.val_root_dir:
        count = len(dataset)
        if count < 2:
            raise ValueError('Cần ít nhất 2 record để chia validation.')
        order = torch.randperm(count, generator=torch.Generator().manual_seed(cfg.seed))
        split = max(1, min(count - 1, round(count * cfg.val_ratio)))
        dataset = Subset(dataset, sorted(order[:split].tolist()))
    if not len(dataset):
        raise ValueError('Validation rỗng.')
    kwargs = dict(batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                  num_workers=cfg.num_workers, collate_fn=face_landmark_collate,
                  pin_memory=cfg.pin_memory and torch.cuda.is_available(),
                  persistent_workers=cfg.persistent_workers and cfg.num_workers > 0)
    if cfg.num_workers and cfg.prefetch_factor is not None:
        kwargs['prefetch_factor'] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def load_model(weights, device):
    checkpoint = read_checkpoint(weights)
    cfg = config_from_checkpoint(checkpoint)
    if checkpoint['metadata'] != build_metadata(cfg):
        raise ValueError('Config checkpoint không khớp metadata.')
    if cfg.face.num_landmarks != MEDIAPIPE_NUM_LANDMARKS or cfg.face.nc != 1:
        raise ValueError('Cần checkpoint một class face, 478 landmark.')
    model = FaceLmkDetector(cfg)
    source = 'ema' if checkpoint.get('ema') is not None else 'model'
    model.load_state_dict(checkpoint[source], strict=True)
    return model.to(device).eval(), source


def select_predictions(prediction, cfg):
    scores = prediction['cls'].sigmoid().squeeze(-1)
    boxes, points = prediction['box'], prediction['lmk']
    keep = (scores > cfg.inference_conf_threshold) & torch.isfinite(scores)
    keep &= torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    indices = keep.nonzero().flatten()
    indices = indices[scores[indices].argsort(descending=True)[:cfg.inference_max_det]]
    if cfg.inference_iou_threshold > 0 and len(indices):
        indices = indices[nms(boxes[indices], scores[indices], cfg.inference_iou_threshold)]
    return boxes[indices].cpu(), points[indices].cpu()


def match_faces(boxes, gt_boxes, threshold):
    """Greedy confidence-ordered, one-to-one IoU matching (boxes already sorted)."""
    matches = torch.full((len(gt_boxes),), -1, dtype=torch.long)
    if not len(boxes) or not len(gt_boxes):
        return matches
    overlaps = box_iou(boxes, gt_boxes)
    for index, row in enumerate(overlaps):
        row = row.clone()
        row[matches >= 0] = -1
        value, target = row.max(0)
        if value >= threshold:
            matches[target] = index
    return matches


def summarize(errors, threshold):
    errors = torch.as_tensor(errors, dtype=torch.float64)
    if not errors.numel():
        raise ValueError('Không có face hợp lệ để đánh giá.')
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Ngưỡng lỗi phải hữu hạn và > 0.')
    if torch.isnan(errors).any() or (errors < 0).any():
        raise ValueError('NME phải không âm; dùng +inf cho prediction bị lỗi.')
    finite = errors[torch.isfinite(errors)]
    xs = torch.linspace(0, threshold, 501, dtype=torch.float64)
    ced = torch.searchsorted(errors.sort().values, xs, right=True) / len(errors)
    return dict(nme=finite.mean().item() if len(finite) else None,
                failure_rate=(errors > threshold).double().mean().item(),
                auc_ced=(1 - errors / threshold).clamp(min=0).mean().item(),
                ced=dict(thresholds=xs.tolist(), fractions=ced.tolist()))


@torch.inference_mode()
def evaluate(model, loader, cfg, device, normalization='interocular',
             threshold=0.1, match_iou=0.5):
    errors, region_errors = [], {name: [] for name in REGIONS}
    counts = dict(images=0, gt_faces=0, evaluated_faces=0, matched_faces=0,
                  missed_faces=0, invalid_predictions=0, excluded_gt_faces=0,
                  unmatched_predictions=0)
    model.eval()
    for batch_index, batch in enumerate(loader):
        output = model(batch['image'].to(device), return_o2m=False)['o2o']
        for index, target in enumerate(batch['targets']):
            boxes, points = select_predictions({key: output[key][index]
                                                for key in ('cls', 'box', 'lmk')}, cfg)
            gt_boxes, gt = target['boxes'], target['landmarks']
            if normalization == 'interocular':
                scales = torch.linalg.vector_norm(gt[:, 33] - gt[:, 263], dim=-1)
            elif normalization == 'bbox':
                scales = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0).prod(-1).sqrt()
            else:
                raise ValueError('normalization phải là interocular hoặc bbox.')
            valid = target['landmarks_valid'] & torch.isfinite(gt).all(dim=(1, 2))
            valid &= torch.isfinite(scales) & (scales > cfg.geometry_eps)
            matches = match_faces(boxes, gt_boxes, match_iou)
            counts['images'] += 1
            counts['gt_faces'] += len(gt)
            counts['excluded_gt_faces'] += int((~valid).sum())
            counts['unmatched_predictions'] += len(boxes) - int((matches >= 0).sum())
            for face in valid.nonzero().flatten().tolist():
                counts['evaluated_faces'] += 1
                prediction = int(matches[face])
                if prediction < 0:
                    counts['missed_faces'] += 1
                    errors.append(float('inf'))
                    continue
                counts['matched_faces'] += 1
                distances = torch.linalg.vector_norm(points[prediction] - gt[face], dim=-1) / scales[face]
                if not torch.isfinite(distances).all():
                    counts['invalid_predictions'] += 1
                    errors.append(float('inf'))
                    continue
                errors.append(distances.mean().item())
                for name, indices in REGIONS.items():
                    region_errors[name].append(distances[list(indices)].mean().item())
        if (batch_index + 1) % 100 == 0:
            print(f'Evaluated {counts["images"]} images / {counts["evaluated_faces"]} faces', flush=True)
    result = summarize(errors, threshold)
    result['region_wise_nme'] = {name: sum(values) / len(values) if values else None
                                 for name, values in region_errors.items()}
    result['counts'] = counts
    result['nme_faces'] = counts['matched_faces'] - counts['invalid_predictions']
    return result


def main():
    if not WEIGHTS_PATH:
        raise ValueError('Điền WEIGHTS_PATH ở đầu evaluation.py trước khi chạy.')
    if CONFIG not in ('train', 'finetune'):
        raise ValueError("CONFIG phải là 'train' hoặc 'finetune'.")
    if NORMALIZATION not in ('interocular', 'bbox'):
        raise ValueError("NORMALIZATION phải là 'interocular' hoặc 'bbox'.")
    if not math.isfinite(THRESHOLD) or THRESHOLD <= 0 or not 0 < MATCH_IOU <= 1:
        raise ValueError('THRESHOLD phải hữu hạn > 0; MATCH_IOU phải thuộc (0, 1].')
    cfg = (TrainConfig if CONFIG == 'train' else FineTuneConfig)()
    if BATCH_SIZE is not None:
        cfg.batch_size = BATCH_SIZE
    if NUM_WORKERS is not None:
        cfg.num_workers = NUM_WORKERS
    device = torch.device(DEVICE or (cfg.device if torch.cuda.is_available() else 'cpu'))
    weights, output = Path(WEIGHTS_PATH), Path(OUTPUT_PATH)
    model, source = load_model(weights, device)
    if model.cfg.image_size != cfg.image_size:
        raise ValueError('Kích thước ảnh validation không khớp checkpoint.')
    loader = validation_loader(cfg)
    result = evaluate(model, loader, cfg, device, NORMALIZATION, THRESHOLD, MATCH_IOU)
    result['settings'] = dict(weights=str(weights.resolve()), weight_source=source,
                              config=CONFIG, val_root=cfg.val_root_dir or cfg.train_root_dir,
                              val_ratio=cfg.val_ratio if not cfg.val_root_dir else None,
                              seed=cfg.seed, normalization=NORMALIZATION,
                              threshold=THRESHOLD, match_iou=MATCH_IOU,
                              confidence=cfg.inference_conf_threshold,
                              nms_iou=cfg.inference_iou_threshold, max_det=cfg.inference_max_det,
                              units='fraction', nme_population='finite matched faces',
                              fr_auc_population='all valid GT faces, including misses')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    for name in ('nme', 'failure_rate', 'auc_ced'):
        value = result[name]
        print(f'{name}: {100 * value:.4f}%' if value is not None else f'{name}: N/A')
    for name, value in result['region_wise_nme'].items():
        print(f'  {name}: {100 * value:.4f}%' if value is not None else f'  {name}: N/A')
    print(f'Counts: {result["counts"]}\nSaved: {output}')


if __name__ == '__main__':
    main()
