import unittest
from unittest.mock import patch

import torch
from torchvision.ops import box_iou

from evaluation.mAPEvaluation import MetricAccumulator


class ValidationMetricsTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_metrics_device_consistency(self):
        results = []
        for device in ('cpu', 'cuda'):
            metric = MetricAccumulator(nc=2, max_det=2)
            metric.update(
                {'o2o': {'box': torch.tensor([[[0., 0., 10., 10.], [0., 0., 7., 10.]]], device=device),
                         'cls': torch.tensor([[[5., -20.], [10., -20.]]], device=device)}},
                [{'boxes': torch.tensor([[0., 0., 10., 10.]], device=device),
                  'labels': torch.tensor([0], device=device)}],
            )
            results.append(metric.compute())
        self.assertEqual(results[0], results[1])

    def test_iou_threshold_boundary_and_duplicate_predictions(self):
        metric = MetricAccumulator(nc=2)
        metric.update(
            {'o2o': {'box': torch.tensor([[[0., 0., 7., 10.], [0., 0., 7., 10.]]]),
                     'cls': torch.tensor([[[10., -20.], [5., -20.]]])}},
            [{'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([0])}],
        )
        with patch('evaluation.mAPEvaluation.box_iou', wraps=box_iou) as iou:
            scores = metric.compute()
        self.assertEqual(iou.call_count, 1)
        self.assertEqual(scores['map_50'], 1.)
        self.assertEqual(scores['map_50_95'], 0.5)
        self.assertEqual(scores['precision'], 0.5)
        self.assertEqual(scores['recall'], 1.)

    def test_confidence_order_across_images_and_empty_targets(self):
        metric = MetricAccumulator(nc=2, max_det=1)
        metric.update(
            {'o2o': {'box': torch.tensor([[[0., 0., 10., 10.]], [[0., 0., 10., 10.]]]),
                     'cls': torch.tensor([[[5., -20.]], [[10., -20.]]])}},
            [{'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([0])},
             {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}],
        )
        scores = metric.compute()
        self.assertEqual(scores['map_50'], 0.5)
        self.assertEqual(scores['map_50_95'], 0.5)
        self.assertEqual(scores['precision'], 0.5)
        self.assertEqual(scores['recall'], 1.)

    def test_ground_truth_without_predictions(self):
        metric = MetricAccumulator(nc=1)
        metric.update(
            {'o2o': {'box': torch.zeros(1, 2, 4), 'cls': torch.full((1, 2, 1), -20.)}},
            [{'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([0])}],
        )
        scores = metric.compute()
        for key in ('map_50', 'map_50_95', 'precision', 'recall'):
            self.assertEqual(scores[key], 0.)
