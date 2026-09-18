import math
import unittest
from unittest.mock import Mock

import torch

from src.config import TrainConfig
from src.model import NMSFreeDetector
from utils.init_weights import initialize_detection_head
from utils.tb_logger import TrainingLogger


class TrainingRegressions(unittest.TestCase):
    def test_gradient_logging_enabled_and_disabled(self):
        model = torch.nn.Linear(2, 1)
        model(torch.ones(1, 2)).sum().backward()
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                writer = Mock()
                cfg = TrainConfig(log_gradients=enabled, log_interval=1, log_hist_interval=1)
                logger = TrainingLogger(writer, cfg)
                self.assertEqual(logger.log_gradients(model, 1, total_norm=2.0), 2.0)
                self.assertEqual(writer.add_scalar.called, enabled)
                self.assertEqual(writer.add_histogram.called, enabled)

    def assert_head_bias(self, head, image_size=640):
        for scale, stride in zip(head.heads, head.strides):
            expected = math.log(5 / head.nc / (image_size / stride) ** 2)
            for branch in ('o2m', 'o2o'):
                bias = getattr(scale, f'out_cls_{branch}').bias
                torch.testing.assert_close(bias, torch.full_like(bias, expected))
                bias = getattr(scale, f'out_reg_{branch}').bias
                torch.testing.assert_close(bias, torch.ones_like(bias))

    def test_model_and_replacement_head_bias(self):
        model = NMSFreeDetector()
        self.assert_head_bias(model.head)
        trunk = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        model.replace_head(nc=3)
        self.assert_head_bias(model.head)
        for key, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, trunk[key])

    def test_explicit_head_initialization(self):
        head = NMSFreeDetector().head
        dfl = {k: v.clone() for k, v in head.dfl.state_dict().items()}
        initialize_detection_head(head, image_size=320)
        self.assert_head_bias(head, image_size=320)
        for key, value in head.dfl.state_dict().items():
            torch.testing.assert_close(value, dfl[key])


if __name__ == '__main__':
    unittest.main()
