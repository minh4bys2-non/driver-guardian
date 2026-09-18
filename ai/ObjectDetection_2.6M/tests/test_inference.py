import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from inference.infer import NMSFreeInference, draw_detections
from src.config import TrainConfig
from src.model import NMSFreeDetector
from train.loss import DetectionLoss
from utils.artifacts import build_metadata
from utils.checkpoint import save_only_model
from train.ema import ModelEMA


class InferenceTests(unittest.TestCase):
    def test_validation_branches_and_inference_skip(self):
        model = NMSFreeDetector(nc=2, img_size=64)
        images = torch.rand(2, 3, 64, 64)
        model.eval()
        bn = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
        with torch.no_grad():
            both = model(images)
            branch = model.head.heads[0].out_cls_o2m
            with patch.object(branch, "forward", wraps=branch.forward) as forward:
                single = model(images, o2o_only=True)
                forward.assert_not_called()
            self.assertNotIn("o2m", single)
            for key in both["o2o"]:
                torch.testing.assert_close(both["o2o"][key], single["o2o"][key])
            targets = [{"boxes": torch.tensor([[8., 8., 40., 40.]]), "labels": torch.tensor([1])}] * 2
            loss, items = DetectionLoss(nc=2)(both, targets)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(items["loss_o2m"], 0)
            self.assertGreater(items["loss_o2o"], 0)
        for key, value in bn.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        model.train()
        preds = model(images, o2o_only=True)
        loss, _ = DetectionLoss(nc=2)(preds, targets)
        loss.backward()
        for name in ("out_cls_o2m", "out_cls_o2o"):
            self.assertGreater(getattr(model.head.heads[0], name).weight.grad.abs().sum(), 0)

    def test_checkpoint_batch_coordinates_and_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(nc=2, img_size=64)
            cfg.checkpoint_metadata = build_metadata(cfg, [{"id": 3, "name": "a"}, {"id": 9, "name": "b"}])
            model = NMSFreeDetector(nc=2, img_size=64)
            ema = ModelEMA(model)
            with torch.no_grad():
                next(ema.ema.parameters()).add_(1)
            path = Path(tmp) / "model.pt"
            save_only_model(path, model, ema, cfg=cfg)
            detector = NMSFreeInference(path, device="cpu", img_size=64, score_thres=0.5, max_det=2)
            torch.testing.assert_close(next(detector.model.parameters()), next(ema.ema.parameters()))
            image = np.zeros((20, 40, 3), dtype=np.uint8)
            actual = detector([image, image])
            self.assertEqual(len(actual), 2)
            self.assertEqual(detector([]), [])
            preds = {"o2o": {"cls": torch.tensor([[[4., 3.]]]),
                             "box": torch.tensor([[[-10., 0., 100., 80.]]])}}
            with patch.object(detector.model, "forward", return_value=preds) as forward:
                result = detector(image)
                self.assertTrue(forward.call_args.kwargs["o2o_only"])
            torch.testing.assert_close(result["boxes_xyxy_orig"], torch.tensor([[0., 0., 40., 20.]] * 2))
            self.assertEqual(result["category_ids"].tolist(), [3, 9])
            self.assertEqual(draw_detections(result, detector.class_names).shape, image.shape)
            self.assertFalse(image.any())
            (path.parent / "categories.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "metadata"):
                NMSFreeInference(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
