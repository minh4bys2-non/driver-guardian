import math
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai.LSTM.src.dataloader import collate_video_frames
from ai.LSTM.src.evaluation import evaluate
from ai.LSTM.src.training import run_epoch


class Logits(nn.Module):
    def forward(self, frames, seq_lens=None):
        assert not self.training and not torch.is_grad_enabled()
        return frames


def loader(scores, labels):
    samples = []
    for index, (score, label) in enumerate(zip(scores, labels)):
        meta = dict(video_id=str(index), video_path="clip.mp4", subject_id="A", source_video_id="source",
                    start_sec=0, end_sec=10)
        samples.append((torch.tensor([1 - score, score]).log(), torch.tensor(label), torch.tensor(1), meta))
    return DataLoader(samples, batch_size=3, collate_fn=collate_video_frames)


def test_metrics_match_known_confusion_and_sample_weighted_loss():
    data = loader([0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1])
    metrics, rows = evaluate(Logits(), data, torch.device("cpu"))
    epoch = run_epoch(Logits(), data, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    for key in ("accuracy", "precision", "recall", "f1", "specificity"):
        assert metrics[key] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    expected_loss = -(2 * math.log(0.9) + 2 * math.log(0.2)) / 4
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert epoch["loss"] == pytest.approx(expected_loss)
    assert epoch["f1"] == metrics["f1"] and epoch["acc"] == metrics["accuracy"]
    assert [r["prediction"] for r in rows] == [0, 1, 0, 1]
    assert [r["prob_drowsy"] for r in rows] == pytest.approx([0.1, 0.8, 0.2, 0.9])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_are_defined(label):
    metrics, _ = evaluate(Logits(), loader([0.1, 0.9], [label, label]), torch.device("cpu"))
    assert metrics["n_samples"] == 2
    assert metrics["roc_auc"] is None and metrics["average_precision"] is None
    if label == 1:
        assert metrics["specificity"] is None


def test_empty_evaluation_is_rejected():
    with pytest.raises(ValueError, match="Empty evaluation"):
        evaluate(Logits(), loader([], []), torch.device("cpu"))


@pytest.mark.parametrize("logits", [[float("nan"), 0.], [0., float("inf")], [0., 1., 2.]])
def test_invalid_logits_are_rejected_even_for_single_class(logits):
    data = DataLoader([(torch.tensor(logits), torch.tensor(0), torch.tensor(1), {})],
                      collate_fn=collate_video_frames)
    with pytest.raises(ValueError, match="finite binary"):
        evaluate(Logits(), data, torch.device("cpu"))


@pytest.mark.parametrize("overrides", [{}, {"dataset_dir": "moved_data", "batch_size": 1, "num_workers": 0}])
def test_main_uses_checkpoint_config_and_records_runtime_overrides(tmp_path, monkeypatch, overrides):
    from ai.LSTM.src import evaluation

    cfg = dict(dataset_dir="saved_data", image_size=64, window_sec=3, target_fps=2,
               batch_size=2, num_workers=0, amp=False)
    checkpoint = dict(config=cfg, model_config={"hidden_dim": 8}, epoch=7)
    checkpoint_path = tmp_path / "best.pt"
    calls = {}

    def load_model(path, device):
        assert path == checkpoint_path and device == torch.device("cpu")
        return Logits(), checkpoint

    def dataset(dataset_dir, **kwargs):
        calls.update(dataset_dir=dataset_dir, **kwargs)
        return loader([0.1, 0.9], [0, 1]).dataset

    monkeypatch.setattr(evaluation, "load_model", load_model)
    monkeypatch.setattr(evaluation, "RawVideoFramesDataset", dataset)
    monkeypatch.setattr(evaluation, "configure_logging", lambda path: None)
    metrics, rows = evaluation.main(checkpoint_path, device="cpu", **overrides)

    assert calls == dict(dataset_dir=overrides.get("dataset_dir", "saved_data"), split="test",
                         img_size=64, window_sec=3, target_fps=2)
    assert metrics["checkpoint_epoch"] == 7 and metrics["accuracy"] == 1
    output = tmp_path / "evaluation_test"
    saved = json.loads((output / "evaluation_config.json").read_text())
    assert saved["config"] == cfg and saved["model_config"] == checkpoint["model_config"]
    assert saved["batch_size"] == overrides.get("batch_size", 2)
    assert saved["dataset_dir"] == calls["dataset_dir"]
    assert json.loads((output / "metrics.json").read_text()) == metrics
    assert len((output / "predictions.csv").read_text().splitlines()) == len(rows) + 1
