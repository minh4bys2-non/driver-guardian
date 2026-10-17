import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai.LSTM.src.tensorboard import log_predictions
from ai.LSTM.src.training import run_epoch


@pytest.mark.parametrize("labels", [[0, 1, 1], [0, 0, 0]])
def test_prediction_charts(tmp_path, labels):
    with SummaryWriter(str(tmp_path)) as writer:
        log_predictions(writer, "val", np.array(labels), np.array([0.1, 0.8, 0.6]),
                        np.array([0, 1, 1]), 1)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert "val/confusion_matrix" in events.Tags()["images"]
    assert ("val/roc" in events.Tags()["images"]) == (1 in labels)
    assert ("val/precision_recall" in events.Tags()["tensors"]) == (1 in labels)
    assert events.Scalars("val/support_alert")[0].value == labels.count(0)


def test_training_events_and_resume_purge(tmp_path):
    class Classifier(nn.Linear):
        def forward(self, x, seq_lens=None):
            return super().forward(x)

    model = Classifier(2, 2)
    samples = [(torch.tensor([1., -1.]), i % 2, 1, "clip") for i in range(3)]
    loader = DataLoader(samples, batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with SummaryWriter(str(tmp_path)) as writer:
        metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                            optimizer, scaler, accumulation=2, writer=writer, epoch=1, split="train")
        writer.add_scalar("train/loss", 99, 11)
    with SummaryWriter(str(tmp_path), purge_step=11) as writer:
        writer.add_scalar("train/loss", 0.25, 11)
    events = EventAccumulator(str(tmp_path)).Reload()
    losses = events.Scalars("train/loss")
    assert [event.step for event in losses] == [5, 10, 11]
    assert losses[-2].value == pytest.approx(metrics["loss"])
    assert losses[-1].value == 0.25
    assert "gradients/weight" in events.Tags()["histograms"]
    assert events.Scalars("train/grad_norm_before_clip")[0].value >= 0


@pytest.mark.parametrize("batches", [1, 3, 10, 11, 23])
def test_training_logs_ten_times_per_epoch(tmp_path, caplog, monkeypatch, batches):
    from ai.LSTM.src.training import logger

    class Classifier(nn.Linear):
        def forward(self, x, seq_lens=None):
            return super().forward(x)

    model = Classifier(2, 2)
    loader = DataLoader([(torch.tensor([1., -1.]), i % 2, 1, "clip") for i in range(batches)])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    monkeypatch.setattr(logger, "propagate", True)
    with caplog.at_level("INFO", logger=logger.name), SummaryWriter(str(tmp_path)) as writer:
        for epoch in (1, 2):
            metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                                optimizer, scaler, accumulation=4, writer=writer, epoch=epoch, split="train")
            events = EventAccumulator(str(tmp_path)).Reload()
            losses = events.Scalars("train/loss")
            assert len(losses) == epoch * min(10, batches)
            assert losses[-1].step == epoch * 10
            assert losses[-1].value == pytest.approx(metrics["loss"])
            assert len({event.step for event in losses}) == len(losses)
    assert sum("batch " in record.message for record in caplog.records) == 2 * min(10, batches)
