import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai.LSTM.src import training
from ai.LSTM.src.checkpoint import atomic_save, load_checkpoint, load_model
from ai.LSTM.src.config import TrainConfig, load_config
from ai.LSTM.src.dataloader import RawVideoFramesDataset, collate_video_frames
from ai.LSTM.src.model import DrowsyClassifier
from ai.LSTM.src.model_backbone_neck import BackboneNeck


def tiny_config(tmp_path, **kwargs):
    kwargs = {"backbone_mode": "scratch", "backbone_checkpoint": None, **kwargs}
    return TrainConfig(save_dir=str(tmp_path / "run"), device="cpu", amp=False, num_workers=0,
                       epochs=2, warmup_epochs=0, batch_size=2, accumulation=2,
                       backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                       input_dim=8, hidden_dim=8, image_size=64, chunk_size=2, **kwargs)


@pytest.fixture
def fake_loaders(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    samples = [(torch.randint(256, (2, 3, 64, 64), dtype=torch.uint8, generator=generator),
                torch.tensor(i % 2), torch.tensor(2), {}) for i in range(3)]
    def build(*args, **kwargs):
        return tuple(DataLoader(samples, batch_size=2, shuffle=split == 0,
                                collate_fn=collate_video_frames) for split in range(3))
    monkeypatch.setattr(training, "build_raw_video_dataloaders", build)
    return samples


def test_config_roundtrip_and_invalid_modes(tmp_path):
    cfg = tiny_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_config(path) == cfg
    with pytest.raises(ValueError, match="requires"):
        TrainConfig(backbone_mode="frozen", backbone_checkpoint=None)
    with pytest.raises(ValueError, match="scratch"):
        TrainConfig(backbone_mode="scratch", backbone_checkpoint="pretrained.pt")
    with pytest.raises(ValueError, match="backbone_mode"):
        TrainConfig(backbone_mode="frozenv1", backbone_checkpoint=None)
    with pytest.raises(ValueError, match="BatchNorm"):
        TrainConfig(backbone_mode="scratch", backbone_checkpoint=None, gradient_checkpointing=True)
    assert TrainConfig(backbone_mode="frozen", backbone_checkpoint="pretrained.pt",
                       gradient_checkpointing=True).backbone_mode == "frozen"
    with pytest.raises(ValueError, match="accumulation"):
        TrainConfig(accumulation=0)


def test_video_sampling_and_collate(tmp_path):
    folder = tmp_path / "val" / "0_alert"
    folder.mkdir(parents=True)
    writer = cv2.VideoWriter(str(folder / "uldd_subA_clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 24))
    assert writer.isOpened()
    for i in range(18):
        writer.write(np.full((24, 32, 3), i * 10, np.uint8))
    writer.release()
    ds = RawVideoFramesDataset(tmp_path, "val", img_size=32, window_sec=2, target_fps=3)
    first, second = ds[0], ds[0]
    torch.testing.assert_close(first[0], second[0])
    assert first[0].shape == (6, 3, 32, 32) and first[0].dtype == torch.uint8
    assert first[3]["start_sec"] == 0.5
    frames, labels, lengths, _ = collate_video_frames([first, second])
    assert frames.shape == (2, 6, 3, 32, 32)
    assert labels.tolist() == [0, 0] and lengths.tolist() == [6, 6]


@pytest.mark.parametrize("mode", ["scratch", "frozen"])
def test_training_and_self_contained_checkpoint(tmp_path, fake_loaders, mode):
    extra = {}
    if mode == "frozen":
        trunk = BackboneNeck(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1))
        path = tmp_path / "pretrained.pt"
        torch.save({"model": trunk.state_dict(), "cfg": {name: getattr(trunk, name)
                    for name in ("backbone_w", "backbone_n", "neck_n")}}, path)
        extra = dict(backbone_mode=mode, backbone_checkpoint=str(path))
    cfg = tiny_config(tmp_path, **extra)
    model, metrics = training.train(cfg)
    assert metrics["n_samples"] == 3
    model.train()
    if mode == "frozen":
        assert not model.backbone_neck.training
        for key, value in trunk.state_dict().items():
            torch.testing.assert_close(model.backbone_neck.state_dict()[key], value)
        assert all(p.grad is None and not p.requires_grad for p in model.backbone_neck.parameters())
        path.unlink()  # Loading a trained checkpoint must not need the original detector.
    else:
        assert model.backbone_neck.training
        assert any(m.num_batches_tracked > 0 for m in model.backbone_neck.modules() if isinstance(m, nn.BatchNorm2d))
    restored, state = load_model(Path(cfg.save_dir) / "best.pt")
    model.eval()
    x = fake_loaders[0][0][None]
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))
    assert state["model_config"]["hidden_dim"] == 8
    log = (Path(cfg.save_dir) / "train.log").read_text()
    assert "Configuration:" in log and "Epoch 2/2" in log and "Test best_epoch=" in log
    with pytest.raises(FileExistsError):
        training.train(cfg)
    cfg.resume = str(Path(cfg.save_dir) / "last.pt")
    training.train(cfg)
    cfg.lr *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        training.train(cfg)


@pytest.mark.parametrize("mode", ["scratch", "frozen"])
def test_resume_matches_uninterrupted_training(tmp_path, fake_loaders, monkeypatch, mode):
    extra = {}
    if mode == "frozen":
        trunk = BackboneNeck(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1))
        path = tmp_path / "pretrained.pt"
        torch.save({"model": trunk.state_dict(), "cfg": {name: getattr(trunk, name)
                    for name in ("backbone_w", "backbone_n", "neck_n")}}, path)
        extra = dict(backbone_mode=mode, backbone_checkpoint=str(path))
    uninterrupted = tiny_config(tmp_path / "full", **extra)
    training.train(uninterrupted)
    interrupted = tiny_config(tmp_path / "resumed", **extra)
    original = training.atomic_save
    def interrupt_after_epoch_one(state, path):
        original(state, path)
        if Path(path).name == "last.pt" and state["epoch"] == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(training, "atomic_save", interrupt_after_epoch_one)
    with pytest.raises(RuntimeError, match="simulated"):
        training.train(interrupted)
    monkeypatch.setattr(training, "atomic_save", original)
    assert "simulated interruption" in (Path(interrupted.save_dir) / "train.log").read_text()
    if mode == "frozen":
        path.unlink()
    interrupted.resume = str(Path(interrupted.save_dir) / "last.pt")
    training.train(interrupted)
    full = load_checkpoint(Path(uninterrupted.save_dir) / "last.pt")
    resumed = load_checkpoint(interrupted.resume)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert full["best_metrics"] == resumed["best_metrics"]
    assert full["scheduler"] == resumed["scheduler"]


@pytest.mark.parametrize("count,batch_size,accumulation", [(5, 3, 4), (11, 3, 2), (5, 2, 1)])
def test_accumulation_weights_short_batch_by_samples(count, batch_size, accumulation):
    class Classifier(nn.Linear):
        def forward(self, x, seq_lens=None):
            return super().forward(x)
    torch.manual_seed(1)
    model = Classifier(3, 2)
    reference = copy.deepcopy(model)
    samples = [(torch.randn(3), torch.tensor(i % 2), torch.tensor(1), {}) for i in range(count)]
    def step(model, batch_size, accumulation):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loader = DataLoader(samples, batch_size=batch_size, collate_fn=collate_video_frames)
        training.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                           optimizer, scaler, accumulation, max_grad_norm=100)
    step(model, batch_size, accumulation)
    step(reference, batch_size * accumulation, 1)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_atomic_save_keeps_previous_checkpoint_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_save({"epoch": 1}, path)
    def fail(state, file):
        file.write(b"partial checkpoint")
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"epoch": 2}, path)
    assert torch.load(path, weights_only=True)["epoch"] == 1
    assert list(tmp_path.iterdir()) == [path]


def test_evaluation_uses_saved_architecture_and_test_split(tmp_path, fake_loaders, monkeypatch):
    from ai.LSTM.src import evaluation
    cfg = tiny_config(tmp_path)
    training.train(cfg)
    calls = []
    def dataset(root, **kwargs):
        calls.append(kwargs)
        return [(x, y, length, {"video_id": str(i), "video_path": "test.mp4", "subject_id": "A",
                               "source_video_id": "source", "start_sec": 0, "end_sec": 10})
                for i, (x, y, length, _) in enumerate(fake_loaders)]
    monkeypatch.setattr(evaluation, "RawVideoFramesDataset", dataset)
    evaluation.main(Path(cfg.save_dir) / "best.pt", device="cpu")
    assert calls[0]["split"] == "test" and calls[0]["img_size"] == cfg.image_size
    metrics = json.loads((Path(cfg.save_dir) / "evaluation_test/metrics.json").read_text())
    assert metrics["split"] == "test" and metrics["n_samples"] == 3


def test_augmentation_repeats_transform():
    from ai.LSTM.src.augment import get_video_augmenter
    frame = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    first, second = get_video_augmenter().apply_sequence([frame, frame.copy()], seed=42)
    np.testing.assert_array_equal(first, second)


def test_notebook_code_is_valid():
    for path in (Path(__file__).parents[1] / "notebooks").glob("*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")


def test_export_uses_trained_trunk_and_classifier(tmp_path):
    from ai.LSTM.export_vulkan import ExportNMSFreeDetector, ExportConvGRUClassifier
    from ai.LSTM.src.models_convgru import ConvGRUClassifier
    model = DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                             input_dim=8, hidden_dim=8).eval()
    path = tmp_path / "best.pt"
    atomic_save({"format_version": 1, "epoch": 1, "model": model.state_dict(),
                 "model_config": model.model_config, "config": tiny_config(tmp_path).to_dict()}, path)
    classifier = ConvGRUClassifier.from_checkpoint(path)
    images = torch.rand(1, 2, 3, 64, 64)
    with torch.no_grad():
        trunk = ExportNMSFreeDetector(model.backbone_neck.backbone, model.backbone_neck.neck)
        features = tuple(x.reshape(1, 2, *x.shape[1:]) for x in trunk(images.flatten(0, 1)))
        actual = ExportConvGRUClassifier(classifier)(*features)
        expected = model(images).softmax(-1)[:, 1]
    torch.testing.assert_close(actual, expected)


def test_scheduler_warmup_and_final_lr():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = training.cosine_warmup(optimizer, epochs=6, warmup_epochs=2, min_lr_ratio=0.1)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:3] == [0.5, 1.0, 1.0]
    assert rates[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("field,value", [("lr", float("nan")), ("weight_decay", float("inf")),
                                         ("max_grad_norm", float("nan")), ("num_workers", 0.5),
                                         ("chunk_size", 1.5), ("warmup_epochs", 0.5)])
def test_rejects_invalid_numeric_config(field, value):
    with pytest.raises(ValueError):
        TrainConfig(**{field: value})


@pytest.mark.parametrize("payload", [[], {"model": {}}, {"format_version": 2}, {"format_version": 1}])
def test_rejects_invalid_checkpoint_format(tmp_path, payload):
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        load_checkpoint(path)


def test_rng_roundtrip_includes_python_numpy_and_torch():
    import random
    from ai.LSTM.src.checkpoint import rng_state, restore_rng
    training.seed_everything(19)
    state = rng_state()
    expected = (random.random(), np.random.rand(3), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.rand(3), expected[1])
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["empty", "nan"])
def test_run_epoch_rejects_empty_or_nonfinite_data(bad):
    class Identity(nn.Module):
        def forward(self, x, seq_lens=None):
            return x
    samples = [] if bad == "empty" else [(torch.tensor([float("nan"), 1.0]), torch.tensor(1), torch.tensor(1), {})]
    loader = DataLoader(samples, collate_fn=collate_video_frames)
    error = ValueError if bad == "empty" else FloatingPointError
    with pytest.raises(error):
        training.run_epoch(Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))


def test_nonfinite_gradient_does_not_update_weights_without_amp():
    class Classifier(nn.Linear):
        def forward(self, x, seq_lens=None):
            return super().forward(x)

    model = Classifier(3, 2)
    before = model.weight.detach().clone()
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = DataLoader([(torch.ones(3), torch.tensor(0), torch.tensor(1), {})],
                        collate_fn=collate_video_frames)
    with pytest.raises(RuntimeError, match="non-finite"):
        training.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                           optimizer, scaler, amp=False)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_best_selection_tiebreak_and_recovery(tmp_path, fake_loaders, monkeypatch):
    cfg = tiny_config(tmp_path)
    cfg.epochs = 3
    real_epoch = training.run_epoch
    validation = iter([{"f1": 0.8, "loss": 0.5}, {"f1": 0.8, "loss": 0.4}, {"f1": 0.7, "loss": 0.1}])
    def run(*args, **kwargs):
        result = real_epoch(*args, **kwargs)
        if kwargs.get("desc", "").startswith("Val"):
            result.update(next(validation))
        return result
    monkeypatch.setattr(training, "run_epoch", run)
    training.train(cfg)
    folder = Path(cfg.save_dir)
    last = load_checkpoint(folder / "last.pt")
    best = load_checkpoint(folder / "best.pt")
    assert last["epoch"] == 3 and best["epoch"] == last["best_epoch"] == 2
    for key in best["model"]:
        torch.testing.assert_close(best["model"][key], last["best_model"][key], rtol=0, atol=0)
    assert any(not torch.equal(best["model"][key], last["model"][key]) for key in best["model"])
    cfg.resume = str(folder / "best.pt")
    with pytest.raises(ValueError, match="full last.pt"):
        training.train(cfg)
    (folder / "best.pt").unlink()
    cfg.resume = str(folder / "last.pt")
    training.train(cfg)
    recovered = load_checkpoint(folder / "best.pt")
    assert recovered["epoch"] == 2
    for key in best["model"]:
        torch.testing.assert_close(recovered["model"][key], best["model"][key], rtol=0, atol=0)
    log = (folder / "train.log").read_text()
    assert "Epoch 1/3" in log and "Resumed epoch 3" in log


def test_atomic_replace_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    from ai.LSTM.src import checkpoint
    path = tmp_path / 'last.pt'
    atomic_save({'epoch': 1}, path)
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        atomic_save({'epoch': 2}, path)
    assert torch.load(path, weights_only=True) == {'epoch': 1}
    assert list(tmp_path.iterdir()) == [path]


def test_config_json_roundtrip_and_independent_defaults(tmp_path):
    from ai.LSTM.src.config import load_config
    cfg = load_config()
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg.to_dict()))
    restored = load_config(path)
    assert restored == cfg
    cfg.lr = 0.5
    assert load_config().lr == restored.lr
