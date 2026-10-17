import pytest
import torch

from ai.LSTM.src.model_backbone_neck import BackboneNeck


@pytest.fixture
def trunk():
    return BackboneNeck(backbone_w=(8, 16, 32, 64, 128), backbone_n=(1, 1, 1, 1), neck_n=2).eval()


def checkpoint(trunk, prefix, location):
    architecture = {f"{prefix}{name}": getattr(trunk, name)
                    for name in ("backbone_w", "backbone_n", "neck_n")}
    state = dict(trunk.state_dict(), **{"head.weight": torch.ones(7, 3)})
    config = {"metadata": {"architecture": architecture}} if location == "metadata" else {"cfg": architecture}
    return {**config, "model": state}


@pytest.mark.parametrize("prefix", ["", "trunk_"])
@pytest.mark.parametrize("location", ["metadata", "cfg"])
def test_loads_architecture_and_only_trunk(tmp_path, trunk, prefix, location):
    path = tmp_path / "detector.pt"
    torch.save(checkpoint(trunk, prefix, location), path)
    loaded = BackboneNeck.from_checkpoint(path)
    assert not loaded.training and not hasattr(loaded, "head")
    assert loaded.backbone_w == trunk.backbone_w and loaded.neck_n == 2
    assert loaded.state_dict().keys() == trunk.state_dict().keys()
    for layer in loaded.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            assert layer.eps == 1e-3 and layer.momentum == 0.03
    for key, value in trunk.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for actual, expected in zip(loaded(images), trunk(images)):
            torch.testing.assert_close(actual, expected)


def test_prefers_ema_and_allows_model_weights(tmp_path, trunk):
    data = checkpoint(trunk, "trunk_", "metadata")
    data["ema"] = {key: value.clone() for key, value in data["model"].items()}
    key = "backbone.stem.conv.weight"
    data["ema"][key].fill_(0.25)
    path = tmp_path / "ema.pt"
    torch.save(data, path)
    torch.testing.assert_close(BackboneNeck.from_checkpoint(path).state_dict()[key], data["ema"][key])
    loaded = BackboneNeck.from_checkpoint(path, use_ema=False, eval_mode=False)
    assert loaded.training
    torch.testing.assert_close(loaded.state_dict()[key], data["model"][key])


@pytest.mark.parametrize("ema", [None, {}])
def test_falls_back_to_model_when_ema_missing(trunk, ema):
    data = checkpoint(trunk, "", "metadata")
    data["ema"] = ema
    BackboneNeck.from_checkpoint(data)


def test_rejects_missing_config_and_weights(trunk):
    data = checkpoint(trunk, "", "metadata")
    del data["metadata"]["architecture"]["neck_n"]
    with pytest.raises(ValueError, match="neck_n"):
        BackboneNeck.from_checkpoint(data)
    data = checkpoint(trunk, "", "metadata")
    del data["model"]["backbone.stem.conv.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        BackboneNeck.from_checkpoint(data)
    data["model"] = {"head.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="backbone/neck"):
        BackboneNeck.from_checkpoint(data, strict=False)
