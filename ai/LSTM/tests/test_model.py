import pytest
import torch
from torch import nn

from ai.LSTM.src.model import DrowsyClassifier
from ai.LSTM.src.model_backbone_neck import BackboneNeck


@pytest.fixture
def model():
    return DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                            input_dim=8, hidden_dim=8, chunk_size=2)


def test_matches_separate_modules(model):
    model.eval()
    frames = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    with torch.inference_mode():
        p4 = model.backbone_neck(frames.flatten(0, 1).float() / 255)[1]
        expected = model.conv_gru(p4.reshape(2, 3, *p4.shape[1:]))
        actual, weights, states = model(frames, return_weights=True, return_state=True)
        sequence = model(frames, return_sequence=True)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2) and sequence.shape == (2, 3, 2)
    assert weights['spatial'].shape == (2, 3, 1, 4, 4)
    assert states[0].shape == (2, 8, 4, 4)


def test_padding_backward_and_batchnorm(model):
    model.freeze_bn = True
    model.gradient_checkpointing = True
    model.train()
    layers = [m for m in model.backbone_neck.modules() if isinstance(m, nn.BatchNorm2d)]
    before = [m.running_mean.clone() for m in layers]
    frames = torch.randn(2, 3, 3, 64, 64, requires_grad=True)
    logits = model(frames, seq_lens=torch.tensor([3, 1]), return_sequence=True)
    assert not logits[1, 1:].any()
    logits.sum().backward()
    assert not frames.grad[1, 1:].any()
    assert frames.grad[0].abs().sum() > 0
    assert model.conv_gru.convgru.gates.weight.grad is not None
    for value, layer in zip(before, layers):
        torch.testing.assert_close(value, layer.running_mean)


@pytest.mark.parametrize('prefix', ['', 'trunk_'])
def test_checkpoint_preserves_weights_and_infers_channels(tmp_path, prefix):
    trunk = BackboneNeck(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1))
    architecture = {prefix + name: getattr(trunk, name) for name in ('backbone_w', 'backbone_n', 'neck_n')}
    path = tmp_path / 'backbone.pt'
    torch.save({'metadata': {'architecture': architecture}, 'model': trunk.state_dict()}, path)
    model = DrowsyClassifier(backbone_checkpoint=path, input_dim=8, hidden_dim=8).eval()
    assert model.conv_gru.p4_channels == 32
    for name, value in trunk.state_dict().items():
        torch.testing.assert_close(model.backbone_neck.state_dict()[name], value)
    with torch.inference_mode():
        assert model(torch.rand(1, 2, 3, 64, 64)).shape == (1, 2)


@pytest.mark.parametrize('lengths', [[0, 1], [4, 1], [1], [1.5, 2]])
def test_invalid_lengths(model, lengths):
    with pytest.raises(ValueError, match='seq_lens'):
        model(torch.rand(2, 3, 3, 64, 64), seq_lens=torch.tensor(lengths))


@pytest.mark.parametrize("frozen", [False, True])
def test_optimizer_updates_only_trainable_parts(frozen):
    torch.manual_seed(23)
    model = DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                             input_dim=8, hidden_dim=8, freeze_backbone=frozen, freeze_bn=frozen)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    frames = torch.randint(256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    loss = nn.CrossEntropyLoss()(model(frames), torch.tensor([0, 1]))
    loss.backward()
    for module in (model.conv_gru.adapter[0], model.conv_gru.convgru.gates,
                   model.conv_gru.convgru.candidate, model.conv_gru.spatial_attention,
                   model.conv_gru.fc_out[1]):
        assert module.weight.grad.abs().sum() > 0
    trunk_weight = model.backbone_neck.backbone.stem.conv.weight
    assert trunk_weight.grad is None if frozen else trunk_weight.grad.abs().sum() > 0
    optimizer.step()
    after = model.state_dict()
    assert not torch.equal(before["conv_gru.fc_out.1.weight"], after["conv_gru.fc_out.1.weight"])
    if frozen:
        for key in before:
            if key.startswith("backbone_neck."):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    else:
        assert not torch.equal(before["backbone_neck.backbone.stem.conv.weight"], trunk_weight)


def test_eval_is_independent_of_chunk_size_and_uint8_conversion(model):
    model.eval()
    frames = torch.randint(256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        expected = model(frames, seq_lens=[3, 1])
        model.chunk_size = 0
        actual = model(frames.float() / 255, seq_lens=[3, 1])
    torch.testing.assert_close(actual, expected)


def test_frozen_features_match_training_and_inference():
    model = DrowsyClassifier(backbone_w=(8, 16, 16, 32, 64), backbone_n=(1, 1, 1, 1),
                             input_dim=8, hidden_dim=8, freeze_backbone=True, freeze_bn=True)
    frames = torch.rand(4, 3, 64, 64)
    before = {key: value.clone() for key, value in model.backbone_neck.state_dict().items()}
    for training in (True, False, True):
        model.train(training)
        assert model.conv_gru.training == training
        assert all(not layer.training for layer in model.backbone_neck.modules())
        with torch.no_grad():
            together = model._extract_p4(frames)
            separately = torch.cat([model._extract_p4(frame[None]) for frame in frames])
        torch.testing.assert_close(together, separately, rtol=1e-4, atol=1e-5)
        if training:
            train_features = together
        else:
            torch.testing.assert_close(together, train_features, rtol=0, atol=0)
    for key, value in model.backbone_neck.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(2, 3, 64, 64), (1, 2, 1, 64, 64), (1, 0, 3, 64, 64)])
def test_invalid_video_shape_is_rejected(model, shape):
    with pytest.raises(ValueError, match='B,T,3,H,W'):
        model(torch.empty(shape))
