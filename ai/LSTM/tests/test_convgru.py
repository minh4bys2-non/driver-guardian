import pytest
import torch

from ai.LSTM.src.models_convgru import ConvGRUClassifier


@pytest.fixture
def classifier():
    torch.manual_seed(17)
    return ConvGRUClassifier(input_dim=4, hidden_dim=6, spatial_in_channels=(4, 8, 16), dropout=0).eval()


def test_streaming_state_matches_full_sequence(classifier):
    features = torch.randn(2, 5, 8, 3, 3)
    with torch.no_grad():
        expected, state = classifier(features, return_sequence=True, return_state=True)
        first, carried = classifier(features[:, :2], return_sequence=True, return_state=True)
        second, final = classifier(features[:, 2:], hidden_state=carried,
                                   return_sequence=True, return_state=True)
    torch.testing.assert_close(torch.cat([first, second], dim=1), expected)
    torch.testing.assert_close(final[0], state[0])


def test_padding_preserves_logits_state_and_normalized_attention(classifier):
    features = torch.randn(2, 4, 8, 3, 3)
    with torch.no_grad():
        logits, weights, state = classifier(features, seq_lens=[4, 2], return_weights=True, return_state=True)
        expected, short_state = classifier(features[1:2, :2], return_state=True)
        sequence = classifier(features, seq_lens=[4, 2], return_sequence=True)
    torch.testing.assert_close(logits[1:], expected)
    torch.testing.assert_close(state[0][1:], short_state[0])
    torch.testing.assert_close(weights['temporal'], torch.tensor([[.25] * 4, [.5, .5, 0, 0]]))
    torch.testing.assert_close(weights['spatial'].sum((-3, -2, -1)),
                               torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]]))
    assert not sequence[1, 2:].any()


def test_single_frame_multiscale_and_adapted_inputs_agree(classifier):
    p4 = torch.randn(2, 8, 3, 3)
    with torch.no_grad():
        expected = classifier(p4)
        torch.testing.assert_close(classifier((None, p4, None)), expected)
        torch.testing.assert_close(classifier(classifier.adapter(p4)), expected)
        torch.testing.assert_close(classifier(p4, return_sequence=True), expected)


@pytest.mark.parametrize('lengths', [[0, 2], [4, 2], [1], [1.5, 2], [float('nan'), 2]])
def test_invalid_lengths_are_rejected(classifier, lengths):
    with pytest.raises(ValueError, match='seq_lens'):
        classifier(torch.randn(2, 3, 8, 3, 3), seq_lens=lengths)


@pytest.mark.parametrize('shape', [(0, 2, 8, 3, 3), (2, 0, 8, 3, 3), (2, 2, 8, 0, 3)])
def test_empty_features_are_rejected(classifier, shape):
    with pytest.raises(ValueError, match='nonempty'):
        classifier(torch.empty(shape))


def test_invalid_hidden_state_is_rejected(classifier):
    with pytest.raises(ValueError, match='hidden_state'):
        classifier(torch.randn(2, 3, 8, 3, 3), hidden_state=torch.zeros(2, 5, 3, 3))
