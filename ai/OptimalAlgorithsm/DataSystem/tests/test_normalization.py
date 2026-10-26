from dataclasses import replace

import numpy as np
import pytest

from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.normalization import PhysicalNormalizer


@pytest.fixture
def normalizer():
    return PhysicalNormalizer(DEFAULT_CONFIG)


@pytest.mark.parametrize('key', [
    'blink_frequency', 'blink_duration', 'perclos', 'yawn_frequency',
    'nod_duration', 'nod_frequency', 'dominant_head_motion_frequency', 'neuralnetwork_score',
])
def test_every_feature_is_reversible_and_rejects_out_of_range(normalizer, key):
    raw = dict.fromkeys(normalizer.scales, 0.)
    upper = normalizer.scales[key]
    for fraction in (0, .125, .5, 1):
        values = raw | {key: upper * fraction}
        result = normalizer.transform(values)
        assert result[key] == pytest.approx(fraction)
        assert result[key] * upper == pytest.approx(values[key])
        assert all(result[other] == 0 for other in raw if other != key)
    for value in (-1e-8, float('nan'), float('inf'), upper + 2e-6, True, None):
        with pytest.raises(ValueError, match=key):
            normalizer.transform(raw | {key: value})
    assert normalizer.transform(raw | {key: upper + 5e-7})[key] == 1


@pytest.mark.parametrize('value', [np.bool_(True), '0.5', [0.5], np.array([0.5])])
def test_rejects_non_numeric_scalars(normalizer, value):
    raw = dict.fromkeys(normalizer.scales, 0.)
    with pytest.raises(ValueError, match='perclos'):
        normalizer.transform(raw | {'perclos': value})


def test_accepts_numpy_numeric_scalars(normalizer):
    raw = {key: np.float64(upper / 2) for key, upper in normalizer.scales.items()}
    assert all(value == pytest.approx(.5) for value in normalizer.transform(raw).values())
    raw = dict.fromkeys(normalizer.scales, np.int64(0))
    assert all(value == 0 for value in normalizer.transform(raw).values())


def test_default_scales_match_detector_units(normalizer):
    assert normalizer.scales == {
        'blink_frequency': 604., 'blink_duration': 2000, 'perclos': 1.,
        'yawn_frequency': 18., 'nod_duration': 3.5, 'nod_frequency': 78.,
        'dominant_head_motion_frequency': 15., 'neuralnetwork_score': 1.,
    }


def test_normalization_identity_is_independent_of_dictionary_order():
    first = replace(DEFAULT_CONFIG, windows={'blink': 10, 'yawn': 20})
    second = replace(DEFAULT_CONFIG, windows={'yawn': 20, 'blink': 10})
    assert PhysicalNormalizer(first).id == PhysicalNormalizer(second).id
    assert PhysicalNormalizer(first).id != PhysicalNormalizer(DEFAULT_CONFIG).id
