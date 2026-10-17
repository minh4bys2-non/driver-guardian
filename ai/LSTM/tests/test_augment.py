import numpy as np
import pytest

from ai.LSTM.src.augment import get_video_augmenter, letterbox


@pytest.mark.parametrize('shape', [(2, 4), (4, 2), (1, 100)])
def test_letterbox_centers_image_and_preserves_rgb(shape):
    frame = np.full((*shape, 3), (10, 40, 200), dtype=np.uint8)
    actual = letterbox(frame, new_size=8, color=(1, 2, 3))
    scale = 8 / max(shape)
    h, w = [max(1, round(size * scale)) for size in shape]
    expected = np.full((8, 8, 3), (1, 2, 3), dtype=np.uint8)
    expected[(8-h)//2:(8-h)//2+h, (8-w)//2:(8-w)//2+w] = (10, 40, 200)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def test_video_seed_can_be_replayed_and_reset():
    augmenter = get_video_augmenter()
    frame = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    assert augmenter.set_video_seed(42) == 42
    first = augmenter.augment_frame_only(frame)
    np.testing.assert_array_equal(augmenter.augment_frame_only(frame), first)
    augmenter.reset_video_seed()
    assert augmenter.video_seed is None
    np.testing.assert_array_equal(augmenter.augment_frame_only(frame, seed=42), first)


def test_augmentation_error_preserves_cause(monkeypatch):
    augmenter = get_video_augmenter()
    def fail(**kwargs):
        raise ValueError('invalid boxes')
    monkeypatch.setattr(augmenter, 'transform', fail)
    with pytest.raises(RuntimeError, match='Video augmentation failed') as error:
        augmenter(np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(error.value.__cause__, ValueError)
