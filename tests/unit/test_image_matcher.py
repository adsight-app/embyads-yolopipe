"""Contract tests for the OpenCV perceptual pair matcher."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from yolopipe.errors import ConversionError, ValidationError
from yolopipe.image_matcher import match_images


def _write_gray(path, array) -> None:
    Image.fromarray(array.astype('uint8'), mode='L').save(path)


def _write_rgb(path, array) -> None:
    Image.fromarray(array.astype('uint8'), mode='RGB').save(path)


def _gradient(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0, 255, width, dtype='float64')
    ys = np.linspace(0, 255, height, dtype='float64')
    return ((xs[None, :] + ys[:, None]) / 2.0).astype('uint8')


def test_identical_images_are_duplicates(tmp_path):
    arr = _gradient(64, 48)
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    _write_gray(a, arr)
    _write_gray(b, arr.copy())
    assert match_images(a, b) is True


def test_unrelated_images_are_not_duplicates(tmp_path):
    rng = np.random.default_rng(1)
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    # A smooth gradient vs. structured checkerboard: low correlation.
    _write_gray(a, _gradient(64, 64))
    checker = np.indices((64, 64)).sum(axis=0) % 2 * 255
    _write_gray(b, checker.astype('uint8'))
    assert match_images(a, b) is False
    # Random noise against the gradient is also not a duplicate.
    c = tmp_path / 'c.png'
    _write_gray(c, rng.integers(0, 256, size=(64, 64), dtype='uint8'))
    assert match_images(a, c) is False


def test_multichannel_input_is_converted_to_grayscale(tmp_path):
    arr = _gradient(50, 50)
    gray = tmp_path / 'gray.png'
    rgb = tmp_path / 'rgb.png'
    _write_gray(gray, arr)
    # Same luminance replicated across channels decodes to identical grayscale.
    _write_rgb(rgb, np.repeat(arr[:, :, None], 3, axis=2))
    assert match_images(gray, rgb) is True


def test_large_images_are_resized_before_matching(tmp_path):
    # Two over-large images of DIFFERENT dimensions still compare because both
    # are resized to the fixed 256x256 square first.
    big_a = tmp_path / 'big_a.png'
    big_b = tmp_path / 'big_b.png'
    _write_gray(big_a, _gradient(400, 300))
    _write_gray(big_b, _gradient(512, 512))
    # Should not raise a shape error and should correlate (both are gradients).
    assert match_images(big_a, big_b) is True


def test_threshold_boundary_is_inclusive(tmp_path):
    # Pin the >= semantics at the exact score of a given pair: a threshold set
    # to the pair's own score must drop (inclusive), one epsilon higher must not.
    import cv2

    from yolopipe.image_matcher import _load_prepared

    arr = _gradient(40, 40)
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    _write_gray(a, arr)
    _write_gray(b, (arr.astype('int16') + 3).clip(0, 255).astype('uint8'))

    prepared_a = _load_prepared(a)
    prepared_b = _load_prepared(b)
    score = float(cv2.matchTemplate(prepared_a, prepared_b, cv2.TM_CCOEFF_NORMED).squeeze())

    assert match_images(a, b, threshold=score) is True  # >= is inclusive
    assert match_images(a, b, threshold=score + 1e-6) is False


def test_default_threshold_matches_constant():
    from yolopipe import constants
    from yolopipe.image_matcher import match_images as mi

    assert mi.__defaults__[0] == constants.DEDUP_THRESHOLD == 0.85


def test_missing_input_raises_validation_error(tmp_path):
    a = tmp_path / 'a.png'
    _write_gray(a, _gradient(10, 10))
    with pytest.raises(ValidationError, match='missing'):
        match_images(a, tmp_path / 'does-not-exist.png')


def test_corrupt_input_raises_conversion_error(tmp_path):
    a = tmp_path / 'a.png'
    _write_gray(a, _gradient(10, 10))
    corrupt = tmp_path / 'corrupt.png'
    corrupt.write_bytes(b'not an image at all')
    with pytest.raises(ConversionError, match='decode'):
        match_images(a, corrupt)


def test_small_shape_mismatch_fails_clearly(tmp_path):
    # Two small (<=256) images of different sizes cannot correlate to a scalar;
    # this must raise a clear error, not an ambiguous NumPy truth-value error.
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    _write_gray(a, _gradient(32, 32))
    _write_gray(b, _gradient(48, 48))
    with pytest.raises(ConversionError, match='shape'):
        match_images(a, b)
