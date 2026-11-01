"""Canonical mask-to-YOLO conversion contract tests with synthetic cv2 masks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from yolopipe.conversion import convert_all, convert_sample
from yolopipe.errors import ConversionError
from yolopipe.models import NormalizedSample

W, H = 100, 80
BILLBOARD_HEX = '#00ff00'  # green
BILLBOARD_BGR = (0, 255, 0)


def _sample(sample_id: str = 'sid', color_map: dict | None = None) -> NormalizedSample:
    return NormalizedSample(
        sample_id=sample_id,
        job_name='20250515-gta5-billboards-01',
        source_image_uri='s3://synthetic/img/a.png',
        source_mask_uri='s3://synthetic/mask/a.png',
        source_filename='a.png',
        extension='.png',
        color_map=color_map or {BILLBOARD_HEX: 'billboard'},
    )


def _write_png(path, array: np.ndarray) -> None:
    assert cv2.imwrite(str(path), array)


def _blank(w: int = W, h: int = H) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fill_rect(array: np.ndarray, x0: int, y0: int, x1: int, y1: int, bgr) -> None:
    array[y0:y1, x0:x1] = bgr


def _make_pair(tmp_path, mask_array: np.ndarray, *, img_size=None):
    img_arr = _blank(*(img_size or (mask_array.shape[1], mask_array.shape[0])))
    mask_path = tmp_path / 'mask.png'
    img_path = tmp_path / 'img.png'
    _write_png(mask_path, mask_array)
    _write_png(img_path, img_arr)
    return img_path, mask_path


def test_background_only_produces_zero_byte_label(tmp_path):
    mask = _blank()
    img_path, mask_path = _make_pair(tmp_path, mask)
    result = convert_sample(_sample(), image_path=img_path, mask_path=mask_path)
    assert result.retained is False
    assert result.label_bytes == b''
    assert len(result.label_bytes) == 0


def test_single_rectangle_label_bytes_and_trailing_newline(tmp_path):
    mask = _blank()
    _fill_rect(mask, 10, 10, 40, 30, BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask)
    result = convert_sample(_sample(), image_path=img_path, mask_path=mask_path)

    assert result.retained is True
    text = result.label_text
    assert text.endswith('\n')
    assert text.count('\n') == 1  # exactly one line + trailing newline
    line = text.rstrip('\n')
    parts = line.split(' ')
    assert parts[0] == '0'
    coords = parts[1:]
    # four corners -> 8 coordinates, each formatted %.6f
    assert len(coords) == 8
    for c in coords:
        assert '.' in c
        assert len(c.split('.')[1]) == 6


def test_disconnected_shapes_produce_two_sorted_lines(tmp_path):
    mask = _blank()
    _fill_rect(mask, 5, 5, 25, 25, BILLBOARD_BGR)
    _fill_rect(mask, 60, 50, 90, 70, BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask)
    result = convert_sample(_sample(), image_path=img_path, mask_path=mask_path)

    lines = result.label_text.rstrip('\n').split('\n')
    assert len(lines) == 2
    assert lines == sorted(lines)  # lexicographic order


def test_small_contour_discarded_as_background(tmp_path):
    mask = _blank()
    # 2x2 area = 4 < 5.0 threshold
    _fill_rect(mask, 10, 10, 12, 12, BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask)
    result = convert_sample(_sample(), image_path=img_path, mask_path=mask_path)
    assert result.retained is False
    assert result.label_bytes == b''
    assert result.discarded_small >= 1


def test_simple_polygon_under_four_points_discarded(tmp_path):
    # A thin triangle simplifies to < 4 points and is discarded.
    mask = _blank()
    pts = np.array([[10, 10], [60, 12], [11, 60]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask)
    result = convert_sample(_sample(), image_path=img_path, mask_path=mask_path)
    assert result.retained is False
    assert result.discarded_simple >= 1
    assert result.discarded_small == 0


def test_color_map_key_permutations_produce_identical_bytes(tmp_path):
    mask = _blank()
    _fill_rect(mask, 10, 10, 40, 30, BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask)

    map_a = {BILLBOARD_HEX: 'billboard', '#000000': 'BACKGROUND'}
    map_b = {'#000000': 'BACKGROUND', BILLBOARD_HEX: 'billboard'}

    r_a = convert_sample(_sample(color_map=map_a), image_path=img_path, mask_path=mask_path)
    r_b = convert_sample(_sample(color_map=map_b), image_path=img_path, mask_path=mask_path)
    assert r_a.label_bytes == r_b.label_bytes
    assert r_a.retained is True


def test_background_class_ignored(tmp_path):
    mask = _blank()
    # Fill with a non-billboard color mapped to BACKGROUND.
    _fill_rect(mask, 10, 10, 50, 50, (255, 0, 0))
    img_path, mask_path = _make_pair(tmp_path, mask)
    sample = _sample(color_map={BILLBOARD_HEX: 'billboard', '#0000ff': 'BACKGROUND'})
    result = convert_sample(sample, image_path=img_path, mask_path=mask_path)
    # Blue fill (#0000ff -> BGR (255,0,0)) maps to BACKGROUND, ignored.
    assert result.retained is False


def test_dimension_mismatch_raises(tmp_path):
    mask = _blank(W, H)
    _fill_rect(mask, 10, 10, 40, 30, BILLBOARD_BGR)
    img_path, mask_path = _make_pair(tmp_path, mask, img_size=(W + 10, H))
    with pytest.raises(ConversionError, match='dimension mismatch'):
        convert_sample(_sample(), image_path=img_path, mask_path=mask_path)


def test_numeric_key_ordering(tmp_path):
    # Numeric-string keys would sort by int value; verify sort helper total order.
    from yolopipe.conversion import _color_key_sort

    keys = ['10', '2', '#ffffff', 'alpha', '1']
    ordered = sorted(keys, key=_color_key_sort)
    assert ordered == ['1', '2', '10', '#ffffff', 'alpha']


def test_convert_all_aggregates_stats(tmp_path):
    mask1 = _blank()
    _fill_rect(mask1, 10, 10, 40, 30, BILLBOARD_BGR)
    mask2 = _blank()  # background only -> empty

    d1 = tmp_path / 's1'
    d2 = tmp_path / 's2'
    d1.mkdir()
    d2.mkdir()
    img1, m1 = _make_pair(d1, mask1)
    img2, m2 = _make_pair(d2, mask2)

    s1 = _sample('s1')
    s2 = _sample('s2')
    result = convert_all(
        [s1, s2],
        image_paths={'s1': img1, 's2': img2},
        mask_paths={'s1': m1, 's2': m2},
    )
    assert result.stats['input_rows'] == 2
    assert result.stats['retained_labels'] == 1
    assert result.stats['empty_labels'] == 1
    assert result.stats['contours'] >= 1
    assert result.labels['s2'].label_bytes == b''
