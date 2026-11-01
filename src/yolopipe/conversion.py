"""Canonical semantic-mask to YOLO-segmentation label conversion (local CPU).

Each survivor's mask is decoded with OpenCV. The source image and mask must
share identical ``(width, height)`` or :class:`ConversionError` is raised. The
``internal-color-map`` keys are iterated in a fixed total order so that the
emitted label bytes are independent of source map/contour ordering: numeric
keys sort as ``(0, int(k), k)`` and all others as ``(1, 0, k)``. Only the
ground-truth class ``billboard`` maps to YOLO class id ``0``; the class
``BACKGROUND`` is ignored exactly. Contours use ``RETR_EXTERNAL`` /
``CHAIN_APPROX_SIMPLE``; contours below the minimum area are dropped, polygons
are simplified with epsilon ``0.005 * perimeter``, polygons with fewer than four
points are discarded, coordinates are normalized by width/height and formatted
with ``%.6f``, complete lines are sorted lexicographically, joined with ``\n``
and given one trailing ``\n``. A sample with no retained contour produces a
zero-byte label file.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import constants
from .errors import ConversionError
from .models import NormalizedSample


@dataclass
class SampleConversion:
    """Per-sample conversion result: the exact label bytes plus retained stats."""

    sample_id: str
    label_text: str
    retained: bool
    contours: int = 0
    discarded_small: int = 0
    discarded_simple: int = 0

    @property
    def label_bytes(self) -> bytes:
        return self.label_text.encode('utf-8')


def _color_key_sort(key: str) -> tuple[int, int, str]:
    """Total order over color-map keys: numeric keys first by value, then others by key."""
    if key.isdigit():
        return (0, int(key), key)
    return (1, 0, key)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` hex color to an OpenCV BGR tuple."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b, g, r)


def _image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for an image, raising ConversionError if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ConversionError(f'unreadable image: {path.name}')
    height, width = img.shape[0], img.shape[1]
    return width, height


def convert_sample(
    sample: NormalizedSample,
    *,
    image_path: Path,
    mask_path: Path,
) -> SampleConversion:
    """Convert one survivor's mask to canonical YOLO-segmentation label text.

    ``image_path`` and ``mask_path`` are local cached files. The source image is
    read only to verify that its dimensions match the mask.
    """
    img_width, img_height = _image_size(Path(image_path))

    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_img is None:
        raise ConversionError(f'unreadable mask: {Path(mask_path).name}')
    mask_height, mask_width = mask_img.shape[0], mask_img.shape[1]

    if (img_width, img_height) != (mask_width, mask_height):
        raise ConversionError(
            f'image/mask dimension mismatch for {sample.sample_id}: '
            f'image=({img_width},{img_height}) mask=({mask_width},{mask_height})'
        )

    width = float(img_width)
    height = float(img_height)

    lines: list[str] = []
    total_contours = 0
    discarded_small = 0
    discarded_simple = 0

    for color_key in sorted(sample.color_map, key=_color_key_sort):
        class_name = sample.color_map[color_key]
        if class_name in constants.IGNORED_CLASSES:
            continue
        if class_name != constants.GROUND_TRUTH_CLASS:
            # Manifest validation already rejects other classes; ignore defensively.
            continue

        target_bgr = _hex_to_bgr(color_key)
        binary = cv2.inRange(mask_img, target_bgr, target_bgr)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_contours += len(contours)

        for contour in contours:
            if cv2.contourArea(contour) < constants.MIN_CONTOUR_AREA:
                discarded_small += 1
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = constants.EPSILON_FACTOR * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < constants.MIN_CONTOUR_POINTS:
                discarded_simple += 1
                continue

            flat = approx.reshape(-1, 2)
            coords: list[str] = []
            for x, y in flat:
                coords.append('%.6f' % (float(x) / width))
                coords.append('%.6f' % (float(y) / height))
            lines.append(f'{constants.CLASS_ID} ' + ' '.join(coords))

    lines.sort()

    if lines:
        label_text = '\n'.join(lines) + '\n'
        retained = True
    else:
        label_text = ''
        retained = False

    return SampleConversion(
        sample_id=sample.sample_id,
        label_text=label_text,
        retained=retained,
        contours=total_contours,
        discarded_small=discarded_small,
        discarded_simple=discarded_simple,
    )


@dataclass
class ConversionResult:
    """Aggregated conversion over all survivors: per-sample results plus stats."""

    labels: dict[str, SampleConversion]
    stats: dict = field(default_factory=dict)


def convert_all(
    samples: list[NormalizedSample],
    *,
    image_paths: dict[str, Path],
    mask_paths: dict[str, Path],
) -> ConversionResult:
    """Convert every survivor deterministically and aggregate conversion stats.

    ``image_paths`` and ``mask_paths`` map ``sample_id`` to local cached files.
    Output ordering follows the input ``samples`` order.
    """
    labels: dict[str, SampleConversion] = {}
    counter: Counter = Counter()

    counter['input_rows'] = 0
    counter['retained_labels'] = 0
    counter['empty_labels'] = 0
    counter['contours'] = 0
    counter['discarded_small'] = 0
    counter['discarded_simple'] = 0

    for sample in samples:
        counter['input_rows'] += 1
        result = convert_sample(
            sample,
            image_path=image_paths[sample.sample_id],
            mask_path=mask_paths[sample.sample_id],
        )
        labels[sample.sample_id] = result
        if result.retained:
            counter['retained_labels'] += 1
        else:
            counter['empty_labels'] += 1
        counter['contours'] += result.contours
        counter['discarded_small'] += result.discarded_small
        counter['discarded_simple'] += result.discarded_simple

    return ConversionResult(labels=labels, stats=dict(counter))
