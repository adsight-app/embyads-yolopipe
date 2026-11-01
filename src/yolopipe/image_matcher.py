"""Perceptual pair matching via OpenCV normalized correlation.

The matcher decodes two local images with OpenCV, reduces multi-channel BGR
input to grayscale, resizes any over-large image to a fixed square, and scores
the pair with ``TM_CCOEFF_NORMED``. A score at or above the threshold marks the
pair as a perceptual duplicate. Shape or decode problems fail with contextual
errors rather than surfacing an ambiguous NumPy truth-value error.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import constants
from .errors import ConversionError, ValidationError

_RESIZE = constants.DEDUP_RESIZE


def _load_prepared(path: Path | str) -> np.ndarray:
    """Decode ``path`` to a grayscale 2-D array, resized when over-large."""
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f'image for matching is missing: {p}')

    image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ConversionError(f'failed to decode image for matching: {p}')

    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim != 2:
        raise ConversionError(f'image for matching has unexpected shape {image.shape!r}: {p}')

    height, width = image.shape[:2]
    if height > _RESIZE or width > _RESIZE:
        image = cv2.resize(image, (_RESIZE, _RESIZE), interpolation=cv2.INTER_LINEAR)

    return image


def _scalar_score(result: np.ndarray, *, path_a: Path, path_b: Path) -> float:
    """Collapse a matchTemplate result to a single float, rejecting ambiguity."""
    squeezed = np.squeeze(result)
    if squeezed.ndim != 0:
        raise ConversionError(
            f'correlation between {path_a} and {path_b} did not reduce to a scalar '
            f'(shape {result.shape!r}); images are likely incompatible sizes'
        )
    return float(squeezed)


def match_images(
    path_a: Path | str,
    path_b: Path | str,
    threshold: float = constants.DEDUP_THRESHOLD,
) -> bool:
    """Return ``True`` when the two images correlate at or above ``threshold``.

    Both images are prepared identically (grayscale, resized to a fixed square
    when over-large) before scoring with ``cv2.TM_CCOEFF_NORMED``.
    """
    a_path = Path(path_a)
    b_path = Path(path_b)
    a = _load_prepared(a_path)
    b = _load_prepared(b_path)

    if a.shape != b.shape:
        raise ConversionError(
            f'cannot correlate images of differing shapes {a.shape!r} and {b.shape!r}: {a_path} vs {b_path}'
        )

    result = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    score = _scalar_score(result, path_a=a_path, path_b=b_path)
    return score >= threshold
