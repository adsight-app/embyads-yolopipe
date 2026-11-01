"""Atomic localization of selected images and survivor masks to the object cache.

Objects are cached under ``./cache/yolopipe/objects/<sample_id>.<role><ext>``
with ``role`` being ``image`` or ``mask``. Downloads go through
:meth:`S3Gateway.download`, which writes to a same-directory temporary file and
``os.replace`` into place, so interrupted transfers never leave a usable
partial. Reuse never trusts mere file existence: a cached object is re-decoded
(and, for masks, dimension-checked against its image) before it is accepted.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .errors import ConversionError, ValidationError
from .models import NormalizedSample, S3Uri
from .paths import cache_objects_dir


def object_cache_path(sample: NormalizedSample, role: str, cwd: Path | None = None) -> Path:
    """Stable cache path for a sample's image or mask, keyed by sample_id."""
    if role not in ('image', 'mask'):
        raise ValidationError(f'unknown object role {role!r}')
    return cache_objects_dir(cwd) / f'{sample.sample_id}.{role}{sample.extension}'


def _decode_dimensions(path: Path) -> tuple[int, int]:
    """Decode ``path`` and return ``(width, height)``; raise on unreadable content."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ConversionError(f'cached object is not a decodable image: {path}')
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ConversionError(f'cached object has empty dimensions: {path}')
    return width, height


def _valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        _decode_dimensions(path)
    except ConversionError:
        return False
    return True


def _localize_one(gateway, uri: str, dest: Path) -> Path:
    """Download ``uri`` to ``dest`` unless a decodable copy is already cached."""
    if _valid_image(dest):
        return dest
    gateway.download(S3Uri.parse(uri), dest)
    # Re-verify the freshly downloaded object decodes before accepting it.
    _decode_dimensions(dest)
    return dest


def localize_images(
    gateway,
    samples: list[NormalizedSample],
    cwd: Path | None = None,
) -> dict[str, Path]:
    """Cache source images for every selected sample, preserving input order.

    Returns a ``sample_id -> cached image path`` map. Runs before deduplication,
    so it fetches images for all selected samples, never masks.
    """
    paths: dict[str, Path] = {}
    for sample in samples:
        dest = object_cache_path(sample, 'image', cwd)
        try:
            paths[sample.sample_id] = _localize_one(gateway, sample.source_image_uri, dest)
        except ConversionError as exc:
            raise ConversionError(
                f'{sample.job_name} sample {sample.sample_id}: unreadable source image: {exc.message}'
            ) from exc
    return paths


def localize_masks(
    gateway,
    survivors: list[NormalizedSample],
    cwd: Path | None = None,
) -> dict[str, Path]:
    """Cache annotation masks for survivors only and validate image/mask parity.

    Returns a ``sample_id -> cached mask path`` map. Each survivor's image must
    already be localized; the mask is rejected unless its ``(width, height)``
    matches the image exactly (R4.3 precondition).
    """
    paths: dict[str, Path] = {}
    for sample in survivors:
        image_path = object_cache_path(sample, 'image', cwd)
        if not _valid_image(image_path):
            raise ConversionError(
                f'{sample.job_name} sample {sample.sample_id}: source image must be localized before its mask'
            )
        mask_path = object_cache_path(sample, 'mask', cwd)
        try:
            _localize_one(gateway, sample.source_mask_uri, mask_path)
        except ConversionError as exc:
            raise ConversionError(
                f'{sample.job_name} sample {sample.sample_id}: unreadable annotation mask: {exc.message}'
            ) from exc

        image_dims = _decode_dimensions(image_path)
        mask_dims = _decode_dimensions(mask_path)
        if image_dims != mask_dims:
            raise ConversionError(
                f'{sample.job_name} sample {sample.sample_id}: image/mask dimension mismatch '
                f'{image_dims!r} vs {mask_dims!r}'
            )
        paths[sample.sample_id] = mask_path
    return paths
