"""Localization contract tests: atomic cache, reuse validation, fetch scoping."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from yolopipe.errors import ConversionError
from yolopipe.localize import localize_images, localize_masks, object_cache_path
from yolopipe.models import NormalizedSample
from yolopipe.storage import S3Gateway


def _png_bytes(width: int, height: int, fill: int = 120) -> bytes:
    arr = np.full((height, width), fill, dtype='uint8')
    buf = io.BytesIO()
    Image.fromarray(arr, mode='L').save(buf, format='PNG')
    return buf.getvalue()


def _sample(name: str, *, ext: str = '.png') -> NormalizedSample:
    return NormalizedSample(
        sample_id=f'id-{name}',
        job_name='job-a',
        source_image_uri=f's3://src/img/{name}{ext}',
        source_mask_uri=f's3://src/mask/{name}{ext}',
        source_filename=f'{name}{ext}',
        extension=ext,
        color_map={'#ff0000': 'billboard'},
    )


def _ensure_roots(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'cache').mkdir()


def test_images_localized_under_sample_keyed_paths(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({s.source_image_uri: _png_bytes(32, 24)})
    gateway = S3Gateway(client)

    paths = localize_images(gateway, [s], cwd=tmp_path)
    expected = object_cache_path(s, 'image', tmp_path)
    assert paths[s.sample_id] == expected
    assert expected.is_file()
    assert expected.name == f'{s.sample_id}.image{s.extension}'
    # The image URI was fetched; no mask was touched.
    assert s.source_image_uri in client.downloaded
    assert s.source_mask_uri not in client.accessed


def test_valid_cached_image_is_not_redownloaded(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({s.source_image_uri: _png_bytes(32, 24)})
    gateway = S3Gateway(client)

    localize_images(gateway, [s], cwd=tmp_path)
    assert client.downloaded.count(s.source_image_uri) == 1
    # Second pass reuses the decodable cached object: no new download.
    localize_images(gateway, [s], cwd=tmp_path)
    assert client.downloaded.count(s.source_image_uri) == 1


def test_corrupt_cached_image_is_redownloaded(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({s.source_image_uri: _png_bytes(32, 24)})
    gateway = S3Gateway(client)

    localize_images(gateway, [s], cwd=tmp_path)
    # Corrupt the cached object; reuse must NOT trust mere existence.
    object_cache_path(s, 'image', tmp_path).write_bytes(b'garbage')
    localize_images(gateway, [s], cwd=tmp_path)
    assert client.downloaded.count(s.source_image_uri) == 2


def test_undecodable_download_raises_contextual_error(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({s.source_image_uri: b'not a png'})
    gateway = S3Gateway(client)
    with pytest.raises(ConversionError, match='id-a'):
        localize_images(gateway, [s], cwd=tmp_path)


def test_masks_fetched_only_for_survivors(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    survivor = _sample('keep')
    discarded = _sample('drop')
    objects = {
        survivor.source_image_uri: _png_bytes(20, 20),
        discarded.source_image_uri: _png_bytes(20, 20),
        survivor.source_mask_uri: _png_bytes(20, 20),
        # Deliberately NOT registering the discarded mask: fetching it raises.
    }
    client = object_s3_factory(objects)
    gateway = S3Gateway(client)

    # All selected images localized first.
    localize_images(gateway, [survivor, discarded], cwd=tmp_path)
    # Only the survivor's mask is fetched.
    mask_paths = localize_masks(gateway, [survivor], cwd=tmp_path)

    assert survivor.source_mask_uri in client.downloaded
    assert discarded.source_mask_uri not in client.accessed
    assert mask_paths[survivor.sample_id].is_file()


def test_mask_dimension_mismatch_fails(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory(
        {
            s.source_image_uri: _png_bytes(20, 20),
            s.source_mask_uri: _png_bytes(20, 30),  # height differs
        }
    )
    gateway = S3Gateway(client)
    localize_images(gateway, [s], cwd=tmp_path)
    with pytest.raises(ConversionError, match='dimension mismatch'):
        localize_masks(gateway, [s], cwd=tmp_path)


def test_mask_requires_localized_image(tmp_path, object_s3_factory):
    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({s.source_mask_uri: _png_bytes(20, 20)})
    gateway = S3Gateway(client)
    with pytest.raises(ConversionError, match='must be localized'):
        localize_masks(gateway, [s], cwd=tmp_path)
    # The mask object must not be fetched when the image precondition fails.
    assert s.source_mask_uri not in client.accessed


def test_failed_download_leaves_no_partial(tmp_path, object_s3_factory):
    from yolopipe.errors import TransferError

    _ensure_roots(tmp_path)
    s = _sample('a')
    client = object_s3_factory({})  # object missing -> download_file raises
    gateway = S3Gateway(client)
    with pytest.raises(TransferError):
        localize_images(gateway, [s], cwd=tmp_path)
    # No cached object and no leftover temp .part file.
    assert not object_cache_path(s, 'image', tmp_path).exists()
    objects_dir = object_cache_path(s, 'image', tmp_path).parent
    assert not any(p.suffix == '.part' for p in objects_dir.iterdir())
