"""Base checkpoint integrity contract tests (fake S3 serving bytes)."""

from __future__ import annotations

import hashlib

import pytest

from yolopipe import constants
from yolopipe.checkpoint import ensure_base_checkpoint
from yolopipe.errors import IntegrityError
from yolopipe.storage import S3Gateway


@pytest.fixture
def gateway(bytes_s3_factory):
    return S3Gateway(bytes_s3_factory())


def _register(gateway, data: bytes, *, with_companion: str | None = None) -> None:
    client = gateway._client  # noqa: SLF001 - test reaches into the injected fake
    client.put(constants.EMBY_OSS_BUCKET, constants.BASE_CHECKPOINT_KEY, data)
    if with_companion is not None:
        client.put(
            constants.EMBY_OSS_BUCKET,
            constants.BASE_CHECKPOINT_SHA256_KEY,
            with_companion.encode('utf-8'),
        )


def test_good_checkpoint_downloads_and_verifies(gateway, tmp_path, monkeypatch):
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    real_sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(constants, 'BASE_CHECKPOINT_SHA256', real_sha)
    # checkpoint module caches the constant reference lazily, so patch there too.
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', real_sha)
    _register(gateway, data)

    path = ensure_base_checkpoint(gateway, tmp_path)
    assert path.exists()
    assert path.stat().st_size == constants.BASE_CHECKPOINT_SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == real_sha


def test_missing_checkpoint_raises_with_guidance(gateway, tmp_path):
    with pytest.raises(IntegrityError, match='missing'):
        ensure_base_checkpoint(gateway, tmp_path)


def test_size_mismatch_raises(gateway, tmp_path, monkeypatch):
    data = b'\x01' * (constants.BASE_CHECKPOINT_SIZE - 1)
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', hashlib.sha256(data).hexdigest())
    _register(gateway, data)
    with pytest.raises(IntegrityError, match='size'):
        ensure_base_checkpoint(gateway, tmp_path)


def test_corrupt_checkpoint_raises_on_sha(gateway, tmp_path):
    # Correct size, wrong content -> sha mismatch against the real constant.
    data = b'\x02' * constants.BASE_CHECKPOINT_SIZE
    _register(gateway, data)
    with pytest.raises(IntegrityError, match='sha256 mismatch'):
        ensure_base_checkpoint(gateway, tmp_path)


def test_companion_mismatch_raises_before_download(gateway, tmp_path):
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    _register(gateway, data, with_companion='deadbeef  yolo11n-seg.pt\n')
    with pytest.raises(IntegrityError, match='companion'):
        ensure_base_checkpoint(gateway, tmp_path)


def test_companion_match_allows_good_download(gateway, tmp_path, monkeypatch):
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    real_sha = hashlib.sha256(data).hexdigest()
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', real_sha)
    _register(gateway, data, with_companion=f'{real_sha}  yolo11n-seg.pt\n')
    path = ensure_base_checkpoint(gateway, tmp_path)
    assert path.exists()
