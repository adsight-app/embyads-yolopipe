"""Base checkpoint retrieval and integrity verification.

Fetches exactly the published base checkpoint, verifies its size and SHA-256
against the authoritative repository constants, and (when present) cross-checks
a companion ``.sha256`` sidecar. Any mismatch is an :class:`IntegrityError`
carrying bootstrap guidance rather than a silent re-download.
"""

from __future__ import annotations

from pathlib import Path

from yolopipe import constants
from yolopipe.errors import IntegrityError
from yolopipe.models import S3Uri, sha256_file

_BOOTSTRAP_GUIDANCE = (
    'the base checkpoint must be published and verified under '
    f's3://{constants.EMBY_OSS_BUCKET}/{constants.BASE_CHECKPOINT_KEY} '
    f'with sha256 {constants.BASE_CHECKPOINT_SHA256} before training can start'
)


def _base_uri() -> S3Uri:
    return S3Uri(bucket=constants.EMBY_OSS_BUCKET, key=constants.BASE_CHECKPOINT_KEY)


def _companion_uri() -> S3Uri:
    return S3Uri(bucket=constants.EMBY_OSS_BUCKET, key=constants.BASE_CHECKPOINT_SHA256_KEY)


def ensure_base_checkpoint(gateway, cache_root_path: Path) -> Path:
    """Download and verify the base checkpoint into the cache; return its local path.

    Verification is strict: the downloaded object must match both
    ``BASE_CHECKPOINT_SIZE`` and ``BASE_CHECKPOINT_SHA256``. If a companion
    ``.sha256`` sidecar exists, its content must equal the repository constant.
    Any failure raises :class:`IntegrityError` with bootstrap guidance; the
    tampered local artifact is removed so a later run does not trust it.
    """
    cache_root_path = Path(cache_root_path)
    dest = cache_root_path / 'base' / 'yolo11n-seg.pt'

    # Optional companion checksum sidecar: if present, it must agree with the constant.
    companion = _companion_uri()
    if gateway.exists(companion):
        declared = gateway.get_text(companion).strip().split()[0].lower()
        if declared != constants.BASE_CHECKPOINT_SHA256:
            raise IntegrityError(
                'companion checksum sidecar does not match the repository constant; ' + _BOOTSTRAP_GUIDANCE
            )

    base = _base_uri()
    if not gateway.exists(base):
        raise IntegrityError('base checkpoint object is missing; ' + _BOOTSTRAP_GUIDANCE)

    gateway.download(base, dest)

    actual_size = dest.stat().st_size
    if actual_size != constants.BASE_CHECKPOINT_SIZE:
        dest.unlink(missing_ok=True)
        raise IntegrityError(
            f'base checkpoint size {actual_size} != expected {constants.BASE_CHECKPOINT_SIZE}; ' + _BOOTSTRAP_GUIDANCE
        )

    actual_sha = sha256_file(dest)
    if actual_sha != constants.BASE_CHECKPOINT_SHA256:
        dest.unlink(missing_ok=True)
        raise IntegrityError('base checkpoint sha256 mismatch; ' + _BOOTSTRAP_GUIDANCE)

    return dest
