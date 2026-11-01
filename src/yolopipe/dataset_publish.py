"""Idempotent, failure-safe publication of a prepared dataset to S3.

All canonical dataset files are published beneath
``s3://emby-oss/yolopipe/datasets/<dataset-id>/`` using the storage gateway's
bounded (8-worker) pool. A checksum manifest and dataset metadata are written,
and the completion marker (``_COMPLETE``) is written LAST so partial attempts
are always distinguishable from finalized immutable outputs. A finalized prefix
whose marker is present and whose published checksums match the local checksums
is verified and reused WITHOUT upload. An incomplete prefix (objects present but
no marker) or an inconsistent one (marker present but checksums differ) fails
with :class:`PublicationError` and is never deleted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import constants
from .errors import PublicationError
from .models import S3Uri, canonical_json
from .storage import S3Gateway

_CHECKSUM_MANIFEST_NAME = 'checksums.json'
_METADATA_NAME = 'dataset.json'


@dataclass
class PublicationResult:
    """Outcome of a publish: ``reused`` true means no upload occurred."""

    dataset_id: str
    prefix: str
    reused: bool
    uploaded: list[str]


def _dataset_prefix(dataset_id: str) -> str:
    return f'{constants.DATASETS_PREFIX}{dataset_id}/'


def _checksum_manifest_bytes(file_checksums: dict[str, str]) -> bytes:
    return canonical_json({k: file_checksums[k] for k in sorted(file_checksums)})


def _read_remote_checksums(gateway: S3Gateway, prefix: str) -> dict[str, str]:
    uri = S3Uri(bucket=constants.EMBY_OSS_BUCKET, key=f'{prefix}{_CHECKSUM_MANIFEST_NAME}')
    text = gateway.get_text(uri)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise PublicationError(f'remote checksum manifest is not an object: {uri}')
    return {str(k): str(v) for k, v in data.items()}


def publish_dataset(
    gateway: S3Gateway,
    *,
    dataset_id: str,
    dataset_root: Path,
    file_checksums: dict[str, str],
    metadata: dict,
) -> PublicationResult:
    """Publish or reuse the content-addressed dataset prefix.

    ``file_checksums`` maps dataset-root-relative POSIX paths to SHA-256 hex
    digests of the persisted canonical files. ``metadata`` is run-agnostic
    dataset metadata serialized as canonical JSON.
    """
    dataset_root = Path(dataset_root)
    prefix = _dataset_prefix(dataset_id)
    marker_uri = S3Uri(bucket=constants.EMBY_OSS_BUCKET, key=f'{prefix}{constants.COMPLETION_MARKER}')

    checksum_bytes = _checksum_manifest_bytes(file_checksums)
    metadata_bytes = canonical_json(metadata)
    expected = {str(k): str(v) for k, v in file_checksums.items()}

    has_objects = gateway.prefix_has_objects(constants.EMBY_OSS_BUCKET, prefix)
    marker_present = gateway.exists(marker_uri)

    if has_objects:
        if not marker_present:
            raise PublicationError(
                f'dataset prefix {prefix} is incomplete (no completion marker); remove it manually before re-publishing'
            )
        # Finalized: verify published checksums match the local dataset before reuse.
        remote = _read_remote_checksums(gateway, prefix)
        if remote != expected:
            raise PublicationError(
                f'dataset prefix {prefix} is finalized but its checksums do not match '
                f'the locally prepared dataset; refusing to overwrite'
            )
        return PublicationResult(dataset_id=dataset_id, prefix=prefix, reused=True, uploaded=[])

    # Fresh publish. Write all data files, then manifest + metadata, marker LAST.
    manifest_path = dataset_root / _CHECKSUM_MANIFEST_NAME
    metadata_path = dataset_root / _METADATA_NAME
    manifest_path.write_bytes(checksum_bytes)
    metadata_path.write_bytes(metadata_bytes)

    items: list[tuple[Path, str]] = []
    for rel in sorted(file_checksums):
        items.append((dataset_root / rel, f'{prefix}{rel}'))
    items.append((manifest_path, f'{prefix}{_CHECKSUM_MANIFEST_NAME}'))
    items.append((metadata_path, f'{prefix}{_METADATA_NAME}'))

    gateway.upload_many(items)

    # Completion marker written strictly last.
    marker_path = dataset_root / constants.COMPLETION_MARKER
    marker_path.write_bytes(b'')
    gateway.upload(marker_path, marker_uri.key)

    uploaded = [key for _, key in items] + [marker_uri.key]
    return PublicationResult(dataset_id=dataset_id, prefix=prefix, reused=False, uploaded=uploaded)
