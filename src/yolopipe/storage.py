"""S3 read/write boundary and atomic local transfer primitives.

The write gateway constructs destination URIs only from typed identifiers and
hard-rejects any bucket other than emby-oss or any key outside yolopipe/.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yolopipe import constants
from yolopipe.errors import StorageBoundaryError, TransferError
from yolopipe.models import S3Uri, sha256_file


class S3Gateway:
    """Thin boundary over an injected boto3-like S3 client.

    The client must provide ``download_file``, ``upload_file``, ``head_object``,
    ``get_object``, and ``list_objects_v2``. Tests inject a fake.
    """

    def __init__(self, client) -> None:
        self._client = client

    # --- Read side (any bucket permitted) ----------------------------------
    def download(self, uri: S3Uri, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix='.part')
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._client.download_file(uri.bucket, uri.key, str(tmp))
            os.replace(tmp, dest)
        except Exception as exc:  # noqa: BLE001 - re-raised as stage error
            tmp.unlink(missing_ok=True)
            raise TransferError(f'failed to download {uri}: {type(exc).__name__}') from exc
        return dest

    def get_text(self, uri: S3Uri) -> str:
        try:
            resp = self._client.get_object(Bucket=uri.bucket, Key=uri.key)
            body = resp['Body'].read()
        except Exception as exc:  # noqa: BLE001
            raise TransferError(f'failed to read {uri}: {type(exc).__name__}') from exc
        return body.decode('utf-8')

    def exists(self, uri: S3Uri) -> bool:
        try:
            self._client.head_object(Bucket=uri.bucket, Key=uri.key)
            return True
        except Exception:  # noqa: BLE001 - absence or access error treated as not-present
            return False

    def prefix_has_objects(self, bucket: str, prefix: str) -> bool:
        resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return resp.get('KeyCount', 0) > 0 or bool(resp.get('Contents'))

    # --- Write side (confined to emby-oss/yolopipe) -------------------------
    @staticmethod
    def _guard_write_key(key: str) -> None:
        if not key.startswith(constants.EMBY_OSS_PREFIX):
            raise StorageBoundaryError(f'refusing to write outside {constants.EMBY_OSS_PREFIX}: {key!r}')
        if '..' in key.split('/'):
            raise StorageBoundaryError(f'refusing to write traversal key: {key!r}')

    def upload(self, local: Path, key: str, bucket: str = constants.EMBY_OSS_BUCKET) -> S3Uri:
        if bucket != constants.EMBY_OSS_BUCKET:
            raise StorageBoundaryError(f'refusing to write to bucket {bucket!r}')
        self._guard_write_key(key)
        try:
            self._client.upload_file(str(local), bucket, key)
        except Exception as exc:  # noqa: BLE001
            raise TransferError(f'failed to upload s3://{bucket}/{key}: {type(exc).__name__}') from exc
        return S3Uri(bucket=bucket, key=key)

    def upload_many(self, items: list[tuple[Path, str]]) -> list[S3Uri]:
        """Upload (local, key) pairs with a bounded worker pool. Order preserved in result."""
        for _, key in items:
            self._guard_write_key(key)
        results: list[S3Uri | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=constants.TRANSFER_WORKERS) as pool:
            futures = {pool.submit(self.upload, local, key): i for i, (local, key) in enumerate(items)}
            for fut in futures:
                idx = futures[fut]
                results[idx] = fut.result()
        return [r for r in results if r is not None]

    def verify_checksum(self, local: Path, expected_sha256: str) -> None:
        actual = sha256_file(local)
        if actual != expected_sha256:
            raise TransferError('checksum mismatch after transfer')
