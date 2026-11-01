"""Shared pytest fixtures and import-path setup for the yolopipe test suite."""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import pytest

# Ensure the package is importable even without an editable install.
_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'manifests'


class ManifestOnlyFakeS3:
    """A fake boto3-like S3 client that serves manifest bodies only.

    Any ``get_object``/``download_file``/``head_object`` for a key that is not a
    registered manifest raises, proving that manifest validation never touches
    referenced image or mask objects.
    """

    def __init__(self, manifest_bodies: dict[str, str]) -> None:
        # Keys are 's3://bucket/key' strings mapping to manifest text.
        self._bodies = manifest_bodies
        self.accessed: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3 signature
        uri = f's3://{Bucket}/{Key}'
        self.accessed.append(uri)
        if uri not in self._bodies:
            raise AssertionError(f'manifest validation accessed a non-manifest object: {uri}')
        return {'Body': io.BytesIO(self._bodies[uri].encode('utf-8'))}

    def download_file(self, Bucket, Key, path):  # noqa: N803
        raise AssertionError(f'manifest validation must not download objects: s3://{Bucket}/{Key}')

    def head_object(self, Bucket, Key):  # noqa: N803
        raise AssertionError(f'manifest validation must not HeadObject: s3://{Bucket}/{Key}')

    def upload_file(self, *args, **kwargs):
        raise AssertionError('manifest validation must not upload objects')

    def list_objects_v2(self, *args, **kwargs):
        raise AssertionError('manifest validation must not list objects')


class ObjectServingFakeS3:
    """A fake boto3-like S3 client that serves arbitrary object bytes.

    Unlike :class:`ManifestOnlyFakeS3`, this fake exists to prove image/mask
    localization behavior: it records every accessed key and only serves bytes
    for keys that were explicitly registered. ``download_file`` writes the bytes
    to the requested path. A request for an unregistered key raises, so tests
    can assert exactly which objects were fetched (e.g. that masks for discarded
    candidates are never requested).
    """

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        # Keys are 's3://bucket/key' strings mapping to raw object bytes.
        self._objects: dict[str, bytes] = dict(objects or {})
        self.downloaded: list[str] = []
        self.accessed: list[str] = []

    def put(self, uri: str, data: bytes) -> None:
        self._objects[uri] = data

    def download_file(self, Bucket, Key, path):  # noqa: N803 - boto3 signature
        uri = f's3://{Bucket}/{Key}'
        self.accessed.append(uri)
        self.downloaded.append(uri)
        if uri not in self._objects:
            raise FileNotFoundError(f'no such object: {uri}')
        Path(path).write_bytes(self._objects[uri])

    def head_object(self, Bucket, Key):  # noqa: N803
        uri = f's3://{Bucket}/{Key}'
        self.accessed.append(uri)
        if uri not in self._objects:
            raise FileNotFoundError(f'no such object: {uri}')
        return {'ContentLength': len(self._objects[uri])}

    def get_object(self, Bucket, Key):  # noqa: N803
        uri = f's3://{Bucket}/{Key}'
        self.accessed.append(uri)
        if uri not in self._objects:
            raise FileNotFoundError(f'no such object: {uri}')
        return {'Body': io.BytesIO(self._objects[uri])}


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding='utf-8')


@pytest.fixture
def fake_s3_factory():
    return ManifestOnlyFakeS3


@pytest.fixture
def object_s3_factory():
    return ObjectServingFakeS3


class BytesFakeS3:
    """A fake boto3-like S3 client backed by an in-memory key -> bytes store.

    Serves ``download_file``/``get_object``/``head_object`` from registered
    object bytes, records ``upload_file`` writes (tracking order and
    concurrency), and answers ``list_objects_v2`` against both the stored and
    uploaded keys. Used by checkpoint and publication tests; it never performs
    real network or GPU work.
    """

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        # Keys are 's3://bucket/key' strings mapping to object bytes.
        self._objects: dict[str, bytes] = dict(objects or {})
        self.uploads: list[str] = []
        self.upload_order: list[str] = []
        self._max_concurrency = 0
        self._active = 0
        self._lock = threading.Lock()

    @staticmethod
    def _uri(bucket: str, key: str) -> str:
        return f's3://{bucket}/{key}'

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self._objects[self._uri(bucket, key)] = data

    # --- read side ----------------------------------------------------------
    def head_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        return {'ContentLength': len(self._objects[uri])}

    def get_object(self, Bucket, Key):  # noqa: N803
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        return {'Body': io.BytesIO(self._objects[uri])}

    def download_file(self, Bucket, Key, path):  # noqa: N803
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        Path(path).write_bytes(self._objects[uri])

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, **kwargs):  # noqa: N803
        wanted = self._uri(Bucket, Prefix)
        matches = [k for k in (*self._objects, *(self._uri(Bucket, u) for u in self.uploads)) if k.startswith(wanted)]
        matches = sorted(set(matches))[:MaxKeys]
        return {'KeyCount': len(matches), 'Contents': [{'Key': m} for m in matches]}

    # --- write side ---------------------------------------------------------
    def upload_file(self, path, Bucket, Key):  # noqa: N803
        with self._lock:
            self._active += 1
            self._max_concurrency = max(self._max_concurrency, self._active)
        time.sleep(0.005)  # widen the window so bounded concurrency is observable
        data = Path(path).read_bytes()
        with self._lock:
            self._objects[self._uri(Bucket, Key)] = data
            self.uploads.append(Key)
            self.upload_order.append(Key)
            self._active -= 1

    @property
    def max_observed_concurrency(self) -> int:
        return self._max_concurrency


@pytest.fixture
def bytes_s3_factory():
    return BytesFakeS3
