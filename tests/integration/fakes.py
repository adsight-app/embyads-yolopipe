"""A combined read/write in-memory fake S3 for end-to-end pipeline tests."""

from __future__ import annotations

import io
import threading
from pathlib import Path


class PipelineFakeS3:
    """Serves registered object bytes and records all reads/writes.

    Supports the full S3Gateway surface: get_object, download_file, head_object,
    list_objects_v2, upload_file. Writes are confined by the gateway, not here.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self.accessed: list[str] = []
        self.downloaded: list[str] = []
        self.uploads: list[str] = []
        self.upload_order: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _uri(bucket: str, key: str) -> str:
        return f's3://{bucket}/{key}'

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self._objects[self._uri(bucket, key)] = data

    def put_file(self, bucket: str, key: str, path: Path) -> None:
        self._objects[self._uri(bucket, key)] = Path(path).read_bytes()

    def get_object(self, Bucket, Key):  # noqa: N803
        uri = self._uri(Bucket, Key)
        self.accessed.append(uri)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        return {'Body': io.BytesIO(self._objects[uri])}

    def download_file(self, Bucket, Key, path):  # noqa: N803
        uri = self._uri(Bucket, Key)
        self.accessed.append(uri)
        self.downloaded.append(uri)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        Path(path).write_bytes(self._objects[uri])

    def head_object(self, Bucket, Key):  # noqa: N803
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise FileNotFoundError(uri)
        return {'ContentLength': len(self._objects[uri])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, **kwargs):  # noqa: N803
        wanted = self._uri(Bucket, Prefix)
        keys = {*self._objects, *(self._uri(Bucket, u) for u in self.uploads)}
        matches = sorted(k for k in keys if k.startswith(wanted))[:MaxKeys]
        return {'KeyCount': len(matches), 'Contents': [{'Key': m} for m in matches]}

    def upload_file(self, path, Bucket, Key):  # noqa: N803
        with self._lock:
            self._objects[self._uri(Bucket, Key)] = Path(path).read_bytes()
            self.uploads.append(Key)
            self.upload_order.append(Key)

    def image_mask_fetches(self) -> list[str]:
        return [u for u in self.downloaded if '/img/' in u or '/mask/' in u]
