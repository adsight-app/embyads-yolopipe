"""Fake-S3 integration tests for idempotent, failure-safe dataset publication."""

from __future__ import annotations

import io
import threading

import cv2
import numpy as np
import pytest

from yolopipe import constants
from yolopipe.conversion import convert_all
from yolopipe.dataset import materialize_dataset
from yolopipe.errors import PublicationError, StorageBoundaryError
from yolopipe.dataset_publish import publish_dataset
from yolopipe.models import NormalizedSample
from yolopipe.split import assign_splits
from yolopipe.storage import S3Gateway

BILLBOARD_BGR = (0, 255, 0)


class ObjectStoreFakeS3:
    """An in-memory object-serving fake boto3-like S3 client.

    Stores object bytes per ``s3://bucket/key`` and records upload order so the
    completion-marker-last invariant is observable. Hard-rejects any write to a
    bucket other than emby-oss or a key outside yolopipe/.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self.upload_order: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _uri(bucket: str, key: str) -> str:
        return f's3://{bucket}/{key}'

    def upload_file(self, path, Bucket, Key):  # noqa: N803 - boto3 signature
        if Bucket != constants.EMBY_OSS_BUCKET:
            raise AssertionError(f'write to forbidden bucket: {Bucket}')
        if not Key.startswith(constants.EMBY_OSS_PREFIX):
            raise AssertionError(f'write outside yolopipe/: {Key}')
        with open(path, 'rb') as fh:
            data = fh.read()
        with self._lock:
            self._objects[self._uri(Bucket, Key)] = data
            self.upload_order.append(Key)

    def get_object(self, Bucket, Key):  # noqa: N803
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise KeyError(uri)
        return {'Body': io.BytesIO(self._objects[uri])}

    def head_object(self, Bucket, Key):  # noqa: N803
        uri = self._uri(Bucket, Key)
        if uri not in self._objects:
            raise KeyError(uri)
        return {'ContentLength': len(self._objects[uri])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):  # noqa: N803
        prefix = self._uri(Bucket, Prefix)
        matches = [k for k in self._objects if k.startswith(prefix)]
        return {'KeyCount': min(len(matches), MaxKeys), 'Contents': [{} for _ in matches[:MaxKeys]]}

    # Test helpers
    def keys(self) -> list[str]:
        base = f's3://{constants.EMBY_OSS_BUCKET}/'
        return sorted(k[len(base) :] for k in self._objects)


def _sample(name: str) -> NormalizedSample:
    return NormalizedSample(
        sample_id=f'id-{name}',
        job_name='20250515-gta5-billboards-01',
        source_image_uri=f's3://synthetic/img/{name}.png',
        source_mask_uri=f's3://synthetic/mask/{name}.png',
        source_filename=f'{name}.png',
        extension='.png',
        color_map={'#00ff00': 'billboard'},
    )


def _prepare_dataset(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    cache = tmp_path / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    image_paths, mask_paths = {}, {}
    for i, s in enumerate(samples):
        img = np.zeros((40, 50, 3), dtype=np.uint8)
        mask = np.zeros((40, 50, 3), dtype=np.uint8)
        if i % 2 == 0:
            mask[5:25, 5:35] = BILLBOARD_BGR
        ip = cache / f'{s.sample_id}.png'
        mp = cache / f'{s.sample_id}_mask.png'
        cv2.imwrite(str(ip), img)
        cv2.imwrite(str(mp), mask)
        image_paths[s.sample_id] = ip
        mask_paths[s.sample_id] = mp

    conv = convert_all(samples, image_paths=image_paths, mask_paths=mask_paths)
    splits = assign_splits(samples)
    root = tmp_path / 'data' / 'run-x' / 'dataset'
    mat = materialize_dataset(
        dataset_root=root,
        samples=samples,
        labels=conv.labels,
        splits=splits,
        cached_image_paths=image_paths,
        catalog_fingerprint='catfp',
        max_samples=None,
        selected_sample_ids=[s.sample_id for s in samples],
        dedup_stats={},
        conversion_stats=conv.stats,
    )
    return mat


def _publish(gateway, mat):
    return publish_dataset(
        gateway,
        dataset_id=mat.dataset_id,
        dataset_root=mat.dataset_root,
        file_checksums=mat.file_checksums,
        metadata={'catalog_fingerprint': 'catfp', 'dataset_id': mat.dataset_id},
    )


def test_publish_confined_to_datasets_prefix(tmp_path):
    mat = _prepare_dataset(tmp_path)
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    result = _publish(gateway, mat)

    assert result.reused is False
    expected_prefix = f'yolopipe/datasets/{mat.dataset_id}/'
    for key in fake.keys():
        assert key.startswith(expected_prefix)


def test_completion_marker_written_last(tmp_path):
    mat = _prepare_dataset(tmp_path)
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    _publish(gateway, mat)

    marker_key = f'yolopipe/datasets/{mat.dataset_id}/{constants.COMPLETION_MARKER}'
    assert fake.upload_order[-1] == marker_key
    assert marker_key not in fake.upload_order[:-1]


def test_second_publish_reuses_without_upload(tmp_path):
    mat = _prepare_dataset(tmp_path)
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    _publish(gateway, mat)
    uploads_after_first = list(fake.upload_order)

    result = _publish(gateway, mat)
    assert result.reused is True
    assert fake.upload_order == uploads_after_first  # no new uploads


def test_incomplete_prefix_fails_without_deletion(tmp_path):
    mat = _prepare_dataset(tmp_path)
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    _publish(gateway, mat)

    # Simulate an incomplete prefix: drop the completion marker but keep objects.
    marker = f's3://{constants.EMBY_OSS_BUCKET}/yolopipe/datasets/{mat.dataset_id}/{constants.COMPLETION_MARKER}'
    del fake._objects[marker]
    keys_before = set(fake.keys())

    with pytest.raises(PublicationError, match='incomplete'):
        _publish(gateway, mat)
    assert set(fake.keys()) == keys_before  # nothing deleted


def test_inconsistent_finalized_prefix_fails(tmp_path):
    mat = _prepare_dataset(tmp_path)
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    _publish(gateway, mat)

    # Corrupt the remote checksum manifest so stored checksums no longer match.
    manifest_key = f's3://{constants.EMBY_OSS_BUCKET}/yolopipe/datasets/{mat.dataset_id}/checksums.json'
    fake._objects[manifest_key] = b'{"images/x.png":"deadbeef"}\n'
    keys_before = set(fake.keys())

    with pytest.raises(PublicationError, match='do not match'):
        _publish(gateway, mat)
    assert set(fake.keys()) == keys_before


def test_gateway_rejects_write_outside_boundary(tmp_path):
    fake = ObjectStoreFakeS3()
    gateway = S3Gateway(fake)
    local = tmp_path / 'f.txt'
    local.write_text('x')
    with pytest.raises(StorageBoundaryError):
        gateway.upload(local, 'not-yolopipe/evil.txt')
