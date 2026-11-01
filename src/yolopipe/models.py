"""Frozen value objects and canonical serialization helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import urlparse

from yolopipe.errors import ValidationError


@dataclass(frozen=True)
class S3Uri:
    """An ``s3://bucket/key`` reference. Key case is preserved; no query/fragment allowed."""

    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str) -> S3Uri:
        if not isinstance(uri, str) or not uri:
            raise ValidationError('S3 URI must be a non-empty string')
        parsed = urlparse(uri)
        if parsed.scheme != 's3':
            raise ValidationError(f'S3 URI must use the s3 scheme: {uri!r}')
        if parsed.query or parsed.fragment:
            raise ValidationError(f'S3 URI must not contain a query or fragment: {uri!r}')
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        if not bucket or not key:
            raise ValidationError(f'S3 URI must name a bucket and key: {uri!r}')
        return cls(bucket=bucket, key=key)

    @property
    def filename(self) -> str:
        segment = self.key.rstrip('/').rsplit('/', 1)[-1]
        if not segment:
            raise ValidationError(f'S3 URI has no object filename: {self}')
        return segment

    def __str__(self) -> str:
        return f's3://{self.bucket}/{self.key}'


@dataclass(frozen=True)
class CatalogEntry:
    """One fixed Ground Truth job: ``job_name`` is also the label-attribute prefix."""

    job_name: str
    output_manifest_uri: str


@dataclass(frozen=True)
class NormalizedSample:
    """A validated, annotated Ground Truth row. Persisted paths are POSIX-relative."""

    sample_id: str
    job_name: str
    source_image_uri: str
    source_mask_uri: str
    source_filename: str
    extension: str
    color_map: dict
    group_id: str | None = None
    image_hash_int: int | None = None

    @property
    def image_dataset_path(self) -> str:
        return f'images/{self.sample_id}{self.extension}'

    @property
    def label_dataset_path(self) -> str:
        return f'labels/{self.sample_id}.txt'


@dataclass
class PreparedDataset:
    """Result of preparation: survivors, splits, and reproducibility metadata."""

    dataset_id: str
    catalog_fingerprint: str
    max_samples: int | None
    selected_sample_ids: list[str]
    samples: list[NormalizedSample]
    splits: dict[str, list[str]]
    dedup_stats: dict
    conversion_stats: dict
    class_names: dict = field(default_factory=lambda: {0: 'gta5-billboard'})


@dataclass
class RunRecord:
    """Training run / model record. ``model_id`` always equals ``run_id``."""

    run_id: str
    dataset_id: str
    max_samples: int | None
    epochs: int
    mode: str  # 'production' | 'smoke'
    effective_args: dict
    environment: dict

    @property
    def model_id(self) -> str:
        return self.run_id


def canonical_json(obj: object) -> bytes:
    """UTF-8 canonical JSON: sorted keys, compact separators, one trailing newline."""
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return (text + '\n').encode('utf-8')


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
