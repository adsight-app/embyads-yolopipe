"""Portable prepared-dataset materialization, serialization, and fingerprinting.

The prepared dataset is materialized under ``./data/yolopipe/<run-id>/dataset/``
as parallel ``images/`` and ``labels/`` trees so Ultralytics resolves labels
from image paths automatically. Images are validated hardlinks to the cached
source objects (``os.link``) with a copy fallback (``shutil.copy2``) when the
filesystem refuses the link. Labels are owned files. Split manifests and the
checksummed ``data.yaml`` are persisted with dataset-root-relative POSIX paths
and contain neither ``run_id`` nor host paths, so one ``dataset_id`` is
byte-equivalent across runs and operating systems. A separate runtime renderer
produces a ``data.yaml`` with a native absolute ``path`` without mutating the
persisted, checksummed file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import constants
from .conversion import SampleConversion
from .errors import ConversionError
from .models import NormalizedSample, canonical_json, sha256_file, sha256_hex

# Exact persisted data.yaml bytes. Serialized deterministically with LF newlines.
_PERSISTED_DATA_YAML = (
    "path: '.'\n"
    'train: train.txt\n'
    'val: val.txt\n'
    'minival: minival.txt\n'
    'names:\n'
    f'  {constants.CLASS_ID}: {constants.CLASS_NAME}\n'
)


@dataclass
class MaterializedDataset:
    """Result of materializing a prepared dataset on the local filesystem."""

    dataset_root: Path
    dataset_id: str
    splits: dict[str, list[str]]
    image_rel_paths: dict[str, str]
    label_rel_paths: dict[str, str]
    file_checksums: dict[str, str]


def persisted_data_yaml_bytes() -> bytes:
    """Return the exact, run-agnostic persisted ``data.yaml`` bytes (LF newlines)."""
    return _PERSISTED_DATA_YAML.encode('utf-8')


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` with a copy fallback on OSError. Atomic-ish."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _split_file_text(sample_ids: list[str], image_rel_paths: dict[str, str]) -> str:
    """Ordered ``./images/<sample_id><ext>`` POSIX lines with one trailing newline."""
    lines = [f'./{image_rel_paths[sid]}' for sid in sample_ids]
    return '\n'.join(lines) + '\n' if lines else ''


def compute_dataset_id(
    *,
    catalog_fingerprint: str,
    max_samples: int | None,
    selected_sample_ids: list[str],
    dedup_stats: dict,
    survivor_ids: list[str],
    conversion_stats: dict,
    split_seed: str,
    split_assignments: dict[str, list[str]],
    class_names: dict,
    image_rel_paths: dict[str, str],
    label_rel_paths: dict[str, str],
    file_checksums: dict[str, str],
) -> str:
    """Content-address the prepared dataset, excluding run/host/time/native paths."""
    metadata = {
        'catalog_fingerprint': catalog_fingerprint,
        'max_samples': max_samples,
        'selected_sample_ids': list(selected_sample_ids),
        'dedup': {
            'algorithm_version': constants.DEDUP_ALGORITHM_VERSION,
            'hash_size': constants.DEDUP_HASH_SIZE,
            'threshold': constants.DEDUP_THRESHOLD,
            'resize': constants.DEDUP_RESIZE,
            'survivor_ids': list(survivor_ids),
        },
        'conversion': {
            'algorithm_version': constants.CONVERSION_ALGORITHM_VERSION,
            'min_contour_area': constants.MIN_CONTOUR_AREA,
            'epsilon_factor': constants.EPSILON_FACTOR,
            'coord_precision': constants.COORD_PRECISION,
            'min_contour_points': constants.MIN_CONTOUR_POINTS,
        },
        'split': {
            'seed': split_seed,
            'algorithm_version': constants.SPLIT_ALGORITHM_VERSION,
            'ratios': constants.SPLIT_RATIOS,
            'assignments': {name: list(ids) for name, ids in split_assignments.items()},
        },
        'class_names': {str(k): v for k, v in class_names.items()},
        'image_rel_paths': dict(image_rel_paths),
        'label_rel_paths': dict(label_rel_paths),
        'file_checksums': dict(file_checksums),
    }
    return sha256_hex(canonical_json(metadata))


def materialize_dataset(
    *,
    dataset_root: Path,
    samples: list[NormalizedSample],
    labels: dict[str, SampleConversion],
    splits: dict[str, list[str]],
    cached_image_paths: dict[str, Path],
    catalog_fingerprint: str,
    max_samples: int | None,
    selected_sample_ids: list[str],
    dedup_stats: dict,
    conversion_stats: dict,
    class_names: dict | None = None,
) -> MaterializedDataset:
    """Materialize images/labels/splits/data.yaml and compute the dataset id.

    ``samples`` are the ordered survivors; ``labels`` maps sample_id to its
    canonical conversion; ``cached_image_paths`` maps sample_id to the cached
    source image file.
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    survivor_ids = [s.sample_id for s in samples]

    image_rel_paths: dict[str, str] = {}
    label_rel_paths: dict[str, str] = {}

    for sample in samples:
        sid = sample.sample_id
        image_rel_paths[sid] = sample.image_dataset_path
        label_rel_paths[sid] = sample.label_dataset_path

        cached = cached_image_paths.get(sid)
        if cached is None:
            raise ConversionError(f'missing cached image for survivor {sid}')
        _link_or_copy(Path(cached), dataset_root / sample.image_dataset_path)

        conversion = labels.get(sid)
        if conversion is None:
            raise ConversionError(f'missing converted label for survivor {sid}')
        (dataset_root / sample.label_dataset_path).write_bytes(conversion.label_bytes)

    # Ordered split manifest files with POSIX dataset-relative image paths.
    for name in constants.SPLIT_ORDER:
        text = _split_file_text(splits.get(name, []), image_rel_paths)
        (dataset_root / f'{name}.txt').write_text(text, encoding='utf-8', newline='\n')

    # Persisted, checksummed, run-agnostic data.yaml.
    (dataset_root / 'data.yaml').write_bytes(persisted_data_yaml_bytes())

    names = class_names or {constants.CLASS_ID: constants.CLASS_NAME}

    # Per-file checksums over the persisted, portable artifacts.
    file_checksums: dict[str, str] = {}
    for sample in samples:
        file_checksums[sample.image_dataset_path] = sha256_file(dataset_root / sample.image_dataset_path)
        file_checksums[sample.label_dataset_path] = sha256_file(dataset_root / sample.label_dataset_path)
    for name in constants.SPLIT_ORDER:
        file_checksums[f'{name}.txt'] = sha256_file(dataset_root / f'{name}.txt')
    file_checksums['data.yaml'] = sha256_file(dataset_root / 'data.yaml')

    dataset_id = compute_dataset_id(
        catalog_fingerprint=catalog_fingerprint,
        max_samples=max_samples,
        selected_sample_ids=selected_sample_ids,
        dedup_stats=dedup_stats,
        survivor_ids=survivor_ids,
        conversion_stats=conversion_stats,
        split_seed=constants.SPLIT_SEED,
        split_assignments=splits,
        class_names=names,
        image_rel_paths=image_rel_paths,
        label_rel_paths=label_rel_paths,
        file_checksums=file_checksums,
    )

    return MaterializedDataset(
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        splits={name: list(splits.get(name, [])) for name in constants.SPLIT_ORDER},
        image_rel_paths=image_rel_paths,
        label_rel_paths=label_rel_paths,
        file_checksums=file_checksums,
    )


def render_runtime_data_yaml(run_dir: Path, *, write: bool = False) -> str:
    """Render a runtime ``data.yaml`` with a native absolute ``path``.

    ``run_dir`` is the dataset root. The persisted ``data.yaml`` is NOT mutated;
    when ``write`` is true the runtime file is written to ``data.runtime.yaml``.
    """
    dataset_root = Path(run_dir)
    native_path = str(dataset_root.resolve())
    runtime = (
        f'path: {native_path}\n'
        'train: train.txt\n'
        'val: val.txt\n'
        'minival: minival.txt\n'
        'names:\n'
        f'  {constants.CLASS_ID}: {constants.CLASS_NAME}\n'
    )
    if write:
        (dataset_root / 'data.runtime.yaml').write_text(runtime, encoding='utf-8', newline='\n')
    return runtime
