"""Deterministic, file-only visualization of prepared samples (no GUI, no S3)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import ValidationError
from .paths import validate_output_dir

_MAX_SAMPLES = 16


def _read_prepared_sample_ids(dataset_root: Path) -> list[str]:
    """All prepared sample ids across every split, from persisted split manifests."""
    ids: set[str] = set()
    for name in ('train', 'val', 'minival'):
        split_file = dataset_root / f'{name}.txt'
        if not split_file.is_file():
            continue
        for line in split_file.read_text(encoding='utf-8').splitlines():
            stem = line.strip().rsplit('/', 1)[-1]
            if stem:
                ids.add(stem.rsplit('.', 1)[0])
    return sorted(ids)


def _overlay(image_path: Path, label_path: Path) -> np.ndarray:
    """Compose a source image with its YOLO polygon overlay."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError(f'cannot read image for visualization: {image_path}')
    height, width = image.shape[:2]
    canvas = image.copy()
    if label_path.is_file():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) < 7:  # class + >= 3 points
                continue
            coords = [float(v) for v in parts[1:]]
            pts = np.array(
                [(int(coords[i] * width), int(coords[i + 1] * height)) for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return canvas


def visualize(dataset_root: Path, output: Path, cwd: Path | None = None) -> dict:
    """Render up to 16 prepared samples as ``NN_<sample_id>.png`` composites.

    Requires a prepared dataset at ``dataset_root``. The output directory is
    validated (never overlapping managed data/cache trees) and created if absent.
    Never opens a GUI or writes to S3.
    """
    dataset_root = Path(dataset_root)
    if not (dataset_root / 'data.yaml').is_file():
        raise ValidationError(f'no prepared dataset at {dataset_root}; run `yolopipe prepare --run-id <id>` first')

    out_dir = validate_output_dir(Path(output), cwd)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = _read_prepared_sample_ids(dataset_root)[:_MAX_SAMPLES]
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'

    written = 0
    for index, sample_id in enumerate(sample_ids):
        matches = list(images_dir.glob(f'{sample_id}.*'))
        if not matches:
            continue
        composite = _overlay(matches[0], labels_dir / f'{sample_id}.txt')
        dest = out_dir / f'{index:02d}_{sample_id}.png'
        cv2.imwrite(str(dest), composite)
        written += 1

    return {'output_dir': str(out_dir), 'count': written}
