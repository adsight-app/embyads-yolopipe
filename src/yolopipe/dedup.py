"""Per-manifest perceptual deduplication of selected samples.

For each manifest's selected rows (in original row order), every image is
hashed with ``imagehash.whash(img, hash_size=16)`` and the hexadecimal hash is
converted to an integer via ``int(str(img_hash), 16)``. Hashes are computed
with a bounded thread pool while results are collected back in original row
order, then stable-sorted ascending by the integer. Zero- or one-row groups
pass through unchanged. Otherwise the first row is retained; each later row is
compared only to the LAST retained image using
:func:`yolopipe.image_matcher.match_images`, and is discarded when the score is
at least the threshold or kept as the new anchor otherwise.

Deduplication is strictly per manifest: images are never compared across jobs.
Survivors remain in wavelet-hash order within a manifest, and manifests are
concatenated in fixed catalog order. Input records are never mutated; survivors
are returned as :func:`dataclasses.replace` copies carrying ``image_hash_int``.
"""

from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import imagehash

from . import constants
from .catalog import CATALOG
from .image_matcher import match_images
from .models import NormalizedSample

_HASH_WORKERS = 8


def _compute_hash_int(path: Path) -> int:
    """Open an image and return its wavelet hash as an integer."""
    with Image.open(path) as img:
        img_hash = imagehash.whash(img, hash_size=constants.DEDUP_HASH_SIZE)
    return int(str(img_hash), 16)


def _hashes_in_row_order(samples: list[NormalizedSample], image_paths: dict[str, Path]) -> list[int]:
    """Compute wavelet-hash integers with a bounded pool, keeping input row order."""
    results: list[int | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=_HASH_WORKERS) as pool:
        futures = {pool.submit(_compute_hash_int, image_paths[s.sample_id]): i for i, s in enumerate(samples)}
        for fut in futures:
            results[futures[fut]] = fut.result()
    return [h for h in results if h is not None]


def _dedup_manifest(
    samples: list[NormalizedSample],
    image_paths: dict[str, Path],
) -> tuple[list[NormalizedSample], dict]:
    """Deduplicate one manifest's selected rows. Returns (survivors, stats)."""
    analyzed = len(samples)
    hashes = _hashes_in_row_order(samples, image_paths)

    # Stable ascending sort by hash integer; ties keep original row order.
    order = sorted(range(len(samples)), key=lambda i: hashes[i])
    ordered = [(samples[i], hashes[i]) for i in order]

    survivors: list[NormalizedSample] = []
    if len(ordered) <= 1:
        kept = ordered
    else:
        kept = [ordered[0]]
        for sample, hash_int in ordered[1:]:
            anchor_sample, _ = kept[-1]
            anchor_path = image_paths[anchor_sample.sample_id]
            candidate_path = image_paths[sample.sample_id]
            if match_images(anchor_path, candidate_path, threshold=constants.DEDUP_THRESHOLD):
                continue  # perceptual duplicate of the last retained image
            kept.append((sample, hash_int))

    for sample, hash_int in kept:
        survivors.append(dataclasses.replace(sample, image_hash_int=hash_int))

    stats = {
        'analyzed': analyzed,
        'retained': len(survivors),
        'discarded': analyzed - len(survivors),
        'survivor_sample_ids': [s.sample_id for s in survivors],
    }
    return survivors, stats


def deduplicate(
    selected_by_manifest: dict[str, list[NormalizedSample]],
    image_paths: dict[str, Path],
) -> tuple[list[NormalizedSample], dict]:
    """Deduplicate every manifest independently and return survivors + metadata.

    ``selected_by_manifest`` maps ``job_name`` to that manifest's selected
    samples in original row order. ``image_paths`` maps ``sample_id`` to the
    localized image path. Survivors are concatenated in fixed catalog order;
    jobs are never compared against one another.
    """
    started = time.perf_counter()

    survivors: list[NormalizedSample] = []
    per_manifest: dict[str, dict] = {}

    for entry in CATALOG:
        samples = selected_by_manifest.get(entry.job_name)
        if not samples:
            continue
        job_survivors, stats = _dedup_manifest(samples, image_paths)
        survivors.extend(job_survivors)
        per_manifest[entry.job_name] = stats

    duration = time.perf_counter() - started

    metadata = {
        'algorithm_version': constants.DEDUP_ALGORITHM_VERSION,
        'hash': 'whash',
        'hash_size': constants.DEDUP_HASH_SIZE,
        'matcher': 'TM_CCOEFF_NORMED',
        'threshold': constants.DEDUP_THRESHOLD,
        'resize': constants.DEDUP_RESIZE,
        'per_manifest': per_manifest,
        'duration_seconds': duration,
        'survivor_sample_ids': [s.sample_id for s in survivors],
    }
    return survivors, metadata
