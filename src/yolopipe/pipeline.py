"""Application service composing validation, preparation, training, and publication.

The pipeline wires the frozen stage modules together and owns the reuse and
collision policy: local prepared datasets are reused only when their recorded
invocation parameters match, finalized remote datasets are reused without
re-upload, and any existing run/model prefix blocks work before downloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import constants, preflight
from .catalog import catalog_fingerprint
from .checkpoint import ensure_base_checkpoint
from .conversion import convert_all
from .dataset import materialize_dataset, render_runtime_data_yaml
from .dataset_publish import publish_dataset
from .dedup import deduplicate
from .errors import ValidationError
from .localize import localize_images, localize_masks
from .manifest import build_manifest_plan
from .models import RunRecord
from .paths import cache_root, data_run_dir, require_working_roots, validate_run_id
from .run_publish import assert_run_absent, publish_model, publish_run
from .selection import select
from .split import assign_splits
from .storage import S3Gateway
from .trainer import PRODUCTION, SINGLE_GPU, SMOKE, train

_PREP_META = 'prepare.json'


@dataclass
class PrepareOutcome:
    dataset_id: str
    dataset_root: Path
    reused: bool
    selected: int
    survivors: int
    splits: dict


def _selected_by_manifest(plan) -> dict:
    selected = set(plan.selected_sample_ids)
    out: dict[str, list] = {}
    for job_name, samples in plan.candidates_by_manifest.items():
        kept = [s for s in samples if s.sample_id in selected]
        if kept:
            out[job_name] = kept
    return out


def _prep_metadata_path(run_dir: Path) -> Path:
    return run_dir / _PREP_META


def _load_local_prep(run_dir: Path) -> dict | None:
    meta = _prep_metadata_path(run_dir)
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None


def _invocation_signature(*, max_samples: int | None) -> dict:
    return {
        'catalog_fingerprint': catalog_fingerprint(),
        'max_samples': max_samples,
        'selection_algorithm': constants.SELECTION_ALGORITHM_VERSION,
        'split_algorithm': constants.SPLIT_ALGORITHM_VERSION,
        'conversion_algorithm': constants.CONVERSION_ALGORITHM_VERSION,
        'dedup_algorithm': constants.DEDUP_ALGORITHM_VERSION,
    }


def prepare(
    gateway: S3Gateway,
    *,
    run_id: str,
    max_samples: int | None,
    cwd: Path | None = None,
    check_run_collision: bool = True,
) -> PrepareOutcome:
    """Ingest, validate, select, deduplicate, convert, split, materialize, publish."""
    validate_run_id(run_id)
    require_working_roots(cwd)

    run_dir = data_run_dir(run_id, cwd)
    dataset_root = run_dir / 'dataset'
    signature = _invocation_signature(max_samples=max_samples)

    if check_run_collision:
        assert_run_absent(gateway, run_id)

    # Local reuse: identical recorded signature -> reuse without manifest download.
    existing = _load_local_prep(run_dir)
    if existing is not None:
        if existing.get('signature') != signature:
            raise ValidationError(
                f'local prepared dataset for run-id {run_id!r} was built with different '
                f'parameters; choose a new run-id or repeat the matching invocation'
            )
        return PrepareOutcome(
            dataset_id=existing['dataset_id'],
            dataset_root=dataset_root,
            reused=True,
            selected=existing['selected'],
            survivors=existing['survivors'],
            splits=existing['splits'],
        )

    # Fresh preparation.
    plan = build_manifest_plan(gateway)
    select(plan, max_samples)
    selected_by_manifest = _selected_by_manifest(plan)

    selected_samples = [s for job in selected_by_manifest.values() for s in job]
    image_paths = localize_images(gateway, selected_samples, cwd)

    survivors, dedup_meta = deduplicate(selected_by_manifest, image_paths)
    mask_paths = localize_masks(gateway, survivors, cwd)

    conversion = convert_all(
        survivors,
        image_paths={s.sample_id: image_paths[s.sample_id] for s in survivors},
        mask_paths=mask_paths,
    )
    splits = assign_splits(survivors)

    materialized = materialize_dataset(
        dataset_root=dataset_root,
        samples=survivors,
        labels=conversion.labels,
        splits=splits,
        cached_image_paths=image_paths,
        catalog_fingerprint=plan.catalog_fingerprint,
        max_samples=max_samples,
        selected_sample_ids=plan.selected_sample_ids,
        dedup_stats=dedup_meta,
        conversion_stats=conversion.stats,
    )

    publish_dataset(
        gateway,
        dataset_id=materialized.dataset_id,
        dataset_root=materialized.dataset_root,
        file_checksums=materialized.file_checksums,
        metadata={
            'catalog_fingerprint': plan.catalog_fingerprint,
            'dataset_id': materialized.dataset_id,
            'dedup': dedup_meta,
            'conversion_stats': conversion.stats,
            'splits': materialized.splits,
        },
    )

    outcome = {
        'signature': signature,
        'dataset_id': materialized.dataset_id,
        'selected': len(plan.selected_sample_ids),
        'survivors': len(survivors),
        'splits': materialized.splits,
    }
    _prep_metadata_path(run_dir).write_text(json.dumps(outcome, sort_keys=True), encoding='utf-8')

    return PrepareOutcome(
        dataset_id=materialized.dataset_id,
        dataset_root=dataset_root,
        reused=False,
        selected=len(plan.selected_sample_ids),
        survivors=len(survivors),
        splits=materialized.splits,
    )


def run_training(
    gateway: S3Gateway,
    *,
    run_id: str,
    max_samples: int | None,
    epochs: int,
    single_gpu: bool = False,
    cwd: Path | None = None,
) -> dict:
    """Preflight, (reuse or prepare) dataset, train, and publish run + model."""
    validate_run_id(run_id)
    require_working_roots(cwd)

    if single_gpu and max_samples is not None:
        raise ValidationError('--single-gpu runs the full dataset and is incompatible with --max-samples (smoke mode)')

    # Collision check before any expensive work.
    assert_run_absent(gateway, run_id)

    # Mode + platform preflight. Smoke mode requires an explicit sample bound;
    # single-GPU mode runs the full dataset on one device without DDP.
    if max_samples is not None:
        mode = SMOKE
        preflight.require_smoke()
    elif single_gpu:
        mode = SINGLE_GPU
        preflight.require_single_gpu()
    else:
        mode = PRODUCTION
        preflight.require_production()

    outcome = prepare(gateway, run_id=run_id, max_samples=max_samples, cwd=cwd, check_run_collision=False)

    base_ckpt = ensure_base_checkpoint(gateway, cache_root(cwd))
    render_runtime_data_yaml(outcome.dataset_root, write=True)
    runtime_yaml = outcome.dataset_root / 'data.runtime.yaml'

    record = RunRecord(
        run_id=run_id,
        dataset_id=outcome.dataset_id,
        max_samples=max_samples,
        epochs=epochs,
        mode=mode,
        effective_args={},
        environment={},
    )

    runs_dir = data_run_dir(run_id, cwd) / 'runs'
    result = train(
        record,
        runtime_yaml,
        base_ckpt,
        mode,
        cache_root_path=cache_root(cwd),
        runs_dir=runs_dir,
    )

    staging = data_run_dir(run_id, cwd) / 'staging'
    publish_run(
        gateway,
        run_id=run_id,
        run_output_dir=result.output_dir,
        effective_args=result.effective_args,
        environment=result.environment,
        dataset_id=outcome.dataset_id,
        staging_dir=staging / 'run',
    )
    best = result.output_dir / 'weights' / 'best.pt'
    publish_model(
        gateway,
        model_id=run_id,
        best_weights_path=best,
        run_id=run_id,
        dataset_id=outcome.dataset_id,
        license_text='AGPL-3.0-only',
        notice_text=f'yolopipe model for run {run_id}',
        staging_dir=staging / 'model',
    )

    return {
        'mode': mode,
        'epochs': result.effective_args['epochs'],
        'dataset_id': outcome.dataset_id,
        'run_prefix': f's3://{constants.EMBY_OSS_BUCKET}/{constants.RUNS_PREFIX}{run_id}/',
        'model_prefix': f's3://{constants.EMBY_OSS_BUCKET}/{constants.MODELS_PREFIX}{run_id}/',
        'local_run_dir': str(result.output_dir),
    }
