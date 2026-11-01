"""Immutable run and model publication under the emby-oss boundary.

Publication is append-only and collision-safe: a run/model prefix that already
holds *any* object (finalized or partially written) is rejected. Files are
uploaded with a bounded worker pool, a canonical checksum manifest is written,
and the ``_COMPLETE`` marker is always uploaded last so readers can distinguish
a finalized prefix from an interrupted one. Nothing is ever deleted or resumed.
"""

from __future__ import annotations

from pathlib import Path

from yolopipe import constants
from yolopipe.errors import RunCollisionError
from yolopipe.models import S3Uri, canonical_json, sha256_file


def _run_prefix(run_id: str) -> str:
    return f'{constants.RUNS_PREFIX}{run_id}/'


def _model_prefix(model_id: str) -> str:
    return f'{constants.MODELS_PREFIX}{model_id}/'


def _assert_prefix_absent(gateway, prefix: str) -> None:
    if gateway.prefix_has_objects(constants.EMBY_OSS_BUCKET, prefix):
        raise RunCollisionError(
            f'refusing to publish: objects already exist under '
            f's3://{constants.EMBY_OSS_BUCKET}/{prefix} (never delete or resume)'
        )


def assert_run_absent(gateway, run_id: str) -> None:
    """Raise :class:`RunCollisionError` if any object exists under runs/ or models/ for ``run_id``.

    This is the pre-work collision guard, called once before training starts.
    The individual publishers guard only their own prefix so that publishing the
    model after the run does not falsely collide with the just-written run.
    """
    for prefix in (_run_prefix(run_id), _model_prefix(run_id)):
        _assert_prefix_absent(gateway, prefix)


def _write_json(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(obj))
    return path


def _checksum_manifest(files: list[tuple[Path, str]]) -> dict:
    """Map each published key to its local file's SHA-256."""
    return {key: sha256_file(local) for local, key in files}


def publish_run(
    gateway,
    *,
    run_id: str,
    run_output_dir: Path,
    effective_args: dict,
    environment: dict,
    dataset_id: str,
    staging_dir: Path,
) -> S3Uri:
    """Upload the complete run directory plus metadata; write ``_COMPLETE`` last.

    ``staging_dir`` is a local scratch directory where generated metadata and the
    checksum manifest are materialized before upload. Returns the marker URI.
    """
    _assert_prefix_absent(gateway, _run_prefix(run_id))

    prefix = _run_prefix(run_id)
    staging_dir = Path(staging_dir)

    items: list[tuple[Path, str]] = []

    # 1. Run directory payload (weights, logs, plots, results) under runs/<id>/output/.
    run_output_dir = Path(run_output_dir)
    for local in sorted(p for p in run_output_dir.rglob('*') if p.is_file()):
        rel = local.relative_to(run_output_dir).as_posix()
        items.append((local, f'{prefix}output/{rel}'))

    # 2. Metadata sidecars. epochs lives here (run metadata), never in a dataset id.
    args_file = _write_json(staging_dir / 'effective_args.json', effective_args)
    env_file = _write_json(staging_dir / 'environment.json', environment)
    linkage_file = _write_json(
        staging_dir / 'linkage.json',
        {'run_id': run_id, 'model_id': run_id, 'dataset_id': dataset_id},
    )
    items.append((args_file, f'{prefix}effective_args.json'))
    items.append((env_file, f'{prefix}environment.json'))
    items.append((linkage_file, f'{prefix}linkage.json'))

    # 3. Checksum manifest covering every payload file uploaded so far.
    manifest = _checksum_manifest(items)
    manifest_file = _write_json(staging_dir / 'checksums.json', manifest)
    items.append((manifest_file, f'{prefix}checksums.json'))

    gateway.upload_many(items)

    # 4. Completion marker LAST, after all payload is durably uploaded.
    marker_file = staging_dir / '_COMPLETE'
    marker_file.write_bytes(b'')
    return gateway.upload(marker_file, f'{prefix}{constants.COMPLETION_MARKER}')


def publish_model(
    gateway,
    *,
    model_id: str,
    best_weights_path: Path,
    run_id: str,
    dataset_id: str,
    license_text: str,
    notice_text: str,
    staging_dir: Path,
) -> S3Uri:
    """Upload ``best.pt`` + checksum + linkage + license/notice; write ``_COMPLETE`` last.

    ``model_id`` must equal ``run_id`` (enforced upstream via ``RunRecord``). The
    collision guard rejects any pre-existing run or model prefix.
    """
    if model_id != run_id:
        raise RunCollisionError(f'model_id ({model_id}) must equal run_id ({run_id})')

    _assert_prefix_absent(gateway, _model_prefix(model_id))

    prefix = _model_prefix(model_id)
    staging_dir = Path(staging_dir)
    best_weights_path = Path(best_weights_path)

    items: list[tuple[Path, str]] = [(best_weights_path, f'{prefix}best.pt')]

    best_sha = sha256_file(best_weights_path)
    sha_file = staging_dir / 'best.pt.sha256'
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    sha_file.write_text(f'{best_sha}  best.pt\n', encoding='utf-8')
    items.append((sha_file, f'{prefix}best.pt.sha256'))

    linkage_file = _write_json(
        staging_dir / 'linkage.json',
        {'model_id': model_id, 'run_id': run_id, 'dataset_id': dataset_id},
    )
    items.append((linkage_file, f'{prefix}linkage.json'))

    license_file = staging_dir / 'LICENSE'
    license_file.write_text(license_text, encoding='utf-8')
    items.append((license_file, f'{prefix}LICENSE'))

    notice_file = staging_dir / 'NOTICE'
    notice_file.write_text(notice_text, encoding='utf-8')
    items.append((notice_file, f'{prefix}NOTICE'))

    manifest = _checksum_manifest(items)
    manifest_file = _write_json(staging_dir / 'checksums.json', manifest)
    items.append((manifest_file, f'{prefix}checksums.json'))

    gateway.upload_many(items)

    marker_file = staging_dir / '_COMPLETE'
    marker_file.write_bytes(b'')
    return gateway.upload(marker_file, f'{prefix}{constants.COMPLETION_MARKER}')
