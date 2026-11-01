"""Local path resolution and run-identifier validation with traversal guards."""

from __future__ import annotations

import re
from pathlib import Path

from yolopipe import constants
from yolopipe.errors import StorageBoundaryError, ValidationError

_RUN_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ValidationError(
            'run-id must be 1-64 chars of letters, digits, dot, underscore, or hyphen and start with a letter or digit'
        )
    if run_id in {'.', '..'}:
        raise ValidationError('run-id must not be a path component')
    return run_id


def require_working_roots(cwd: Path | None = None) -> None:
    base = cwd or Path.cwd()
    for name in ('data', 'cache'):
        if not (base / name).is_dir():
            raise ValidationError(f'required working directory ./{name} does not exist; create it before running')


def data_run_dir(run_id: str, cwd: Path | None = None) -> Path:
    validate_run_id(run_id)
    base = cwd or Path.cwd()
    return base / 'data' / 'yolopipe' / run_id


def cache_root(cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    return base / 'cache' / 'yolopipe'


def cache_objects_dir(cwd: Path | None = None) -> Path:
    return cache_root(cwd) / constants.CACHE_OBJECTS_SUBDIR


def ultralytics_config_dir(cwd: Path | None = None) -> Path:
    return cache_root(cwd) / constants.ULTRALYTICS_CONFIG_SUBDIR


def validate_output_dir(output: Path, cwd: Path | None = None) -> Path:
    """Resolve a visualization output dir, rejecting overlap with managed data/cache trees."""
    base = (cwd or Path.cwd()).resolve()
    resolved = Path(output).resolve()
    managed = [(base / 'data' / 'yolopipe').resolve(), (base / 'cache' / 'yolopipe').resolve()]
    for tree in managed:
        if resolved == tree or tree in resolved.parents or resolved in tree.parents:
            raise StorageBoundaryError(f'visualization output {resolved} overlaps a pipeline-managed tree')
    return resolved
