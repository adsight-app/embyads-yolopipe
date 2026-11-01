"""The `validate` workflow: credentials, manifests, storage, checkpoint, GPU readiness.

Manifest files are downloaded and schema-validated (including strict
combined-catalog uniqueness) but no dataset image or mask object is fetched.
Plain validation treats GPU readiness as informational; ``require_gpu`` makes a
not-ready host a hard failure.
"""

from __future__ import annotations

from pathlib import Path

from . import preflight
from .checkpoint import ensure_base_checkpoint
from .errors import PreflightError
from .manifest import build_manifest_plan
from .paths import cache_root, require_working_roots
from .storage import S3Gateway


def run_validate(gateway: S3Gateway, *, require_gpu: bool = False, cwd: Path | None = None) -> dict:
    """Validate the environment without converting or training. Returns a report dict."""
    require_working_roots(cwd)

    # Manifest-only validation across all nine catalog manifests.
    plan = build_manifest_plan(gateway)

    # Base checkpoint integrity (download + verify into cache).
    checkpoint_path = ensure_base_checkpoint(gateway, cache_root(cwd))

    readiness = preflight.gpu_readiness()
    report = {
        'annotated_rows': plan.annotated_rows,
        'skipped_rows': plan.skipped_rows,
        'catalog_fingerprint': plan.catalog_fingerprint,
        'checkpoint': str(checkpoint_path),
        'training_readiness': {
            'state': 'ready' if readiness.ready else 'not_ready',
            'reason': readiness.reason,
            'device_count': readiness.device_count,
            'platform': readiness.platform,
        },
    }

    if require_gpu and not readiness.ready:
        raise PreflightError(f'--require-gpu set but host is not training-ready: {readiness.reason}')

    return report
