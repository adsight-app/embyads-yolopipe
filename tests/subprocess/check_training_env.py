"""Child-process probe: assert the four isolation env vars are present.

Invoked by ``tests/unit/test_trainer.py`` as a fresh subprocess. It calls
:func:`yolopipe.training_env.configure_training_env` and then verifies that the
expected environment variables are exported in ``os.environ`` (as a DDP child
would inherit them). Exits 0 on success, non-zero with a diagnostic on failure.
It imports neither Torch nor Ultralytics.

Usage: python check_training_env.py <cache_root_path>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable without an editable install.
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from yolopipe.training_env import configure_training_env  # noqa: E402


def main() -> int:
    cache_root = Path(sys.argv[1])
    configure_training_env(cache_root)

    expected_config = str((cache_root / 'ultralytics').resolve())
    checks = {
        'YOLO_OFFLINE': 'true',
        'YOLO_AUTOINSTALL': 'false',
        'YOLO_CONFIG_DIR': expected_config,
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    }
    for key, want in checks.items():
        got = os.environ.get(key)
        if got != want:
            sys.stderr.write(f'env {key} expected {want!r}, got {got!r}\n')
            return 1

    # Config dir must be absolute and under the cache, never a home fallback.
    if not Path(os.environ['YOLO_CONFIG_DIR']).is_absolute():
        sys.stderr.write('YOLO_CONFIG_DIR must be absolute\n')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
