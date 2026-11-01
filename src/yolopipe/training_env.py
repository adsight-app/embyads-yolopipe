"""Pre-import Ultralytics/Torch isolation via process environment variables.

This module MUST be importable and callable *before* any ``torch`` or
``ultralytics`` import, and it must not itself import either. It exports plain
``os.environ`` variables so that DDP child processes inherit the isolation.
"""

from __future__ import annotations

import os
from pathlib import Path

from yolopipe import constants


def _ultralytics_config_path(cache_root_path: Path) -> Path:
    """Return the native, absolute Ultralytics config directory under the cache."""
    return (Path(cache_root_path) / constants.ULTRALYTICS_CONFIG_SUBDIR).resolve()


def configure_training_env(cache_root_path: Path) -> dict[str, str]:
    """Export offline/deterministic isolation env vars. Return the values set.

    Safe to call repeatedly and before any Torch/Ultralytics import. The config
    directory is created so Ultralytics never falls back to a home-directory
    settings file. Values live in ``os.environ`` so subprocess (DDP) children
    inherit them automatically.
    """
    config_dir = _ultralytics_config_path(cache_root_path)
    config_dir.mkdir(parents=True, exist_ok=True)

    env = {
        'YOLO_OFFLINE': 'true',
        'YOLO_AUTOINSTALL': 'false',
        'YOLO_CONFIG_DIR': str(config_dir),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    }
    for key, value in env.items():
        os.environ[key] = value
    return env
