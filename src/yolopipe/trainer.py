"""Fixed YOLO segmentation training recipe with pre-import isolation.

The only permitted variation from the frozen recipe is ``epochs`` (always) and,
in smoke mode, a single-GPU ``device`` plus a reduced ``batch``. Training runs in
a child process so that DDP failures surface as a non-zero exit status with local
logs preserved, and so the isolated environment is inherited by DDP workers.

Torch and Ultralytics are imported lazily *inside* the worker, only after
:func:`yolopipe.training_env.configure_training_env` has exported the offline
and determinism environment variables.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from yolopipe import constants
from yolopipe.errors import TrainingError
from yolopipe.models import RunRecord

PRODUCTION = 'production'
SMOKE = 'smoke'

# Batch per single GPU used in smoke mode (scaled down from the global batch).
SMOKE_BATCH = 4


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a completed training run."""

    output_dir: Path
    effective_args: dict
    environment: dict


def _resolve_epochs(run_record: RunRecord) -> int:
    epochs = run_record.epochs if run_record.epochs else constants.DEFAULT_EPOCHS
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise TrainingError(f'epochs must be a positive integer; got {run_record.epochs!r}')
    return epochs


def build_effective_args(run_record: RunRecord, data_yaml_path: Path, project_dir: Path, mode: str) -> dict:
    """Construct the exact Ultralytics ``train`` kwargs for the given mode.

    Fixed for both modes: ``imgsz=640``, ``deterministic=True``, ``amp=False``,
    ``save=True``, ``save_period=-1``, ``workers=8``. Production uses the four
    production devices and the global batch; smoke uses a single device and a
    reduced batch. No other key may be overridden.
    """
    if mode not in (PRODUCTION, SMOKE):
        raise TrainingError(f'unknown training mode: {mode!r}')

    epochs = _resolve_epochs(run_record)
    args = {
        'data': str(data_yaml_path),
        'epochs': epochs,
        'imgsz': constants.IMAGE_SIZE,
        'workers': constants.WORKERS,
        'deterministic': True,
        'amp': False,
        'save': True,
        'save_period': -1,
        'project': str(project_dir),
        'name': run_record.run_id,
        'exist_ok': False,
    }
    if mode == PRODUCTION:
        args['device'] = list(constants.PRODUCTION_DEVICES)
        args['batch'] = constants.GLOBAL_BATCH
    else:  # SMOKE
        args['device'] = 0
        args['batch'] = SMOKE_BATCH
    return args


def _worker(cache_root_path: str, base_ckpt_path: str, args: dict, result_queue) -> None:
    """Child-process entry point: isolate env, import, train, report environment.

    Must configure the environment *before* importing Torch/Ultralytics. On
    failure, pushes the error text and exits non-zero so the parent can report
    subprocess diagnostics while local logs remain on disk.
    """
    from yolopipe.training_env import configure_training_env

    configure_training_env(Path(cache_root_path))

    # Lazy imports strictly after environment isolation.
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    torch.use_deterministic_algorithms(True)

    # Defense-in-depth: disable any network/auto-install side effects post-import.
    ultralytics.utils.ONLINE = False
    ultralytics.utils.AUTOINSTALL = False

    model = YOLO(base_ckpt_path)
    model.train(**args)

    environment = _collect_environment(torch, ultralytics)
    result_queue.put({'environment': environment})


def _collect_environment(torch, ultralytics) -> dict:
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415

    try:
        cuda_version = torch.version.cuda
    except Exception:  # noqa: BLE001
        cuda_version = None
    try:
        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        device_count = 0
    try:
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
    except Exception:  # noqa: BLE001
        devices = []
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': getattr(torch, '__version__', None),
        'cuda': cuda_version,
        'ultralytics': getattr(ultralytics, '__version__', None),
        'device_count': device_count,
        'devices': devices,
    }


def _run_in_child(cache_root_path: Path, base_ckpt_path: Path, args: dict) -> dict:
    """Run the worker in a child process and propagate failures with exit status."""
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(str(cache_root_path), str(base_ckpt_path), args, result_queue),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise TrainingError(
            f'training subprocess failed with exit status {proc.exitcode}; inspect the preserved run logs for details'
        )
    try:
        payload = result_queue.get_nowait()
    except Exception:  # noqa: BLE001
        payload = {}
    return payload.get('environment', {})


def train(
    run_record: RunRecord,
    data_yaml_path: Path,
    base_ckpt_path: Path,
    mode: str,
    *,
    cache_root_path: Path,
    runs_dir: Path,
) -> TrainResult:
    """Train with the fixed recipe in a child process and validate required outputs.

    ``cache_root_path`` locates the Ultralytics config isolation directory;
    ``runs_dir`` is the local project directory under which the run output lands.
    Returns the run output directory, the exact effective args, and the captured
    environment. Requires both ``weights/best.pt`` and ``weights/last.pt``.
    """
    runs_dir = Path(runs_dir)
    args = build_effective_args(run_record, Path(data_yaml_path), runs_dir, mode)

    environment = _run_in_child(Path(cache_root_path), Path(base_ckpt_path), args)
    environment = {'mode': mode, **environment}

    output_dir = runs_dir / run_record.run_id
    best = output_dir / 'weights' / 'best.pt'
    last = output_dir / 'weights' / 'last.pt'
    missing = [p.name for p in (best, last) if not p.exists()]
    if missing:
        raise TrainingError(f'training completed but required weights are missing: {", ".join(missing)}')

    return TrainResult(output_dir=output_dir, effective_args=args, environment=environment)
