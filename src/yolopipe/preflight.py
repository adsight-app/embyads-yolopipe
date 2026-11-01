"""GPU readiness reporting and strict training preflight.

``gpu_readiness`` never raises: it is safe to call during non-training
validation on a CPU/Windows host without GPUs. ``require_production`` and
``require_smoke`` enforce the hard platform contracts and raise
:class:`PreflightError` when they are not met. All Torch imports are lazy so the
module imports cleanly on hosts where Torch is absent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from yolopipe import constants
from yolopipe.errors import PreflightError


@dataclass(frozen=True)
class GpuReadiness:
    """Structured, non-raising readiness snapshot for a host."""

    ready: bool
    reason: str
    device_count: int
    platform: str


def _probe() -> tuple[bool, int, bool]:
    """Return (cuda_available, device_count, nccl_available). Never raises for absence."""
    try:
        import torch  # noqa: PLC0415 - lazy by design
    except Exception:  # noqa: BLE001 - torch absent on this host
        return (False, 0, False)
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:  # noqa: BLE001
        cuda_available, device_count = False, 0
    try:
        nccl_available = bool(torch.distributed.is_nccl_available())
    except Exception:  # noqa: BLE001
        nccl_available = False
    return (cuda_available, device_count, nccl_available)


def gpu_readiness() -> GpuReadiness:
    """Report GPU training readiness without failing. Suitable for plain validation."""
    cuda_available, device_count, nccl_available = _probe()
    required = len(constants.PRODUCTION_DEVICES)

    if sys.platform != 'linux':
        reason = f'production training requires linux; this host is {sys.platform}'
        return GpuReadiness(False, reason, device_count, sys.platform)
    if not cuda_available:
        return GpuReadiness(False, 'no CUDA device is available', device_count, sys.platform)
    if device_count < required:
        reason = f'production training requires {required} GPUs; found {device_count}'
        return GpuReadiness(False, reason, device_count, sys.platform)
    if not nccl_available:
        return GpuReadiness(False, 'NCCL backend is not available for DDP', device_count, sys.platform)

    return GpuReadiness(True, 'ready', device_count, sys.platform)


def require_production() -> None:
    """Raise :class:`PreflightError` unless the strict Linux four-GPU NCCL contract holds."""
    cuda_available, device_count, nccl_available = _probe()
    required = len(constants.PRODUCTION_DEVICES)

    if sys.platform != 'linux':
        raise PreflightError(f'production training is Linux-only (NCCL DDP); this host is {sys.platform}')
    if not cuda_available:
        raise PreflightError('production training requires an available CUDA device')
    if device_count < required:
        raise PreflightError(
            f'production training requires {required} GPUs (devices {list(constants.PRODUCTION_DEVICES)}); '
            f'found {device_count}'
        )
    if not nccl_available:
        raise PreflightError('production training requires the NCCL backend for DDP')


def require_single_gpu() -> None:
    """Raise :class:`PreflightError` unless Linux with at least one available CUDA GPU.

    Single-GPU mode runs the full dataset and default recipe on device ``0`` without
    DDP/NCCL. It is Linux-only to match the production training platform contract.
    """
    cuda_available, device_count, _nccl = _probe()
    if sys.platform != 'linux':
        raise PreflightError(f'single-GPU training is Linux-only; this host is {sys.platform}')
    if not cuda_available or device_count < 1:
        raise PreflightError('single-GPU training requires at least one available CUDA GPU')


def require_smoke() -> None:
    """Raise :class:`PreflightError` unless a single CUDA GPU is available (any OS)."""
    cuda_available, device_count, _nccl = _probe()
    if not cuda_available or device_count < 1:
        raise PreflightError('smoke training requires at least one available CUDA GPU')
