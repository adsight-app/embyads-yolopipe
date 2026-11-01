"""GPU readiness and strict preflight contract tests (fake torch module)."""

from __future__ import annotations

import sys
import types

import pytest

from yolopipe import preflight
from yolopipe.errors import PreflightError


def _fake_torch(*, cuda_available: bool, device_count: int, nccl: bool) -> types.ModuleType:
    mod = types.ModuleType('torch')

    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
    )
    distributed = types.SimpleNamespace(is_nccl_available=lambda: nccl)
    mod.cuda = cuda
    mod.distributed = distributed
    return mod


@pytest.fixture
def install_torch(monkeypatch):
    def _install(**kwargs):
        monkeypatch.setitem(sys.modules, 'torch', _fake_torch(**kwargs))

    return _install


def test_plain_readiness_never_raises_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)  # force import failure path
    monkeypatch.delitem(sys.modules, 'torch')
    # Ensure the import fails cleanly by pointing at a non-importable name.
    monkeypatch.setattr(preflight, '_probe', lambda: (False, 0, False))
    result = preflight.gpu_readiness()
    assert result.ready is False
    assert isinstance(result.reason, str)


def test_linux_four_gpu_nccl_is_ready(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    install_torch(cuda_available=True, device_count=4, nccl=True)
    result = preflight.gpu_readiness()
    assert result.ready is True
    assert result.device_count == 4
    assert result.platform == 'linux'
    preflight.require_production()  # must not raise


def test_windows_is_rejected_for_production(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    install_torch(cuda_available=True, device_count=4, nccl=True)
    assert preflight.gpu_readiness().ready is False
    with pytest.raises(PreflightError, match='Linux-only'):
        preflight.require_production()


def test_too_few_gpus_fails_production(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    install_torch(cuda_available=True, device_count=2, nccl=True)
    assert preflight.gpu_readiness().ready is False
    with pytest.raises(PreflightError, match='4 GPUs'):
        preflight.require_production()


def test_missing_nccl_fails_production(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    install_torch(cuda_available=True, device_count=4, nccl=False)
    with pytest.raises(PreflightError, match='NCCL'):
        preflight.require_production()


def test_no_cuda_fails_production(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    install_torch(cuda_available=False, device_count=0, nccl=True)
    with pytest.raises(PreflightError, match='CUDA'):
        preflight.require_production()


def test_smoke_allows_single_gpu_on_any_os(monkeypatch, install_torch):
    for plat in ('win32', 'linux', 'darwin'):
        monkeypatch.setattr(sys, 'platform', plat)
        install_torch(cuda_available=True, device_count=1, nccl=False)
        preflight.require_smoke()  # must not raise on any OS


def test_smoke_fails_without_cuda(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    install_torch(cuda_available=False, device_count=0, nccl=False)
    with pytest.raises(PreflightError, match='CUDA'):
        preflight.require_smoke()


def test_readiness_reports_platform_without_raising(monkeypatch, install_torch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    install_torch(cuda_available=False, device_count=0, nccl=False)
    result = preflight.gpu_readiness()
    assert result.ready is False
    assert result.platform == 'darwin'
