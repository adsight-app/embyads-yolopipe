"""Trainer recipe, environment-ordering, and subprocess propagation tests.

All Torch/Ultralytics interaction is faked; no real GPU, training, or network
access occurs. The DDP subprocess machinery is exercised both via the real
isolation probe script and via in-process fakes that monkeypatch ``sys.modules``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yolopipe import constants, trainer
from yolopipe.errors import TrainingError
from yolopipe.models import RunRecord

PROBE = Path(__file__).resolve().parents[1] / 'subprocess' / 'check_training_env.py'


def _run_record(epochs: int = 0, *, run_id: str = 'run-abc', mode: str = trainer.PRODUCTION) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        dataset_id='ds-1',
        max_samples=None,
        epochs=epochs,
        mode=mode,
        effective_args={},
        environment={},
    )


# --- environment is exported before any import, inheritable by children -------
def test_env_vars_present_in_fresh_child(tmp_path):
    cache_root = tmp_path / 'cache' / 'yolopipe'
    # Fresh subprocess with a scrubbed environment proves no home fallback.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith('YOLO_')}
    clean_env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    result = subprocess.run(
        [sys.executable, str(PROBE), str(cache_root)],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_env_configured_before_torch_import(tmp_path, monkeypatch):
    """The worker must set env vars before importing torch/ultralytics."""
    observed = {}

    # Fake torch whose import-surface records the env at the moment the worker
    # first touches it (immediately after configure_training_env runs).
    fake_torch = types.ModuleType('torch')

    def _use_det(flag):
        observed['deterministic'] = flag
        observed['env_at_torch'] = dict(os.environ)

    fake_torch.use_deterministic_algorithms = _use_det
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i: f'FakeGPU{i}',
    )
    fake_torch.version = types.SimpleNamespace(cuda='13.0')
    fake_torch.__version__ = '2.10.0'

    captured_args = {}

    class FakeYOLO:
        def __init__(self, ckpt):
            captured_args['ckpt'] = ckpt
            observed['env_at_yolo'] = dict(os.environ)

        def train(self, **kwargs):
            captured_args['train'] = kwargs
            out = Path(kwargs['project']) / kwargs['name'] / 'weights'
            out.mkdir(parents=True, exist_ok=True)
            (out / 'best.pt').write_bytes(b'best')
            (out / 'last.pt').write_bytes(b'last')

    fake_ultra = types.ModuleType('ultralytics')
    fake_ultra.__version__ = '8.4.33'
    fake_ultra.utils = types.SimpleNamespace(ONLINE=True, AUTOINSTALL=True)
    fake_ultra.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'ultralytics', fake_ultra)
    monkeypatch.setitem(sys.modules, 'ultralytics.YOLO', fake_ultra.YOLO)

    # Scrub env so we can prove the worker sets it.
    for key in ('YOLO_OFFLINE', 'YOLO_AUTOINSTALL', 'YOLO_CONFIG_DIR', 'CUBLAS_WORKSPACE_CONFIG'):
        monkeypatch.delenv(key, raising=False)

    cache_root = tmp_path / 'cache'
    runs_dir = tmp_path / 'runs'
    args = trainer.build_effective_args(_run_record(epochs=5), tmp_path / 'data.yaml', runs_dir, trainer.SMOKE)

    class _Q:
        def __init__(self):
            self.item = None

        def put(self, v):
            self.item = v

    q = _Q()
    trainer._worker(str(cache_root), str(tmp_path / 'base.pt'), args, q)

    # Env must have been present by the time torch was touched.
    assert observed['env_at_torch']['YOLO_OFFLINE'] == 'true'
    assert observed['env_at_torch']['YOLO_AUTOINSTALL'] == 'false'
    assert observed['env_at_torch']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert Path(observed['env_at_torch']['YOLO_CONFIG_DIR']).is_absolute()
    assert observed['deterministic'] is True
    # Defense-in-depth patch applied post-import.
    assert fake_ultra.utils.ONLINE is False
    assert fake_ultra.utils.AUTOINSTALL is False
    assert q.item['environment']['torch'] == '2.10.0'


# --- exact recipe kwargs ------------------------------------------------------
def test_production_kwargs_are_exact(tmp_path):
    args = trainer.build_effective_args(
        _run_record(epochs=0), tmp_path / 'data.yaml', tmp_path / 'runs', trainer.PRODUCTION
    )
    assert args == {
        'data': str(tmp_path / 'data.yaml'),
        'epochs': constants.DEFAULT_EPOCHS,
        'imgsz': 640,
        'workers': 8,
        'deterministic': True,
        'amp': False,
        'save': True,
        'save_period': -1,
        'project': str(tmp_path / 'runs'),
        'name': 'run-abc',
        'exist_ok': False,
        'device': [0, 1, 2, 3],
        'batch': 16,
    }


def test_smoke_kwargs_single_device_scaled_batch(tmp_path):
    args = trainer.build_effective_args(
        _run_record(epochs=1, mode=trainer.SMOKE), tmp_path / 'data.yaml', tmp_path / 'runs', trainer.SMOKE
    )
    assert args['device'] == 0
    assert args['batch'] == trainer.SMOKE_BATCH
    assert args['imgsz'] == 640
    assert args['deterministic'] is True
    assert args['amp'] is False
    assert args['save'] is True
    assert args['save_period'] == -1
    assert args['epochs'] == 1


def test_single_gpu_kwargs_one_device_full_batch(tmp_path):
    args = trainer.build_effective_args(
        _run_record(epochs=0, mode=trainer.SINGLE_GPU), tmp_path / 'data.yaml', tmp_path / 'runs', trainer.SINGLE_GPU
    )
    # Single-GPU: device 0 but the full global batch (not the reduced smoke batch).
    assert args['device'] == 0
    assert args['batch'] == constants.GLOBAL_BATCH
    assert args['epochs'] == constants.DEFAULT_EPOCHS
    assert args['imgsz'] == 640
    assert args['deterministic'] is True
    assert args['amp'] is False


def test_default_vs_override_epochs(tmp_path):
    default = trainer.build_effective_args(
        _run_record(epochs=0), tmp_path / 'd.yaml', tmp_path / 'runs', trainer.PRODUCTION
    )
    override = trainer.build_effective_args(
        _run_record(epochs=7), tmp_path / 'd.yaml', tmp_path / 'runs', trainer.PRODUCTION
    )
    assert default['epochs'] == 100
    assert override['epochs'] == 7


@pytest.mark.parametrize('bad', [-1, 0 - 5, -100])
def test_invalid_epochs_rejected(tmp_path, bad):
    rec = _run_record(epochs=bad)
    with pytest.raises(TrainingError, match='positive integer'):
        trainer.build_effective_args(rec, tmp_path / 'd.yaml', tmp_path / 'runs', trainer.PRODUCTION)


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(TrainingError, match='mode'):
        trainer.build_effective_args(_run_record(epochs=1), tmp_path / 'd.yaml', tmp_path / 'runs', 'turbo')


# --- subprocess propagation and required outputs ------------------------------
def test_child_failure_propagates(tmp_path, monkeypatch):
    def _boom(cache_root_path, base_ckpt_path, args):
        raise TrainingError('training subprocess failed with exit status 1; inspect the preserved run logs')

    monkeypatch.setattr(trainer, '_run_in_child', _boom)
    with pytest.raises(TrainingError, match='exit status'):
        trainer.train(
            _run_record(epochs=1),
            tmp_path / 'data.yaml',
            tmp_path / 'base.pt',
            trainer.PRODUCTION,
            cache_root_path=tmp_path / 'cache',
            runs_dir=tmp_path / 'runs',
        )


def test_missing_weights_raises(tmp_path, monkeypatch):
    # Child "succeeds" but produces no weights.
    monkeypatch.setattr(trainer, '_run_in_child', lambda *a, **k: {'torch': 'x'})
    with pytest.raises(TrainingError, match='required weights are missing'):
        trainer.train(
            _run_record(epochs=1, run_id='run-x'),
            tmp_path / 'data.yaml',
            tmp_path / 'base.pt',
            trainer.PRODUCTION,
            cache_root_path=tmp_path / 'cache',
            runs_dir=tmp_path / 'runs',
        )


def test_train_returns_outputs_and_records_mode(tmp_path, monkeypatch):
    runs_dir = tmp_path / 'runs'

    def _fake_child(cache_root_path, base_ckpt_path, args):
        out = runs_dir / 'run-ok' / 'weights'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'best.pt').write_bytes(b'b')
        (out / 'last.pt').write_bytes(b'l')
        return {'torch': '2.10.0', 'device_count': 1}

    monkeypatch.setattr(trainer, '_run_in_child', _fake_child)
    result = trainer.train(
        _run_record(epochs=3, run_id='run-ok', mode=trainer.SMOKE),
        tmp_path / 'data.yaml',
        tmp_path / 'base.pt',
        trainer.SMOKE,
        cache_root_path=tmp_path / 'cache',
        runs_dir=runs_dir,
    )
    assert result.output_dir == runs_dir / 'run-ok'
    assert result.environment['mode'] == trainer.SMOKE
    assert result.effective_args['device'] == 0
    assert result.effective_args['batch'] == trainer.SMOKE_BATCH
