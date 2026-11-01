"""End-to-end CLI workflow tests with a combined fake S3 and mocked trainer."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from yolopipe import cli, constants, pipeline
from yolopipe.catalog import CATALOG

from tests.integration.fakes import PipelineFakeS3

BILLBOARD_HEX = '#00ff00'
BILLBOARD_BGR = (0, 255, 0)


def _png_bytes(with_box: bool) -> bytes:
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    if with_box:
        img[5:25, 5:35] = BILLBOARD_BGR
    ok, buf = cv2.imencode('.png', img)
    assert ok
    return buf.tobytes()


def _manifest_rows(job: str, n: int) -> str:
    rows = []
    for i in range(n):
        name = f'{job}-{i:03d}'
        rows.append(
            json.dumps(
                {
                    'source-ref': f's3://inputs/img/{name}.png',
                    f'{job}-ref': f's3://inputs/mask/{name}.png',
                    f'{job}-ref-metadata': {
                        'type': 'groundtruth/semantic-segmentation',
                        'internal-color-map': {
                            '0': {'class-name': 'BACKGROUND', 'hex-color': '#000000', 'confidence': 0},
                            '1': {'class-name': 'billboard', 'hex-color': BILLBOARD_HEX, 'confidence': 0},
                        },
                    },
                }
            )
        )
    return '\n'.join(rows)


@pytest.fixture
def populated_s3():
    """A fake S3 holding all nine manifests, their images/masks, and the base checkpoint."""
    fake = PipelineFakeS3()
    # Each job gets a few unique samples. Vary pixels so perceptual dedup keeps them.
    for idx, entry in enumerate(CATALOG):
        job = entry.job_name
        from yolopipe.models import S3Uri

        muri = S3Uri.parse(entry.output_manifest_uri)
        fake.put(muri.bucket, muri.key, _manifest_rows(job, 3).encode('utf-8'))
        for i in range(3):
            name = f'{job}-{i:03d}'
            # Distinct content per sample to avoid perceptual collapse.
            img = np.zeros((40, 50, 3), dtype=np.uint8)
            img[:, :] = ((idx * 17 + i * 53) % 255, (i * 91) % 255, (idx * 7) % 255)
            img[5:25, 5:35] = BILLBOARD_BGR
            ok, ibuf = cv2.imencode('.png', img)
            mask = np.zeros((40, 50, 3), dtype=np.uint8)
            mask[5:25, 5 + i : 35] = BILLBOARD_BGR
            ok2, mbuf = cv2.imencode('.png', mask)
            assert ok and ok2
            fake.put('inputs', f'img/{name}.png', ibuf.tobytes())
            fake.put('inputs', f'mask/{name}.png', mbuf.tobytes())

    # Base checkpoint with a matching sha + size so checkpoint verification passes.
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    fake.put(constants.EMBY_OSS_BUCKET, constants.BASE_CHECKPOINT_KEY, data)
    return fake


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'cache').mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def patched_gateway(populated_s3, monkeypatch):
    monkeypatch.setattr(cli, '_make_gateway', lambda profile, region: _gw(populated_s3))
    return populated_s3


def _gw(fake):
    from yolopipe.storage import S3Gateway

    return S3Gateway(fake)


def _invoke(*args):
    runner = CliRunner()
    return runner.invoke(cli.cli, list(args), obj={}, catch_exceptions=False)


# --- validate -----------------------------------------------------------------
def test_validate_plain_passes_on_cpu_host(working_dir, patched_gateway, monkeypatch):
    # Patch checksum to match the zero-filled fake checkpoint.
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', hashlib.sha256(data).hexdigest())

    from yolopipe.validate_cmd import run_validate

    report = run_validate(_gw(patched_gateway), require_gpu=False)
    assert report['training_readiness']['state'] == 'not_ready'
    assert report['annotated_rows'] == 27  # 9 jobs x 3 rows
    # Only manifests + checkpoint were read; no image/mask fetched.
    assert patched_gateway.image_mask_fetches() == []


def test_validate_require_gpu_fails_on_cpu_host(working_dir, patched_gateway, monkeypatch):
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', hashlib.sha256(data).hexdigest())

    from yolopipe.errors import PreflightError
    from yolopipe.validate_cmd import run_validate

    with pytest.raises(PreflightError):
        run_validate(_gw(patched_gateway), require_gpu=True)


# --- prepare ------------------------------------------------------------------
def _prep_sha(monkeypatch):
    data = b'\x00' * constants.BASE_CHECKPOINT_SIZE
    from yolopipe import checkpoint as cp

    monkeypatch.setattr(cp.constants, 'BASE_CHECKPOINT_SHA256', hashlib.sha256(data).hexdigest())


def test_prepare_full_run_publishes_dataset(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    outcome = pipeline.prepare(_gw(patched_gateway), run_id='acc-1', max_samples=None)
    assert outcome.survivors >= 3
    assert sum(len(v) for v in outcome.splits.values()) == outcome.survivors
    # Dataset published under the content-addressed prefix only.
    prefix = f'{constants.DATASETS_PREFIX}{outcome.dataset_id}/'
    assert any(k.startswith(prefix) for k in patched_gateway.uploads)
    assert all(k.startswith('yolopipe/') for k in patched_gateway.uploads)


def test_prepare_local_reuse_skips_manifest_download(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    gw = _gw(patched_gateway)
    pipeline.prepare(gw, run_id='acc-reuse', max_samples=None)
    accessed_after_first = len(patched_gateway.accessed)

    # Second prepare with identical params reuses local prep: no new S3 access.
    outcome = pipeline.prepare(gw, run_id='acc-reuse', max_samples=None)
    assert outcome.reused is True
    assert len(patched_gateway.accessed) == accessed_after_first


def test_prepare_mismatched_max_samples_fails(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    gw = _gw(patched_gateway)
    pipeline.prepare(gw, run_id='acc-mm', max_samples=None)
    from yolopipe.errors import ValidationError

    with pytest.raises(ValidationError, match='different'):
        pipeline.prepare(gw, run_id='acc-mm', max_samples=5)


def test_prepare_rejects_existing_run_prefix(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    patched_gateway.put(constants.EMBY_OSS_BUCKET, f'{constants.RUNS_PREFIX}taken/anything', b'x')
    from yolopipe.errors import RunCollisionError

    with pytest.raises(RunCollisionError):
        pipeline.prepare(_gw(patched_gateway), run_id='taken', max_samples=None)


def test_max_samples_below_three_rejected(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    from yolopipe.errors import SelectionError

    with pytest.raises(SelectionError):
        pipeline.prepare(_gw(patched_gateway), run_id='acc-tiny', max_samples=2)


# --- train (smoke, mocked trainer) --------------------------------------------
def test_train_smoke_mode_uses_single_gpu_recipe(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)

    # Avoid real GPU/preflight and real training.
    monkeypatch.setattr(pipeline.preflight, 'require_smoke', lambda: None)

    captured = {}

    def _fake_train(record, runtime_yaml, base_ckpt, mode, *, cache_root_path, runs_dir):
        from yolopipe.trainer import SMOKE_BATCH, TrainResult

        out = runs_dir / record.run_id / 'weights'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'best.pt').write_bytes(b'best')
        (out / 'last.pt').write_bytes(b'last')
        args = {'epochs': record.epochs, 'device': 0, 'batch': SMOKE_BATCH}
        captured['args'] = args
        captured['mode'] = mode
        return TrainResult(output_dir=runs_dir / record.run_id, effective_args=args, environment={'mode': mode})

    monkeypatch.setattr(pipeline, 'train', _fake_train)

    result = pipeline.run_training(_gw(patched_gateway), run_id='smoke-1', max_samples=100, epochs=3)
    assert captured['mode'] == 'smoke'
    assert captured['args']['device'] == 0
    assert result['epochs'] == 3
    # Run and model prefixes published with model_id == run_id.
    assert result['run_prefix'].endswith('/runs/smoke-1/')
    assert result['model_prefix'].endswith('/models/smoke-1/')
    marker = constants.COMPLETION_MARKER
    run_markers = [k for k in patched_gateway.upload_order if k.endswith(marker)]
    assert any('/runs/smoke-1/' in k for k in run_markers)
    assert any('/models/smoke-1/' in k for k in run_markers)


def test_train_production_requires_gpu_preflight(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    from yolopipe.errors import PreflightError

    def _boom():
        raise PreflightError('production training is Linux-only (NCCL DDP)')

    monkeypatch.setattr(pipeline.preflight, 'require_production', _boom)
    with pytest.raises(PreflightError, match='Linux-only'):
        pipeline.run_training(_gw(patched_gateway), run_id='prod-1', max_samples=None, epochs=1)


# --- visualize ----------------------------------------------------------------
def test_visualize_requires_prepared_dataset(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    from yolopipe.errors import ValidationError
    from yolopipe.paths import data_run_dir
    from yolopipe.visualize import visualize as run_visualize

    with pytest.raises(ValidationError, match='prepare'):
        run_visualize(data_run_dir('missing') / 'dataset', working_dir / 'viz')


def test_visualize_writes_bounded_named_files(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    pipeline.prepare(_gw(patched_gateway), run_id='viz-1', max_samples=None)

    from yolopipe.paths import data_run_dir
    from yolopipe.visualize import visualize as run_visualize

    out = working_dir / 'viz-out'
    result = run_visualize(data_run_dir('viz-1') / 'dataset', out)
    files = sorted(p.name for p in out.glob('*.png'))
    assert 0 < len(files) <= 16
    for i, name in enumerate(files):
        assert name.startswith(f'{i:02d}_')
    assert result['count'] == len(files)


def test_visualize_rejects_overlap_with_managed_tree(working_dir, patched_gateway, monkeypatch):
    _prep_sha(monkeypatch)
    pipeline.prepare(_gw(patched_gateway), run_id='viz-2', max_samples=None)
    from yolopipe.errors import StorageBoundaryError
    from yolopipe.paths import data_run_dir
    from yolopipe.visualize import visualize as run_visualize

    with pytest.raises(StorageBoundaryError):
        run_visualize(data_run_dir('viz-2') / 'dataset', working_dir / 'cache' / 'yolopipe' / 'x')


# --- CLI surface --------------------------------------------------------------
def test_cli_help_lists_all_commands():
    result = _invoke('--help')
    for cmd in ('validate', 'prepare', 'train', 'visualize'):
        assert cmd in result.output


def test_cli_aws_defaults():
    runner = CliRunner()
    captured = {}

    import yolopipe.cli as climod

    def _fake_validate(gateway, *, require_gpu=False, cwd=None):
        return {'training_readiness': {'state': 'not_ready', 'reason': 'x'}}

    def _capture_gateway(profile, region):
        captured['profile'] = profile
        captured['region'] = region

        class _Dummy:
            pass

        return _Dummy()

    climod._make_gateway = _capture_gateway  # noqa: SLF001
    import sys
    import types

    fake_mod = types.ModuleType('yolopipe.validate_cmd')
    fake_mod.run_validate = _fake_validate
    sys.modules['yolopipe.validate_cmd'] = fake_mod
    try:
        runner.invoke(climod.cli, ['validate'], obj={}, catch_exceptions=False)
    finally:
        del sys.modules['yolopipe.validate_cmd']
    assert captured['profile'] == 'embytest'
    assert captured['region'] == 'eu-west-1'
