"""Immutable run/model publication contract tests (fake S3 serving bytes)."""

from __future__ import annotations

import pytest

from yolopipe import constants
from yolopipe.errors import RunCollisionError
from yolopipe.run_publish import assert_run_absent, publish_model, publish_run
from yolopipe.storage import S3Gateway

RUN_ID = 'run-xyz'


@pytest.fixture
def client(bytes_s3_factory):
    return bytes_s3_factory()


@pytest.fixture
def gateway(client):
    return S3Gateway(client)


def _make_run_output(tmp_path):
    out = tmp_path / 'local-run' / RUN_ID
    weights = out / 'weights'
    weights.mkdir(parents=True)
    (weights / 'best.pt').write_bytes(b'best-weights')
    (weights / 'last.pt').write_bytes(b'last-weights')
    (out / 'results.csv').write_text('epoch,loss\n1,0.5\n', encoding='utf-8')
    (out / 'train.log').write_text('log line\n', encoding='utf-8')
    return out


def test_absent_prefix_passes(gateway):
    assert_run_absent(gateway, RUN_ID)  # must not raise


def test_existing_finalized_prefix_collides(gateway, client):
    client.put(constants.EMBY_OSS_BUCKET, f'{constants.RUNS_PREFIX}{RUN_ID}/{constants.COMPLETION_MARKER}', b'')
    with pytest.raises(RunCollisionError):
        assert_run_absent(gateway, RUN_ID)


def test_existing_incomplete_prefix_collides(gateway, client):
    # Any object (even a partial weight) under the model prefix must collide.
    client.put(constants.EMBY_OSS_BUCKET, f'{constants.MODELS_PREFIX}{RUN_ID}/best.pt', b'partial')
    with pytest.raises(RunCollisionError):
        assert_run_absent(gateway, RUN_ID)


def test_publish_run_uploads_marker_last_and_bounded(gateway, client, tmp_path):
    out = _make_run_output(tmp_path)
    marker = publish_run(
        gateway,
        run_id=RUN_ID,
        run_output_dir=out,
        effective_args={'epochs': 7, 'imgsz': 640},
        environment={'mode': 'production', 'torch': '2.10.0'},
        dataset_id='ds-abc',
        staging_dir=tmp_path / 'stage',
    )
    # Completion marker is the final upload.
    assert client.upload_order[-1].endswith(constants.COMPLETION_MARKER)
    assert marker.key.endswith(constants.COMPLETION_MARKER)
    # Weights, metadata, and checksum manifest all uploaded.
    keys = set(client.uploads)
    assert f'{constants.RUNS_PREFIX}{RUN_ID}/output/weights/best.pt' in keys
    assert f'{constants.RUNS_PREFIX}{RUN_ID}/effective_args.json' in keys
    assert f'{constants.RUNS_PREFIX}{RUN_ID}/environment.json' in keys
    assert f'{constants.RUNS_PREFIX}{RUN_ID}/checksums.json' in keys
    # Bounded uploads: never more concurrency than the transfer worker pool.
    assert client.max_observed_concurrency <= constants.TRANSFER_WORKERS


def test_publish_run_rejects_preexisting(gateway, client, tmp_path):
    out = _make_run_output(tmp_path)
    client.put(constants.EMBY_OSS_BUCKET, f'{constants.RUNS_PREFIX}{RUN_ID}/anything', b'x')
    with pytest.raises(RunCollisionError):
        publish_run(
            gateway,
            run_id=RUN_ID,
            run_output_dir=out,
            effective_args={},
            environment={},
            dataset_id='ds',
            staging_dir=tmp_path / 'stage',
        )


def test_publish_model_marker_last_and_model_id_equals_run_id(gateway, client, tmp_path):
    best = tmp_path / 'best.pt'
    best.write_bytes(b'the-best-weights')
    marker = publish_model(
        gateway,
        model_id=RUN_ID,
        best_weights_path=best,
        run_id=RUN_ID,
        dataset_id='ds-abc',
        license_text='AGPL-3.0-only',
        notice_text='NOTICE',
        staging_dir=tmp_path / 'stage',
    )
    assert client.upload_order[-1].endswith(constants.COMPLETION_MARKER)
    assert marker.key == f'{constants.MODELS_PREFIX}{RUN_ID}/{constants.COMPLETION_MARKER}'
    keys = set(client.uploads)
    assert f'{constants.MODELS_PREFIX}{RUN_ID}/best.pt' in keys
    assert f'{constants.MODELS_PREFIX}{RUN_ID}/best.pt.sha256' in keys
    assert f'{constants.MODELS_PREFIX}{RUN_ID}/LICENSE' in keys
    assert f'{constants.MODELS_PREFIX}{RUN_ID}/NOTICE' in keys


def test_publish_model_requires_model_id_equals_run_id(gateway, tmp_path):
    best = tmp_path / 'best.pt'
    best.write_bytes(b'w')
    with pytest.raises(RunCollisionError, match='must equal'):
        publish_model(
            gateway,
            model_id='model-other',
            best_weights_path=best,
            run_id=RUN_ID,
            dataset_id='ds',
            license_text='L',
            notice_text='N',
            staging_dir=tmp_path / 'stage',
        )


def test_epochs_in_run_metadata_not_dataset_id(gateway, client, tmp_path):
    out = _make_run_output(tmp_path)
    publish_run(
        gateway,
        run_id=RUN_ID,
        run_output_dir=out,
        effective_args={'epochs': 42},
        environment={},
        dataset_id='ds-stable',
        staging_dir=tmp_path / 'stage',
    )
    # epochs rides in effective args metadata; the linkage dataset_id is unaffected.
    linkage_key = f'{constants.RUNS_PREFIX}{RUN_ID}/linkage.json'
    body = client.get_object(Bucket=constants.EMBY_OSS_BUCKET, Key=linkage_key)['Body'].read().decode()
    assert 'ds-stable' in body
    assert '42' not in body
