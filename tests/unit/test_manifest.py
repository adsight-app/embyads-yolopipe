"""Manifest-only validation contract tests."""

from __future__ import annotations

import json

import pytest

from yolopipe.catalog import CATALOG
from yolopipe.errors import ManifestError
from yolopipe.manifest import build_manifest_plan
from yolopipe.models import sha256_hex
from yolopipe.storage import S3Gateway

from tests.conftest import ManifestOnlyFakeS3, read_fixture

FIRST_JOB = CATALOG[0].job_name
SECOND_JOB = CATALOG[1].job_name


def _bodies(mapping: dict[str, str]) -> dict[str, str]:
    """Build the nine-manifest body map; unspecified jobs get an empty body."""
    bodies: dict[str, str] = {}
    for entry in CATALOG:
        bodies[entry.output_manifest_uri] = mapping.get(entry.job_name, '')
    return bodies


def _row(job: str, name: str, *, color='#ff0000', cls='billboard', group=None, img_bucket='synthetic-gta5'):
    row = {
        'source-ref': f's3://{img_bucket}/img/{name}.png',
        f'{job}-ref': f's3://synthetic-gta5/mask/{name}.png',
        f'{job}-ref-metadata': {
            'type': 'groundtruth/semantic-segmentation',
            'internal-color-map': {'#000000': 'BACKGROUND', color: cls},
        },
    }
    if group is not None:
        row['group-id'] = group
    return json.dumps(row)


def _gateway(bodies: dict[str, str]) -> tuple[S3Gateway, ManifestOnlyFakeS3]:
    client = ManifestOnlyFakeS3(bodies)
    return S3Gateway(client), client


def test_annotated_rows_parse_with_counts_and_sample_ids():
    bodies = _bodies({FIRST_JOB: read_fixture('annotated.manifest')})
    gateway, client = _gateway(bodies)

    plan = build_manifest_plan(gateway)

    assert plan.total_rows == 4
    assert plan.annotated_rows == 4
    assert plan.skipped_rows == 0

    samples = plan.candidates_by_manifest[FIRST_JOB]
    assert [s.source_filename for s in samples] == ['a001.png', 'a002.png', 'a003.jpg', 'a004.png']
    # sample_id is sha256 of the normalized source-ref URI.
    assert samples[0].sample_id == sha256_hex(b's3://synthetic-gta5/img/a001.png')
    assert samples[2].extension == '.jpg'
    assert samples[0].group_id is None
    assert samples[1].group_id == 'scene-07'


def test_only_manifest_objects_are_accessed():
    bodies = _bodies({FIRST_JOB: read_fixture('annotated.manifest')})
    gateway, client = _gateway(bodies)

    build_manifest_plan(gateway)

    manifest_uris = {e.output_manifest_uri for e in CATALOG}
    assert set(client.accessed) == manifest_uris
    # No image/mask key was ever touched (the fake would have raised).


def test_failure_rows_are_counted_and_skipped():
    bodies = _bodies({FIRST_JOB: read_fixture('failure-row.manifest')})
    gateway, _ = _gateway(bodies)

    plan = build_manifest_plan(gateway)

    assert plan.total_rows == 4
    assert plan.annotated_rows == 2
    assert plan.skipped_rows == 2


def test_malformed_json_fails():
    bodies = _bodies({FIRST_JOB: read_fixture('malformed.manifest')})
    gateway, _ = _gateway(bodies)

    with pytest.raises(ManifestError, match='malformed JSON'):
        build_manifest_plan(gateway)


def test_missing_required_key_without_failure_shape_fails():
    # source-ref present, ref absent, metadata WITHOUT failure-reason -> validation error.
    bad = json.dumps(
        {
            'source-ref': 's3://synthetic-gta5/img/x.png',
            f'{FIRST_JOB}-ref-metadata': {'type': 'groundtruth/semantic-segmentation'},
        }
    )
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='missing required field'):
        build_manifest_plan(gateway)


def test_unsupported_annotation_type_fails():
    bad = json.dumps(
        {
            'source-ref': 's3://synthetic-gta5/img/x.png',
            f'{FIRST_JOB}-ref': 's3://synthetic-gta5/mask/x.png',
            f'{FIRST_JOB}-ref-metadata': {
                'type': 'groundtruth/object-detection',
                'internal-color-map': {'#ff0000': 'billboard'},
            },
        }
    )
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='unsupported annotation type'):
        build_manifest_plan(gateway)


def test_unexpected_class_fails():
    bad = _row(FIRST_JOB, 'x', color='#123456', cls='car')
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='unexpected class'):
        build_manifest_plan(gateway)


def test_background_only_without_billboard_fails():
    bad = json.dumps(
        {
            'source-ref': 's3://synthetic-gta5/img/x.png',
            f'{FIRST_JOB}-ref': 's3://synthetic-gta5/mask/x.png',
            f'{FIRST_JOB}-ref-metadata': {
                'type': 'groundtruth/semantic-segmentation',
                'internal-color-map': {'#000000': 'BACKGROUND'},
            },
        }
    )
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='trainable class'):
        build_manifest_plan(gateway)


def test_invalid_hex_color_fails():
    bad = _row(FIRST_JOB, 'x', color='#xyz')
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='hex'):
        build_manifest_plan(gateway)


def test_non_s3_source_ref_fails():
    bad = json.dumps(
        {
            'source-ref': 'https://example.com/x.png',
            f'{FIRST_JOB}-ref': 's3://synthetic-gta5/mask/x.png',
            f'{FIRST_JOB}-ref-metadata': {
                'type': 'groundtruth/semantic-segmentation',
                'internal-color-map': {'#ff0000': 'billboard'},
            },
        }
    )
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='source-ref'):
        build_manifest_plan(gateway)


def test_empty_group_id_fails():
    bad = _row(FIRST_JOB, 'x', group='')
    gateway, _ = _gateway(_bodies({FIRST_JOB: bad}))
    with pytest.raises(ManifestError, match='group-id'):
        build_manifest_plan(gateway)


def test_duplicate_uri_across_manifests_fails():
    same = _row(FIRST_JOB, 'dup')  # same source-ref URI reused in second manifest
    other = _row(SECOND_JOB, 'dup')
    gateway, _ = _gateway(_bodies({FIRST_JOB: same, SECOND_JOB: other}))
    with pytest.raises(ManifestError, match='duplicate normalized source-ref URI'):
        build_manifest_plan(gateway)


def test_duplicate_filename_across_manifests_fails():
    # Different URIs (different buckets) but identical final filename segment.
    first = _row(FIRST_JOB, 'shot', img_bucket='bucket-one')
    second = _row(SECOND_JOB, 'shot', img_bucket='bucket-two')
    gateway, _ = _gateway(_bodies({FIRST_JOB: first, SECOND_JOB: second}))
    with pytest.raises(ManifestError, match='duplicate source filename'):
        build_manifest_plan(gateway)


def test_candidates_preserve_within_manifest_order():
    body = '\n'.join(_row(FIRST_JOB, f'n{i}') for i in range(5))
    gateway, _ = _gateway(_bodies({FIRST_JOB: body}))
    plan = build_manifest_plan(gateway)
    assert [s.source_filename for s in plan.candidates_by_manifest[FIRST_JOB]] == [f'n{i}.png' for i in range(5)]
