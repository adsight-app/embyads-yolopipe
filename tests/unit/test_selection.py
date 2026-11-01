"""Deterministic group-aware bounded selection contract tests."""

from __future__ import annotations

import hashlib

import pytest

from yolopipe.catalog import CATALOG
from yolopipe.errors import SelectionError
from yolopipe.manifest import ManifestPlan
from yolopipe.models import NormalizedSample
from yolopipe.selection import select

FIRST_JOB = CATALOG[0].job_name
SECOND_JOB = CATALOG[1].job_name


def _sample(name: str, *, job=FIRST_JOB, group=None) -> NormalizedSample:
    return NormalizedSample(
        sample_id=f'id-{name}',
        job_name=job,
        source_image_uri=f's3://synthetic-gta5/img/{name}.png',
        source_mask_uri=f's3://synthetic-gta5/mask/{name}.png',
        source_filename=f'{name}.png',
        extension='.png',
        color_map={'#ff0000': 'billboard'},
        group_id=group,
    )


def _plan(candidates_by_manifest: dict[str, list[NormalizedSample]]) -> ManifestPlan:
    return ManifestPlan(
        catalog_fingerprint='fp',
        total_rows=sum(len(v) for v in candidates_by_manifest.values()),
        annotated_rows=sum(len(v) for v in candidates_by_manifest.values()),
        skipped_rows=0,
        candidates_by_manifest=candidates_by_manifest,
    )


def _rank(unit_key: str) -> str:
    return hashlib.sha256(f'select:yolopipe-gta5-v1:{unit_key}'.encode('utf-8')).hexdigest()


def test_no_cap_selects_all():
    plan = _plan({FIRST_JOB: [_sample(f'n{i}') for i in range(6)]})
    select(plan, None)
    assert len(plan.selected_sample_ids) == 6
    assert plan.effective_max_samples is None


def test_requires_integer_n_at_least_three():
    plan = _plan({FIRST_JOB: [_sample(f'n{i}') for i in range(6)]})
    for bad in (2, 0, -1):
        with pytest.raises(SelectionError, match='at least 3'):
            select(plan, bad)
    with pytest.raises(SelectionError, match='integer'):
        select(plan, True)  # bool is not an acceptable integer
    with pytest.raises(SelectionError, match='integer'):
        select(plan, 3.0)


def test_singleton_units_use_sample_namespace_and_rank_order():
    samples = [_sample(f'n{i}') for i in range(6)]
    plan = _plan({FIRST_JOB: samples})
    select(plan, None)

    keys = [u.key for u in plan.selection_units]
    assert all(k.startswith('sample:') for k in keys)
    # Units are ranked by (sha256('select:...:<key>'), key) ascending.
    expected = sorted((f'sample:{s.sample_id}' for s in samples), key=lambda k: (_rank(k), k))
    assert keys == expected


def test_grouped_unit_uses_group_namespace_and_is_indivisible():
    samples = [
        _sample('g1', group='scene'),
        _sample('g2', group='scene'),
        _sample('solo'),
    ]
    plan = _plan({FIRST_JOB: samples})
    select(plan, None)

    group_units = [u for u in plan.selection_units if u.key == 'group:scene']
    assert len(group_units) == 1
    assert group_units[0].sample_ids == sorted(['id-g1', 'id-g2'])


def test_selection_is_manifest_order_independent():
    a = [_sample(f'n{i}') for i in range(4)]
    b = [_sample(f'm{i}', job=SECOND_JOB) for i in range(4)]

    plan_ab = _plan({FIRST_JOB: a, SECOND_JOB: b})
    plan_ba = _plan({SECOND_JOB: b, FIRST_JOB: a})
    select(plan_ab, 5)
    select(plan_ba, 5)

    assert plan_ab.selected_sample_ids == plan_ba.selected_sample_ids


def test_cap_allows_only_final_group_overshoot():
    # One group of 4 ranked first so it overshoots a cap of 3 by whole-unit rule.
    # Build enough singletons + a group; verify total >= N and the overshoot
    # comes from a single group unit.
    samples = [_sample(f's{i}') for i in range(10)]
    samples += [_sample(f'grp{i}', group='bundle') for i in range(4)]
    plan = _plan({FIRST_JOB: samples})
    select(plan, 3)

    assert len(plan.selected_sample_ids) >= 3
    # Reconstruct selection the same way to confirm whole-unit semantics.
    count = 0
    for unit in plan.selection_units:
        if count >= 3:
            break
        count += len(unit.sample_ids)
    assert count == len(plan.selected_sample_ids)


def test_cap_exact_with_singletons():
    plan = _plan({FIRST_JOB: [_sample(f'n{i}') for i in range(10)]})
    select(plan, 4)
    assert len(plan.selected_sample_ids) == 4
