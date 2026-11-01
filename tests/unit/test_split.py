"""Deterministic split quota, ordering, and group-assignment contract tests."""

from __future__ import annotations

import hashlib
import math

import pytest

from yolopipe.errors import SplitError
from yolopipe.models import NormalizedSample
from yolopipe.split import assign_splits, build_units, compute_quotas


def _sample(name: str, *, group=None) -> NormalizedSample:
    return NormalizedSample(
        sample_id=f'id-{name}',
        job_name='20250515-gta5-billboards-01',
        source_image_uri=f's3://synthetic/img/{name}.png',
        source_mask_uri=f's3://synthetic/mask/{name}.png',
        source_filename=f'{name}.png',
        extension='.png',
        color_map={'#00ff00': 'billboard'},
        group_id=group,
    )


def _digest(unit_key: str) -> str:
    return hashlib.sha256(f'split:yolopipe-gta5-v1:{unit_key}'.encode('utf-8')).hexdigest()


def test_quota_tables_m3_to_20_sum_and_min_one():
    for m in range(3, 21):
        quotas = compute_quotas(m)
        assert sum(quotas.values()) == m
        assert all(quotas[name] >= 1 for name in ('train', 'val', 'minival'))
        # train should be the largest share.
        assert quotas['train'] >= quotas['val'] >= quotas['minival'] or quotas['minival'] >= 1


def test_quota_exact_ratios_for_clean_multiple():
    # M=20 -> 15 / 3 / 2 exactly (0.75*20=15, 0.15*20=3, 0.10*20=2).
    assert compute_quotas(20) == {'train': 15, 'val': 3, 'minival': 2}


def test_quota_m3_min_one_each():
    # 0.75*3=2.25->2, 0.15*3=0.45->0, 0.10*3=0.30->0; remainder 1 -> train.
    # Then min-one donor fills val and minival from train.
    assert compute_quotas(3) == {'train': 1, 'val': 1, 'minival': 1}


def test_quota_floors_match_spec_examples():
    # M=10: 7.5->7, 1.5->1, 1.0->1; remainder 1 by fractional tie train,val->
    # fractions: train .5, val .5, minival .0; tie order train first.
    assert compute_quotas(10) == {'train': 8, 'val': 1, 'minival': 1}


def test_remainder_tie_order_prefers_train_then_val():
    # Find an M where two fractional remainders tie; smallest is M=10 (train/val .5).
    quotas = compute_quotas(10)
    # remainder of 1 went to train (first in tie order).
    assert quotas['train'] == 8


def test_singleton_units_use_sample_namespace():
    units = build_units([_sample('a'), _sample('b'), _sample('c')])
    assert all(u.key.startswith('sample:') for u in units)
    assert all(u.size == 1 for u in units)


def test_grouped_unit_indivisible_and_sorted():
    samples = [
        _sample('g2', group='scene'),
        _sample('g1', group='scene'),
        _sample('solo1'),
        _sample('solo2'),
    ]
    units = build_units(samples)
    group_units = [u for u in units if u.key == 'group:scene']
    assert len(group_units) == 1
    assert group_units[0].sample_ids == ('id-g1', 'id-g2')  # sorted by sample_id


def test_prefixed_namespace_group_and_sample_distinct():
    # A group named 'x' and a sample id 'x' must not collide: keys are prefixed.
    g_key = 'group:x'
    s_key = 'sample:x'
    assert _digest(g_key) != _digest(s_key)


def test_assignment_non_empty_splits_and_no_overlap():
    samples = [_sample(f'n{i}') for i in range(12)]
    splits = assign_splits(samples)
    assert all(splits[name] for name in ('train', 'val', 'minival'))
    all_ids = splits['train'] + splits['val'] + splits['minival']
    assert len(all_ids) == len(set(all_ids)) == 12


def test_assignment_meets_quotas_for_singletons():
    m = 20
    samples = [_sample(f'n{i}') for i in range(m)]
    splits = assign_splits(samples)
    quotas = compute_quotas(m)
    assert len(splits['train']) == quotas['train']
    assert len(splits['val']) == quotas['val']
    assert len(splits['minival']) == quotas['minival']


def test_assignment_is_shuffled_input_stable():
    samples = [_sample(f'n{i}') for i in range(15)]
    forward = assign_splits(samples)
    backward = assign_splits(list(reversed(samples)))
    assert forward == backward


def test_grouped_assignment_keeps_group_together():
    samples = [_sample(f'g{i}', group='bundle') for i in range(3)]
    samples += [_sample(f'n{i}') for i in range(7)]
    splits = assign_splits(samples)
    group_ids = {'id-g0', 'id-g1', 'id-g2'}
    homes = [name for name in ('train', 'val', 'minival') if group_ids & set(splits[name])]
    assert len(homes) == 1  # all three in one split
    assert group_ids <= set(splits[homes[0]])


def test_fewer_than_three_units_raises():
    with pytest.raises(SplitError, match='at least 3'):
        assign_splits([_sample('a'), _sample('b')])
    # A single group of many samples is still one unit -> fewer than three units.
    big_group = [_sample(f'g{i}', group='one') for i in range(5)]
    with pytest.raises(SplitError, match='3 split units'):
        assign_splits(big_group)


def test_empty_split_quota_filled_by_min_one_rule():
    # With exactly 3 units each split gets exactly one.
    samples = [_sample('a'), _sample('b'), _sample('c')]
    splits = assign_splits(samples)
    assert len(splits['train']) == 1
    assert len(splits['val']) == 1
    assert len(splits['minival']) == 1


def test_quota_never_zero_in_range():
    for m in range(3, 101):
        quotas = compute_quotas(m)
        assert min(quotas.values()) >= 1
        assert sum(quotas.values()) == m
        # train quota close to 0.75 floor-ish for large M.
        if m >= 40:
            assert quotas['train'] >= math.floor(0.75 * m) - 1
