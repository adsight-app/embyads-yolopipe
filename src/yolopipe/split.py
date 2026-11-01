"""Deterministic classic 75/15/10 split with group-aware greedy assignment.

Assignment is seeded by the fixed constant ``yolopipe-gta5-v1`` and the SHA-256
key ``split:yolopipe-gta5-v1:<unit-key>`` where ``<unit-key>`` is
``group:<group_id>`` for grouped units and ``sample:<sample_id>`` for ungrouped
singletons. For ``M >= 3`` survivors the target quotas are the floors of
``(0.75M, 0.15M, 0.10M)`` with the remainder distributed by descending
fractional part (ties ``train`` > ``val`` > ``minival``), followed by a
minimum-one donor rule. Units are ordered by the ``split:`` digest tuple and
assigned greedily: when the remaining unit count equals the number of empty
splits, the unit goes to the first empty split in ``train`` > ``val`` >
``minival`` order; otherwise it goes to the split with the greatest remaining
deficit (``target - assigned``) using the same tie order. Grouped units are
indivisible. Fewer than three units, an empty split, or a sample appearing in
two splits is a :class:`SplitError`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import constants
from .errors import SplitError
from .models import NormalizedSample


@dataclass(frozen=True)
class SplitUnit:
    """An indivisible split unit: a group of samples or a single sample."""

    key: str  # 'group:<group_id>' or 'sample:<sample_id>'
    sample_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.sample_ids)


def _digest(unit_key: str) -> str:
    return hashlib.sha256(f'split:{constants.SPLIT_SEED}:{unit_key}'.encode('utf-8')).hexdigest()


def build_units(samples: list[NormalizedSample]) -> list[SplitUnit]:
    """Group samples into indivisible units: grouped samples ordered by sample_id."""
    groups: dict[str, list[str]] = {}
    singletons: list[SplitUnit] = []
    group_order: list[str] = []

    for sample in samples:
        if sample.group_id:
            if sample.group_id not in groups:
                groups[sample.group_id] = []
                group_order.append(sample.group_id)
            groups[sample.group_id].append(sample.sample_id)
        else:
            singletons.append(SplitUnit(key=f'sample:{sample.sample_id}', sample_ids=(sample.sample_id,)))

    units: list[SplitUnit] = []
    for group_id in group_order:
        ids = tuple(sorted(groups[group_id]))
        units.append(SplitUnit(key=f'group:{group_id}', sample_ids=ids))
    units.extend(singletons)
    return units


def compute_quotas(total: int) -> dict[str, int]:
    """Largest-remainder quotas for ``total`` samples with the minimum-one donor rule."""
    if total < 3:
        raise SplitError(f'need at least 3 samples to split into three non-empty parts, got {total}')

    raw = {name: total * constants.SPLIT_RATIOS[name] for name in constants.SPLIT_ORDER}
    quotas = {name: int(raw[name]) for name in constants.SPLIT_ORDER}  # floors
    remainder = total - sum(quotas.values())

    # Distribute remainder by descending fractional part; ties by SPLIT_ORDER.
    order_index = {name: i for i, name in enumerate(constants.SPLIT_ORDER)}
    by_fraction = sorted(
        constants.SPLIT_ORDER,
        key=lambda name: (-(raw[name] - int(raw[name])), order_index[name]),
    )
    for i in range(remainder):
        quotas[by_fraction[i]] += 1

    # Minimum-one donor rule: while any quota is zero, increment it and
    # decrement the largest quota greater than 1 (tie order SPLIT_ORDER).
    while any(quotas[name] == 0 for name in constants.SPLIT_ORDER):
        zero_name = next(name for name in constants.SPLIT_ORDER if quotas[name] == 0)
        donor = max(
            (name for name in constants.SPLIT_ORDER if quotas[name] > 1),
            key=lambda name: (quotas[name], -order_index[name]),
        )
        quotas[zero_name] += 1
        quotas[donor] -= 1

    return quotas


def assign_splits(samples: list[NormalizedSample]) -> dict[str, list[str]]:
    """Assign samples to train/val/minival deterministically. Raises SplitError on failure."""
    units = build_units(samples)
    if len(units) < 3:
        raise SplitError(f'need at least 3 split units for non-empty train/val/minival, got {len(units)}')

    total = sum(unit.size for unit in units)
    targets = compute_quotas(total)

    ordered = sorted(units, key=lambda u: (_digest(u.key), u.key))

    assigned: dict[str, int] = {name: 0 for name in constants.SPLIT_ORDER}
    splits: dict[str, list[str]] = {name: [] for name in constants.SPLIT_ORDER}
    order_index = {name: i for i, name in enumerate(constants.SPLIT_ORDER)}

    remaining_units = len(ordered)
    for unit in ordered:
        empty_splits = [name for name in constants.SPLIT_ORDER if not splits[name]]
        if remaining_units == len(empty_splits) and empty_splits:
            chosen = empty_splits[0]
        else:
            chosen = max(
                constants.SPLIT_ORDER,
                key=lambda name: (targets[name] - assigned[name], -order_index[name]),
            )
        splits[chosen].extend(unit.sample_ids)
        assigned[chosen] += unit.size
        remaining_units -= 1

    _validate_splits(splits)
    return splits


def _validate_splits(splits: dict[str, list[str]]) -> None:
    for name in constants.SPLIT_ORDER:
        if not splits[name]:
            raise SplitError(f'split {name!r} is empty after assignment')
    seen: dict[str, str] = {}
    for name in constants.SPLIT_ORDER:
        for sample_id in splits[name]:
            if sample_id in seen:
                raise SplitError(f'sample {sample_id!r} appears in multiple splits ({seen[sample_id]} and {name})')
            seen[sample_id] = name
