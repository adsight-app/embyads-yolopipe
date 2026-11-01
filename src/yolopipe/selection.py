"""Deterministic, group-aware bounded selection over a validated ManifestPlan.

Selection forms whole units keyed ``group:<group-id>`` (grouped rows) or
``sample:<sample-id>`` (ungrouped singletons), ranks them by a domain-separated
SHA-256 of the unit key, and adds whole units until the selected sample count
reaches the requested bound ``N`` (``N >= 3``). Only the final group may push the
count past ``N``. With no cap, every annotated sample is selected.
"""

from __future__ import annotations

import hashlib

from .constants import SPLIT_SEED
from .errors import SelectionError
from .manifest import ManifestPlan, SelectionUnit

_SELECT_PREFIX = 'select'
_MIN_SAMPLES = 3


def _rank_key(unit_key: str) -> tuple[str, str]:
    digest = hashlib.sha256(f'{_SELECT_PREFIX}:{SPLIT_SEED}:{unit_key}'.encode('utf-8')).hexdigest()
    return digest, unit_key


def _build_units(plan: ManifestPlan) -> list[SelectionUnit]:
    """Build units in fixed catalog order, grouping by ``group_id`` across manifests.

    Grouped units collect every sample sharing a ``group_id`` (ordered by
    ``sample_id``). Ungrouped samples become singleton ``sample:`` units.
    """
    groups: dict[str, list[str]] = {}
    group_order: list[str] = []
    units: list[SelectionUnit] = []

    for sample in plan.all_candidates:
        if sample.group_id:
            if sample.group_id not in groups:
                groups[sample.group_id] = []
                group_order.append(sample.group_id)
            groups[sample.group_id].append(sample.sample_id)
        else:
            units.append(SelectionUnit(key=f'sample:{sample.sample_id}', sample_ids=[sample.sample_id]))

    for group_id in group_order:
        sample_ids = sorted(groups[group_id])
        units.append(SelectionUnit(key=f'group:{group_id}', sample_ids=sample_ids))

    return units


def select(plan: ManifestPlan, max_samples: int | None = None) -> ManifestPlan:
    """Populate ``plan`` with ranked selection units and the selected sample set.

    Mutates and returns the same ``plan``. ``max_samples`` of ``None`` selects all
    annotated samples; an integer bound requires ``max_samples >= 3``.
    """
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise SelectionError('--max-samples must be an integer')
        if max_samples < _MIN_SAMPLES:
            raise SelectionError(f'--max-samples must be at least {_MIN_SAMPLES}')

    units = _build_units(plan)
    units.sort(key=lambda unit: _rank_key(unit.key))

    selected_ids: list[str] = []
    if max_samples is None:
        for unit in units:
            selected_ids.extend(unit.sample_ids)
    else:
        for unit in units:
            if len(selected_ids) >= max_samples:
                break
            selected_ids.extend(unit.sample_ids)

    plan.selection_units = units
    plan.selected_sample_ids = selected_ids
    plan.effective_max_samples = max_samples
    return plan
