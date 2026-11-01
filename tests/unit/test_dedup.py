"""Per-manifest perceptual deduplication contract tests."""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image

from yolopipe import constants
from yolopipe.catalog import CATALOG
from yolopipe.dedup import _compute_hash_int, deduplicate
from yolopipe.models import NormalizedSample

JOB_A = CATALOG[0].job_name
JOB_B = CATALOG[1].job_name


def _sample(name: str, *, job: str = JOB_A) -> NormalizedSample:
    return NormalizedSample(
        sample_id=f'id-{name}',
        job_name=job,
        source_image_uri=f's3://src/img/{name}.png',
        source_mask_uri=f's3://src/mask/{name}.png',
        source_filename=f'{name}.png',
        extension='.png',
        color_map={'#ff0000': 'billboard'},
    )


def _save(path, array) -> None:
    Image.fromarray(array.astype('uint8'), mode='L').save(path)


def _gradient(seed: int, size: int = 64) -> np.ndarray:
    # Distinct, smooth, mutually low-correlation structures per seed.
    xs = np.linspace(0, 255, size)
    ys = np.linspace(0, 255, size)
    gx, gy = np.meshgrid(xs, ys)
    patterns = [
        gx,
        gy,
        255 - gx,
        (gx + gy) / 2,
        np.abs(gx - gy),
    ]
    return patterns[seed % len(patterns)].astype('uint8')


def _noise(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(size, size), dtype='uint8')


def _write_samples(tmp_path, specs):
    """specs: list of (name, array). Returns (samples, image_paths)."""
    samples = []
    image_paths = {}
    for name, array in specs:
        s = _sample(name)
        path = tmp_path / f'{name}.png'
        _save(path, array)
        samples.append(s)
        image_paths[s.sample_id] = path
    return samples, image_paths


def test_whash_int_conversion_matches_imagehash(tmp_path):
    import imagehash

    path = tmp_path / 'x.png'
    _save(path, _gradient(0))
    with Image.open(path) as img:
        expected = int(str(imagehash.whash(img, hash_size=16)), 16)
    assert _compute_hash_int(path) == expected


def test_zero_rows_pass_through(tmp_path):
    survivors, meta = deduplicate({JOB_A: []}, {})
    assert survivors == []
    # Empty manifests are not recorded.
    assert JOB_A not in meta['per_manifest']


def test_single_row_passes_through(tmp_path):
    samples, paths = _write_samples(tmp_path, [('a', _gradient(0))])
    survivors, meta = deduplicate({JOB_A: samples}, paths)
    assert [s.sample_id for s in survivors] == ['id-a']
    assert survivors[0].image_hash_int == _compute_hash_int(paths['id-a'])
    assert meta['per_manifest'][JOB_A] == {
        'analyzed': 1,
        'retained': 1,
        'discarded': 0,
        'survivor_sample_ids': ['id-a'],
    }


def test_unrelated_images_all_retained_in_hash_order(tmp_path):
    specs = [('a', _gradient(0)), ('b', _gradient(1)), ('c', _gradient(2))]
    samples, paths = _write_samples(tmp_path, specs)
    survivors, meta = deduplicate({JOB_A: samples}, paths)

    expected_order = sorted((s.sample_id for s in samples), key=lambda sid: _compute_hash_int(paths[sid]))
    assert [s.sample_id for s in survivors] == expected_order
    assert meta['per_manifest'][JOB_A]['retained'] == 3
    assert meta['per_manifest'][JOB_A]['discarded'] == 0


def test_identical_pair_deduplicated(tmp_path):
    arr = _gradient(0)
    specs = [('a', arr), ('b', arr.copy()), ('c', _gradient(2))]
    samples, paths = _write_samples(tmp_path, specs)
    survivors, meta = deduplicate({JOB_A: samples}, paths)

    ids = {s.sample_id for s in survivors}
    # One of the identical pair is removed; the unrelated image survives.
    assert 'id-c' in ids
    assert len(survivors) == 2
    assert meta['per_manifest'][JOB_A]['discarded'] == 1


def test_stable_equal_hash_order_keeps_row_order(tmp_path, monkeypatch):
    # Force equal hashes so the stable sort must preserve original row order.
    import yolopipe.dedup as dedup_mod

    specs = [('a', _gradient(0)), ('b', _gradient(1)), ('c', _gradient(2))]
    samples, paths = _write_samples(tmp_path, specs)
    monkeypatch.setattr(dedup_mod, '_compute_hash_int', lambda _p: 42)
    # Also prevent any accidental matching so all survive.
    monkeypatch.setattr(dedup_mod, 'match_images', lambda *a, **k: False)

    survivors, _ = deduplicate({JOB_A: samples}, paths)
    assert [s.sample_id for s in survivors] == ['id-a', 'id-b', 'id-c']


def test_lower_hash_wins_as_first_anchor(tmp_path, monkeypatch):
    import yolopipe.dedup as dedup_mod

    specs = [('a', _gradient(0)), ('b', _gradient(1))]
    samples, paths = _write_samples(tmp_path, specs)
    # b has the lower hash, so it becomes the first anchor and survives the pair.
    hashes = {paths['id-a']: 100, paths['id-b']: 10}
    monkeypatch.setattr(dedup_mod, '_compute_hash_int', lambda p: hashes[p])
    monkeypatch.setattr(dedup_mod, 'match_images', lambda *a, **k: True)

    survivors, _ = deduplicate({JOB_A: samples}, paths)
    assert [s.sample_id for s in survivors] == ['id-b']


def test_last_retained_adjacency_non_transitivity(tmp_path, monkeypatch):
    import yolopipe.dedup as dedup_mod

    specs = [('a', _gradient(0)), ('b', _gradient(1)), ('c', _gradient(2))]
    samples, paths = _write_samples(tmp_path, specs)
    # Hash order a < b < c.
    hashes = {paths['id-a']: 1, paths['id-b']: 2, paths['id-c']: 3}
    monkeypatch.setattr(dedup_mod, '_compute_hash_int', lambda p: hashes[p])

    # Only adjacent pairs match: a~b and b~c, but a!~c. Non-transitive.
    def fake_match(anchor_path, candidate_path, threshold):
        pair = {anchor_path, candidate_path}
        if pair == {paths['id-a'], paths['id-b']}:
            return True
        if pair == {paths['id-b'], paths['id-c']}:
            return True
        return False

    monkeypatch.setattr(dedup_mod, 'match_images', fake_match)

    survivors, _ = deduplicate({JOB_A: samples}, paths)
    # a retained; b matches a -> discarded (anchor stays a); c vs a -> no match
    # (a!~c) so c survives. Comparing only to the LAST retained (a), not b.
    assert [s.sample_id for s in survivors] == ['id-a', 'id-c']


def test_same_pair_removed_in_one_manifest_but_retained_across_two(tmp_path):
    arr = _gradient(0)
    # One manifest holding both: one is removed.
    s1 = _sample('x')
    s2 = _sample('y')
    p1 = tmp_path / 'x.png'
    p2 = tmp_path / 'y.png'
    _save(p1, arr)
    _save(p2, arr.copy())
    paths = {s1.sample_id: p1, s2.sample_id: p2}

    survivors_one, _ = deduplicate({JOB_A: [s1, s2]}, paths)
    assert len(survivors_one) == 1

    # Same pair split across two different manifests: both retained (never
    # compared across jobs).
    s2b = dataclasses.replace(s2, job_name=JOB_B)
    survivors_two, meta = deduplicate({JOB_A: [s1], JOB_B: [s2b]}, paths)
    assert {s.sample_id for s in survivors_two} == {'id-x', 'id-y'}
    assert meta['per_manifest'][JOB_A]['retained'] == 1
    assert meta['per_manifest'][JOB_B]['retained'] == 1


def test_survivors_concatenated_in_catalog_order(tmp_path):
    sa = _sample('a', job=JOB_A)
    sb = _sample('b', job=JOB_B)
    pa = tmp_path / 'a.png'
    pb = tmp_path / 'b.png'
    _save(pa, _gradient(0))
    _save(pb, _gradient(1))
    paths = {sa.sample_id: pa, sb.sample_id: pb}

    # Provide JOB_B first in the mapping; survivors must still follow catalog order.
    survivors, _ = deduplicate({JOB_B: [sb], JOB_A: [sa]}, paths)
    assert [s.job_name for s in survivors] == [JOB_A, JOB_B]


def test_inputs_are_not_mutated(tmp_path):
    samples, paths = _write_samples(tmp_path, [('a', _gradient(0))])
    before = dataclasses.asdict(samples[0])
    survivors, _ = deduplicate({JOB_A: samples}, paths)
    assert samples[0].image_hash_int is None
    assert dataclasses.asdict(samples[0]) == before
    # Survivor is a distinct object carrying the hash.
    assert survivors[0] is not samples[0]
    assert survivors[0].image_hash_int is not None


def test_metadata_records_algorithm_constants(tmp_path):
    samples, paths = _write_samples(tmp_path, [('a', _gradient(0)), ('b', _gradient(1))])
    survivors, meta = deduplicate({JOB_A: samples}, paths)
    assert meta['algorithm_version'] == constants.DEDUP_ALGORITHM_VERSION
    assert meta['hash'] == 'whash'
    assert meta['hash_size'] == constants.DEDUP_HASH_SIZE == 16
    assert meta['matcher'] == 'TM_CCOEFF_NORMED'
    assert meta['threshold'] == constants.DEDUP_THRESHOLD == 0.85
    assert meta['resize'] == constants.DEDUP_RESIZE == 256
    assert isinstance(meta['duration_seconds'], float)
    assert meta['survivor_sample_ids'] == [s.sample_id for s in survivors]
