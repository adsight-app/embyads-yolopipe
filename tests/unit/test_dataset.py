"""Portable materialization, serialization, and dataset-id fingerprint tests."""

from __future__ import annotations

import os

import cv2
import numpy as np

from yolopipe.conversion import convert_all
from yolopipe.dataset import (
    compute_dataset_id,
    materialize_dataset,
    persisted_data_yaml_bytes,
    render_runtime_data_yaml,
)
from yolopipe.models import NormalizedSample
from yolopipe.split import assign_splits

BILLBOARD_BGR = (0, 255, 0)


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


def _make_cached_images(tmp_path, samples):
    cache = tmp_path / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    mask_paths = {}
    for i, s in enumerate(samples):
        img = np.zeros((40, 50, 3), dtype=np.uint8)
        mask = np.zeros((40, 50, 3), dtype=np.uint8)
        # give one sample a real contour, rest empty, deterministically
        if i % 2 == 0:
            mask[5:25, 5:35] = BILLBOARD_BGR
        img_p = cache / f'{s.sample_id}.png'
        mask_p = cache / f'{s.sample_id}_mask.png'
        cv2.imwrite(str(img_p), img)
        cv2.imwrite(str(mask_p), mask)
        image_paths[s.sample_id] = img_p
        mask_paths[s.sample_id] = mask_p
    return image_paths, mask_paths


def _prepare(tmp_path, samples, *, run_id='run-x', max_samples=None):
    image_paths, mask_paths = _make_cached_images(tmp_path, samples)
    conv = convert_all(samples, image_paths=image_paths, mask_paths=mask_paths)
    splits = assign_splits(samples)
    dataset_root = tmp_path / 'data' / run_id / 'dataset'
    return materialize_dataset(
        dataset_root=dataset_root,
        samples=samples,
        labels=conv.labels,
        splits=splits,
        cached_image_paths=image_paths,
        catalog_fingerprint='catfp',
        max_samples=max_samples,
        selected_sample_ids=[s.sample_id for s in samples],
        dedup_stats={'retained': len(samples)},
        conversion_stats=conv.stats,
    )


def test_persisted_data_yaml_exact_bytes():
    expected = b"path: '.'\ntrain: train.txt\nval: val.txt\nminival: minival.txt\nnames:\n  0: gta5-billboard\n"
    assert persisted_data_yaml_bytes() == expected


def test_materialize_creates_parallel_trees_and_posix_split_paths(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    result = _prepare(tmp_path, samples)

    root = result.dataset_root
    assert (root / 'images').is_dir()
    assert (root / 'labels').is_dir()
    assert (root / 'data.yaml').read_bytes() == persisted_data_yaml_bytes()

    for name in ('train', 'val', 'minival'):
        text = (root / f'{name}.txt').read_text()
        for line in text.splitlines():
            assert line.startswith('./images/')
            assert '\\' not in line  # POSIX separators only
            assert line.endswith('.png')


def test_each_sample_has_image_and_label_file(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    result = _prepare(tmp_path, samples)
    root = result.dataset_root
    for s in samples:
        assert (root / 'images' / f'{s.sample_id}.png').exists()
        assert (root / 'labels' / f'{s.sample_id}.txt').exists()


def test_dataset_id_stable_across_run_id_and_host(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    r1 = _prepare(tmp_path / 'a', samples, run_id='run-one')
    r2 = _prepare(tmp_path / 'b', samples, run_id='run-two')
    assert r1.dataset_id == r2.dataset_id


def test_dataset_id_changes_when_cap_changes(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    r1 = _prepare(tmp_path / 'a', samples, max_samples=None)
    r2 = _prepare(tmp_path / 'b', samples, max_samples=5)
    assert r1.dataset_id != r2.dataset_id


def test_dataset_id_changes_when_dedup_survivors_change(tmp_path):
    base = compute_dataset_id(
        catalog_fingerprint='fp',
        max_samples=None,
        selected_sample_ids=['a', 'b', 'c'],
        dedup_stats={},
        survivor_ids=['a', 'b', 'c'],
        conversion_stats={},
        split_seed='yolopipe-gta5-v1',
        split_assignments={'train': ['a'], 'val': ['b'], 'minival': ['c']},
        class_names={0: 'gta5-billboard'},
        image_rel_paths={},
        label_rel_paths={},
        file_checksums={},
    )
    changed = compute_dataset_id(
        catalog_fingerprint='fp',
        max_samples=None,
        selected_sample_ids=['a', 'b', 'c'],
        dedup_stats={},
        survivor_ids=['a', 'b'],  # one fewer survivor
        conversion_stats={},
        split_seed='yolopipe-gta5-v1',
        split_assignments={'train': ['a'], 'val': ['b'], 'minival': ['c']},
        class_names={0: 'gta5-billboard'},
        image_rel_paths={},
        label_rel_paths={},
        file_checksums={},
    )
    assert base != changed


def test_dataset_id_changes_when_split_assignments_change():
    kwargs = dict(
        catalog_fingerprint='fp',
        max_samples=None,
        selected_sample_ids=['a', 'b', 'c'],
        dedup_stats={},
        survivor_ids=['a', 'b', 'c'],
        conversion_stats={},
        split_seed='yolopipe-gta5-v1',
        class_names={0: 'gta5-billboard'},
        image_rel_paths={},
        label_rel_paths={},
        file_checksums={},
    )
    a = compute_dataset_id(split_assignments={'train': ['a'], 'val': ['b'], 'minival': ['c']}, **kwargs)
    b = compute_dataset_id(split_assignments={'train': ['b'], 'val': ['a'], 'minival': ['c']}, **kwargs)
    assert a != b


def test_dataset_id_changes_when_checksums_change():
    kwargs = dict(
        catalog_fingerprint='fp',
        max_samples=None,
        selected_sample_ids=['a'],
        dedup_stats={},
        survivor_ids=['a'],
        conversion_stats={},
        split_seed='yolopipe-gta5-v1',
        split_assignments={'train': ['a'], 'val': ['b'], 'minival': ['c']},
        class_names={0: 'gta5-billboard'},
        image_rel_paths={},
        label_rel_paths={},
    )
    a = compute_dataset_id(file_checksums={'images/a.png': 'aaa'}, **kwargs)
    b = compute_dataset_id(file_checksums={'images/a.png': 'bbb'}, **kwargs)
    assert a != b


def test_runtime_yaml_has_native_absolute_path_and_leaves_persisted_intact(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    result = _prepare(tmp_path, samples)
    before = (result.dataset_root / 'data.yaml').read_bytes()

    runtime = render_runtime_data_yaml(result.dataset_root)
    native = str(result.dataset_root.resolve())
    assert runtime.splitlines()[0] == f'path: {native}'
    assert os.path.isabs(native)

    after = (result.dataset_root / 'data.yaml').read_bytes()
    assert after == before == persisted_data_yaml_bytes()


def test_label_files_match_conversion_bytes(tmp_path):
    samples = [_sample(f'n{i}') for i in range(6)]
    image_paths, mask_paths = _make_cached_images(tmp_path, samples)
    conv = convert_all(samples, image_paths=image_paths, mask_paths=mask_paths)
    splits = assign_splits(samples)
    root = tmp_path / 'ds'
    materialize_dataset(
        dataset_root=root,
        samples=samples,
        labels=conv.labels,
        splits=splits,
        cached_image_paths=image_paths,
        catalog_fingerprint='fp',
        max_samples=None,
        selected_sample_ids=[s.sample_id for s in samples],
        dedup_stats={},
        conversion_stats=conv.stats,
    )
    for s in samples:
        on_disk = (root / 'labels' / f'{s.sample_id}.txt').read_bytes()
        assert on_disk == conv.labels[s.sample_id].label_bytes
