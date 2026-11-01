"""Repository-contract tests: metadata, isolation, platform-neutral artifacts.

These assert the standalone-repository guarantees from the spec (R1, R9, and the
isolation acceptance criteria) without requiring AWS or a GPU.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'src' / 'yolopipe'

EXPECTED_RUNTIME_DEPS = {
    'ultralytics': '8.4.33',
    'torch': '2.10.0',
    'torchvision': '0.25.0',
    'opencv-python': '4.11.0.86',
    'pillow': '12.2.0',
    'imagehash': '4.3.2',
    'boto3': '1.40.16',
    'click': '8.1.8',
    'pyyaml': '6.0.2',
}

FORBIDDEN_SUBSTRINGS = (
    'embyolotrainer',
    'embytest-models',
    'dagster',
    'dask',
    'telegram',
    'describe_labeling_job',
    'rf-detr',
    'codeartifact',
)


def _pyproject() -> dict:
    with open(REPO / 'pyproject.toml', 'rb') as fh:
        return tomllib.load(fh)


def test_required_files_exist():
    assert (REPO / 'pyproject.toml').is_file()
    assert (REPO / 'README.md').is_file()
    assert (REPO / 'LICENSE').is_file()
    assert (REPO / 'uv.lock').is_file()


def test_project_identity_and_license():
    proj = _pyproject()['project']
    assert proj['name'] == 'embyads-yolopipe'
    assert proj['license'] == 'AGPL-3.0-only'
    assert proj['scripts']['yolopipe'] == 'yolopipe.cli:main'
    assert proj['requires-python'] == '>=3.12,<3.13'


def test_dependencies_are_exactly_pinned():
    deps = _pyproject()['project']['dependencies']
    pinned = {}
    for spec in deps:
        name, _, version = spec.partition('==')
        pinned[name] = version
    assert pinned == EXPECTED_RUNTIME_DEPS


def test_pytorch_index_is_public_and_explicit():
    proj = _pyproject()
    indexes = proj['tool']['uv']['index']
    pytorch = [i for i in indexes if i['name'] == 'pytorch-cu130']
    assert len(pytorch) == 1
    assert pytorch[0]['url'] == 'https://download.pytorch.org/whl/cu130'
    assert pytorch[0]['explicit'] is True
    # No private index host appears anywhere.
    raw = (REPO / 'pyproject.toml').read_text(encoding='utf-8').lower()
    assert 'codeartifact' not in raw
    assert 'amazonaws.com' not in raw


def test_no_ci_or_publishing_config():
    assert not (REPO / '.github').exists()
    assert not (REPO / '.gitlab-ci.yml').exists()
    assert not (REPO / 'azure-pipelines.yml').exists()
    # No build/publish automation scripts.
    for bad in ('publish.sh', 'release.sh', 'Makefile'):
        assert not (REPO / bad).exists()


def test_single_user_facing_markdown_document():
    tracked = [
        p
        for p in REPO.rglob('*.md')
        if not any(part in {'.venv', '.git', '.pytest_cache', '.ruff_cache'} for part in p.parts)
    ]
    assert tracked == [REPO / 'README.md']


def test_source_tree_has_no_forbidden_identifiers():
    offenders = []
    for path in SRC.rglob('*.py'):
        text = path.read_text(encoding='utf-8').lower()
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                offenders.append(f'{path.name}:{needle}')
    assert offenders == [], offenders


def test_readme_and_license_have_no_legacy_bucket():
    for name in ('README.md', 'LICENSE'):
        text = (REPO / name).read_text(encoding='utf-8')
        assert 'embytest-models' not in text
        assert 'embyolotrainer' not in text


def test_package_imports_without_torch_or_ultralytics(monkeypatch):
    """The package and CLI must import without importing torch/ultralytics eagerly."""
    for mod in list(sys.modules):
        if mod == 'torch' or mod.startswith('torch.') or mod == 'ultralytics' or mod.startswith('ultralytics.'):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    import importlib

    import yolopipe.cli

    importlib.reload(yolopipe.cli)
    # Importing the CLI must not have pulled in torch/ultralytics.
    assert 'torch' not in sys.modules
    assert 'ultralytics' not in sys.modules


@pytest.mark.parametrize('module', ['cli', 'pipeline', 'manifest', 'dataset', 'trainer', 'preflight'])
def test_core_modules_import(module):
    import importlib

    importlib.import_module(f'yolopipe.{module}')
