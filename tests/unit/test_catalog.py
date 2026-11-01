"""Contract tests for the fixed nine-manifest catalog and its static fingerprint."""

from __future__ import annotations

from yolopipe.catalog import CATALOG, catalog_fingerprint

# Computed once from the canonical JSON of the ordered catalog; asserted as a literal
# so any accidental edit to the catalog is caught immediately.
EXPECTED_FINGERPRINT = '313f2d09c1dedd19d996462310ec81881c6cb065f64673e83a461bbc2ae1b2b6'

EXPECTED_ENTRIES = (
    (
        '20250515-gta5-billboards-01',
        's3://embytest-datasets/batch-aa-2025-05/20250515-gta5-billboards-01/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-02',
        's3://embytest-datasets/batch-ab-2025-05-out/20250521-gta5-billboards-02/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-03',
        's3://embytest-datasets/batch-ac-2025-05/20250521-gta5-billboards-03/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-04',
        's3://embytest-datasets/batch-ad-2025-06/20250521-gta5-billboards-04/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-05',
        's3://embytest-datasets/batch-ae-2025-06/20250521-gta5-billboards-05/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-06',
        's3://embytest-datasets/batch-af-2025-06/20250521-gta5-billboards-06/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-07',
        's3://embytest-datasets/batch-ag-2025-06-gta5-gen2/VovaGTA5/20250521-gta5-billboards-07/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-08',
        's3://embytest-datasets/batch-ah-2025-06-gta5-gen2/20250521-gta5-billboards-08/manifests/output/output.manifest',
    ),
    (
        '20250521-gta5-billboards-09',
        's3://embytest-datasets/batch-ai-2025-06-gta5-gen2/JuliaGTA5/20250521-gta5-billboards-09/manifests/output/output.manifest',
    ),
)


def test_catalog_has_exactly_nine_entries_in_order():
    assert len(CATALOG) == 9
    actual = tuple((e.job_name, e.output_manifest_uri) for e in CATALOG)
    assert actual == EXPECTED_ENTRIES


def test_catalog_excludes_clone_job():
    job_names = {e.job_name for e in CATALOG}
    assert not any('clone' in name for name in job_names)
    assert '20250521-gta5-billboards-06-clone' not in job_names


def test_fingerprint_is_fixed_literal():
    assert catalog_fingerprint() == EXPECTED_FINGERPRINT


def test_fingerprint_is_deterministic():
    assert catalog_fingerprint() == catalog_fingerprint()
