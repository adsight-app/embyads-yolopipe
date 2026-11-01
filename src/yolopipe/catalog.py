"""The fixed nine-manifest GTA5 catalog and its static fingerprint."""

from __future__ import annotations

from yolopipe.models import CatalogEntry, canonical_json, sha256_hex

CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        '20250515-gta5-billboards-01',
        's3://embytest-datasets/batch-aa-2025-05/20250515-gta5-billboards-01/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-02',
        's3://embytest-datasets/batch-ab-2025-05-out/20250521-gta5-billboards-02/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-03',
        's3://embytest-datasets/batch-ac-2025-05/20250521-gta5-billboards-03/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-04',
        's3://embytest-datasets/batch-ad-2025-06/20250521-gta5-billboards-04/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-05',
        's3://embytest-datasets/batch-ae-2025-06/20250521-gta5-billboards-05/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-06',
        's3://embytest-datasets/batch-af-2025-06/20250521-gta5-billboards-06/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-07',
        's3://embytest-datasets/batch-ag-2025-06-gta5-gen2/VovaGTA5/20250521-gta5-billboards-07/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-08',
        's3://embytest-datasets/batch-ah-2025-06-gta5-gen2/20250521-gta5-billboards-08/manifests/output/output.manifest',
    ),
    CatalogEntry(
        '20250521-gta5-billboards-09',
        's3://embytest-datasets/batch-ai-2025-06-gta5-gen2/JuliaGTA5/20250521-gta5-billboards-09/manifests/output/output.manifest',
    ),
)


def catalog_fingerprint() -> str:
    """SHA-256 of canonical JSON of the ordered job/manifest records. Fully static."""
    payload = [{'job_name': e.job_name, 'output_manifest_uri': e.output_manifest_uri} for e in CATALOG]
    return sha256_hex(canonical_json(payload))
