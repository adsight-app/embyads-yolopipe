"""Ground Truth JSONL manifest validation and the resulting ManifestPlan.

This module performs manifest-level validation ONLY. It reads each catalog
manifest body via :meth:`S3Gateway.get_text` and never touches referenced
image or mask objects. It distinguishes annotated rows from explicit Ground
Truth failure rows, derives each sample's canonical identity, and enforces
combined-catalog uniqueness of normalized source URIs and source filenames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import constants
from .catalog import CATALOG, catalog_fingerprint
from .errors import ManifestError, ValidationError
from .models import CatalogEntry, NormalizedSample, S3Uri, sha256_hex

_HEX_COLOR_LEN = 7  # '#rrggbb'
_HEX_DIGITS = frozenset('0123456789abcdefABCDEF')


@dataclass
class SelectionUnit:
    """A selection/split unit: a named group or a singleton sample."""

    key: str  # 'group:<group-id>' or 'sample:<sample-id>'
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class ManifestPlan:
    """Fully validated manifest plan plus (optionally) a bounded selection.

    ``candidates_by_manifest`` preserves original within-manifest order for each
    job, in fixed catalog order. ``selection_units``, ``selected_sample_ids`` and
    ``effective_max_samples`` are populated by :mod:`yolopipe.selection`.
    """

    catalog_fingerprint: str
    total_rows: int
    annotated_rows: int
    skipped_rows: int
    candidates_by_manifest: dict[str, list[NormalizedSample]]
    selection_units: list[SelectionUnit] = field(default_factory=list)
    selected_sample_ids: list[str] = field(default_factory=list)
    effective_max_samples: int | None = None

    @property
    def all_candidates(self) -> list[NormalizedSample]:
        """All annotated candidates concatenated in fixed catalog order."""
        ordered: list[NormalizedSample] = []
        for entry in CATALOG:
            ordered.extend(self.candidates_by_manifest.get(entry.job_name, []))
        return ordered


def _normalize_source_ref(uri: str, *, job_name: str, line_no: int) -> tuple[str, S3Uri]:
    """Validate a source-ref S3 URI and return its canonical string and parsed form."""
    try:
        parsed = S3Uri.parse(uri)
    except ValidationError as exc:
        raise ManifestError(f'{job_name} line {line_no}: invalid source-ref: {exc.message}') from exc
    # Canonical normalized form is the reconstructed s3://bucket/key string.
    return str(parsed), parsed


def _require(row: dict, key: str, *, job_name: str, line_no: int) -> object:
    if key not in row:
        raise ManifestError(f'{job_name} line {line_no}: missing required field {key!r}')
    return row[key]


def _validate_s3_ref(value: object, *, field_name: str, job_name: str, line_no: int) -> S3Uri:
    if not isinstance(value, str):
        raise ManifestError(f'{job_name} line {line_no}: {field_name} must be a string')
    try:
        return S3Uri.parse(value)
    except ValidationError as exc:
        raise ManifestError(f'{job_name} line {line_no}: {field_name} is not a valid S3 URI: {exc.message}') from exc


def _validate_hex_color(value: object, *, job_name: str, line_no: int) -> None:
    if not isinstance(value, str) or len(value) != _HEX_COLOR_LEN or value[0] != '#':
        raise ManifestError(f'{job_name} line {line_no}: color {value!r} is not a #rrggbb hex string')
    if any(ch not in _HEX_DIGITS for ch in value[1:]):
        raise ManifestError(f'{job_name} line {line_no}: color {value!r} has non-hex digits')


def _validate_color_map(color_map: object, *, job_name: str, line_no: int) -> dict:
    """Validate a Ground Truth ``internal-color-map`` and normalize it to ``{hex-color: class-name}``.

    The on-disk Ground Truth schema maps a string label index to an object::

        {"0": {"class-name": "BACKGROUND", "hex-color": "#ffffff", "confidence": 0},
         "1": {"class-name": "billboard",  "hex-color": "#2ca02c", "confidence": 0}}

    Downstream conversion matches mask pixels by hex color and decides trainability
    by class name, so this returns the canonical ``{hex-color: class-name}`` form.
    A single hex color must map to a single class name within one row.
    """
    if not isinstance(color_map, dict) or not color_map:
        raise ManifestError(f'{job_name} line {line_no}: internal-color-map must be a non-empty object')
    normalized: dict[str, str] = {}
    classes: set[str] = set()
    for label_index, entry in color_map.items():
        if not isinstance(entry, dict):
            raise ManifestError(
                f'{job_name} line {line_no}: color-map entry for index {label_index!r} must be an object'
            )
        class_name = entry.get('class-name')
        if not isinstance(class_name, str) or not class_name:
            raise ManifestError(
                f'{job_name} line {line_no}: color-map class-name for index {label_index!r} must be a non-empty string'
            )
        hex_color = entry.get('hex-color')
        _validate_hex_color(hex_color, job_name=job_name, line_no=line_no)
        if hex_color in normalized and normalized[hex_color] != class_name:
            raise ManifestError(
                f'{job_name} line {line_no}: hex-color {hex_color!r} maps to conflicting classes '
                f'{normalized[hex_color]!r} and {class_name!r}'
            )
        normalized[hex_color] = class_name
        classes.add(class_name)
    unknown = classes - {constants.GROUND_TRUTH_CLASS} - set(constants.IGNORED_CLASSES)
    if unknown:
        offending = ', '.join(sorted(unknown))
        raise ManifestError(f'{job_name} line {line_no}: unexpected class(es) in color-map: {offending}')
    if constants.GROUND_TRUTH_CLASS not in classes:
        raise ManifestError(
            f'{job_name} line {line_no}: color-map must contain the trainable class {constants.GROUND_TRUTH_CLASS!r}'
        )
    return normalized


def _validate_group_id(value: object, *, job_name: str, line_no: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f'{job_name} line {line_no}: group-id must be a non-empty string when present')
    return value


def _is_failure_row(row: dict, ref_key: str, meta_key: str) -> bool:
    """Explicit Ground Truth unannotated shape: ref absent AND metadata has failure-reason."""
    if ref_key in row:
        return False
    meta = row.get(meta_key)
    return isinstance(meta, dict) and 'failure-reason' in meta


def _parse_row(
    row: dict,
    *,
    entry: CatalogEntry,
    line_no: int,
) -> NormalizedSample | None:
    """Return a NormalizedSample for an annotated row, or None for a counted failure row."""
    job_name = entry.job_name
    ref_key = f'{job_name}-ref'
    meta_key = f'{job_name}-ref-metadata'

    if _is_failure_row(row, ref_key, meta_key):
        return None

    source_ref = _require(row, 'source-ref', job_name=job_name, line_no=line_no)
    normalized_uri, source_uri = _normalize_source_ref(
        source_ref if isinstance(source_ref, str) else '',
        job_name=job_name,
        line_no=line_no,
    )

    mask_value = _require(row, ref_key, job_name=job_name, line_no=line_no)
    mask_uri = _validate_s3_ref(mask_value, field_name=ref_key, job_name=job_name, line_no=line_no)

    meta = _require(row, meta_key, job_name=job_name, line_no=line_no)
    if not isinstance(meta, dict):
        raise ManifestError(f'{job_name} line {line_no}: {meta_key} must be an object')

    ann_type = meta.get('type')
    if ann_type != constants.ANNOTATION_TYPE:
        raise ManifestError(
            f'{job_name} line {line_no}: unsupported annotation type {ann_type!r}; '
            f'expected {constants.ANNOTATION_TYPE!r}'
        )

    color_map = _validate_color_map(meta.get('internal-color-map'), job_name=job_name, line_no=line_no)

    group_id = _validate_group_id(row.get('group-id'), job_name=job_name, line_no=line_no)

    try:
        extension = '.' + source_uri.filename.rsplit('.', 1)[1] if '.' in source_uri.filename else ''
    except ValidationError as exc:  # no filename segment
        raise ManifestError(f'{job_name} line {line_no}: source-ref has no object filename') from exc
    source_filename = source_uri.filename

    sample_id = sha256_hex(normalized_uri.encode('utf-8'))

    return NormalizedSample(
        sample_id=sample_id,
        job_name=job_name,
        source_image_uri=normalized_uri,
        source_mask_uri=str(mask_uri),
        source_filename=source_filename,
        extension=extension,
        color_map=color_map,
        group_id=group_id,
    )


def _parse_manifest_body(body: str, *, entry: CatalogEntry) -> tuple[list[NormalizedSample], int, int]:
    """Parse one manifest body. Returns (annotated_samples, total_rows, skipped_rows)."""
    samples: list[NormalizedSample] = []
    total = 0
    skipped = 0
    for line_no, raw in enumerate(body.splitlines(), start=1):
        if not raw.strip():
            continue
        total += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f'{entry.job_name} line {line_no}: malformed JSON: {exc.msg}') from exc
        if not isinstance(row, dict):
            raise ManifestError(f'{entry.job_name} line {line_no}: row must be a JSON object')
        sample = _parse_row(row, entry=entry, line_no=line_no)
        if sample is None:
            skipped += 1
        else:
            samples.append(sample)
    return samples, total, skipped


def build_manifest_plan(gateway) -> ManifestPlan:
    """Validate all nine catalog manifests and return an unselected ManifestPlan.

    ``gateway`` is an :class:`yolopipe.storage.S3Gateway`. Only manifest bodies are
    read (via ``get_text``); no referenced image or mask object is accessed.
    """
    candidates_by_manifest: dict[str, list[NormalizedSample]] = {}
    total_rows = 0
    annotated_rows = 0
    skipped_rows = 0

    seen_uris: dict[str, str] = {}
    seen_filenames: dict[str, str] = {}

    for entry in CATALOG:
        manifest_uri = S3Uri.parse(entry.output_manifest_uri)
        body = gateway.get_text(manifest_uri)
        samples, total, skipped = _parse_manifest_body(body, entry=entry)

        for sample in samples:
            if sample.source_image_uri in seen_uris:
                raise ManifestError(
                    f'duplicate normalized source-ref URI {sample.source_image_uri!r} '
                    f'in {entry.job_name} (first seen in {seen_uris[sample.source_image_uri]})'
                )
            if sample.source_filename in seen_filenames:
                raise ManifestError(
                    f'duplicate source filename {sample.source_filename!r} '
                    f'in {entry.job_name} (first seen in {seen_filenames[sample.source_filename]})'
                )
            seen_uris[sample.source_image_uri] = entry.job_name
            seen_filenames[sample.source_filename] = entry.job_name

        candidates_by_manifest[entry.job_name] = samples
        total_rows += total
        annotated_rows += len(samples)
        skipped_rows += skipped

    return ManifestPlan(
        catalog_fingerprint=catalog_fingerprint(),
        total_rows=total_rows,
        annotated_rows=annotated_rows,
        skipped_rows=skipped_rows,
        candidates_by_manifest=candidates_by_manifest,
    )
