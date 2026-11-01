"""Stage-aware exception hierarchy with stable, credential-safe messages."""

from __future__ import annotations


class YoloPipeError(Exception):
    """Base error. ``stage`` names the pipeline phase; ``exit_code`` is the CLI exit code."""

    stage = 'general'
    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(YoloPipeError):
    stage = 'validation'
    exit_code = 2


class ManifestError(ValidationError):
    stage = 'manifest'


class SelectionError(ValidationError):
    stage = 'selection'


class StorageBoundaryError(YoloPipeError):
    """Attempted write outside the emby-oss/yolopipe boundary, or local path escape."""

    stage = 'storage'
    exit_code = 3


class TransferError(YoloPipeError):
    stage = 'transfer'
    exit_code = 4


class ConversionError(YoloPipeError):
    stage = 'conversion'
    exit_code = 5


class SplitError(YoloPipeError):
    stage = 'split'
    exit_code = 6


class PreflightError(YoloPipeError):
    stage = 'preflight'
    exit_code = 7


class TrainingError(YoloPipeError):
    stage = 'training'
    exit_code = 8


class IntegrityError(YoloPipeError):
    stage = 'integrity'
    exit_code = 9


class PublicationError(YoloPipeError):
    stage = 'publication'
    exit_code = 10


class RunCollisionError(PublicationError):
    """A run/model prefix already exists (finalized or incomplete)."""
