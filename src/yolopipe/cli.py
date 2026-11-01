"""The ``yolopipe`` command-line interface.

Four subcommands — ``validate``, ``prepare``, ``train``, ``visualize`` — map to
the pipeline service. AWS profile/region default to ``embytest``/``eu-west-1``
and may be overridden. Stage errors map to stable, non-zero exit codes with
credential-safe messages; a non-ready GPU is informational for plain
``validate`` and a hard failure only for ``validate --require-gpu`` or ``train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import constants, reporting
from .errors import YoloPipeError
from .storage import S3Gateway


def _make_gateway(profile: str, region: str) -> S3Gateway:
    """Build an S3 gateway from a boto3 session. Imported lazily for fast --help."""
    import boto3  # noqa: PLC0415

    session = boto3.Session(profile_name=profile, region_name=region)
    return S3Gateway(session.client('s3'))


@click.group()
@click.option('--aws-profile', default=constants.DEFAULT_AWS_PROFILE, show_default=True)
@click.option('--aws-region', default=constants.DEFAULT_AWS_REGION, show_default=True)
@click.pass_context
def cli(ctx: click.Context, aws_profile: str, aws_region: str) -> None:
    """Standalone GTA5 YOLO segmentation pipeline."""
    ctx.ensure_object(dict)
    ctx.obj['aws_profile'] = aws_profile
    ctx.obj['aws_region'] = aws_region


def _gateway(ctx: click.Context) -> S3Gateway:
    return _make_gateway(ctx.obj['aws_profile'], ctx.obj['aws_region'])


@cli.command()
@click.option('--require-gpu', is_flag=True, default=False)
@click.pass_context
def validate(ctx: click.Context, require_gpu: bool) -> None:
    """Validate credentials, manifests, storage, checkpoint, and GPU readiness."""
    from .validate_cmd import run_validate

    report = run_validate(_gateway(ctx), require_gpu=require_gpu)
    click.echo(reporting.format_validation(report))


@cli.command()
@click.option('--run-id', required=True)
@click.option('--max-samples', type=int, default=None)
@click.pass_context
def prepare(ctx: click.Context, run_id: str, max_samples: int | None) -> None:
    """Ingest, validate, deduplicate, convert, split, and publish a dataset."""
    from .pipeline import prepare as run_prepare

    outcome = run_prepare(_gateway(ctx), run_id=run_id, max_samples=max_samples)
    click.echo(
        reporting.format_prepare(
            {
                'dataset_id': outcome.dataset_id,
                'reused': outcome.reused,
                'selected': outcome.selected,
                'survivors': outcome.survivors,
                'splits': outcome.splits,
                'dataset_prefix': f's3://{constants.EMBY_OSS_BUCKET}/{constants.DATASETS_PREFIX}{outcome.dataset_id}/',
            }
        )
    )


@cli.command()
@click.option('--run-id', required=True)
@click.option('--max-samples', type=int, default=None)
@click.option('--epochs', type=int, default=constants.DEFAULT_EPOCHS, show_default=True)
@click.pass_context
def train(ctx: click.Context, run_id: str, max_samples: int | None, epochs: int) -> None:
    """Prepare (or reuse) a dataset and run the fixed training recipe."""
    from .pipeline import run_training

    result = run_training(_gateway(ctx), run_id=run_id, max_samples=max_samples, epochs=epochs)
    click.echo(reporting.format_train(result))


@cli.command()
@click.option('--run-id', required=True)
@click.option('--output', required=True, type=click.Path())
@click.pass_context
def visualize(ctx: click.Context, run_id: str, output: str) -> None:
    """Render up to 16 prepared-sample overlays to a local directory."""
    from .paths import data_run_dir, validate_run_id
    from .visualize import visualize as run_visualize

    validate_run_id(run_id)
    dataset_root = data_run_dir(run_id) / 'dataset'
    result = run_visualize(dataset_root, Path(output))
    click.echo(reporting.format_visualize(result))


def main(argv: list[str] | None = None) -> int:
    """Entry point mapping :class:`YoloPipeError` to stable, non-zero exit codes."""
    try:
        cli.main(args=argv, standalone_mode=False)
    except YoloPipeError as exc:
        click.echo(f'error [{exc.stage}]: {exc.message}', err=True)
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Abort:
        click.echo('aborted', err=True)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
