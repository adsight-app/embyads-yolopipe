"""Human-readable CLI summaries. No remote telemetry, server, or progress daemon."""

from __future__ import annotations


def format_validation(result: dict) -> str:
    lines = ['validation: ok']
    lines.append(
        f'  manifests validated: {result.get("annotated_rows", 0)} annotated, '
        f'{result.get("skipped_rows", 0)} skipped failure rows'
    )
    lines.append(f'  catalog fingerprint: {result.get("catalog_fingerprint", "?")}')
    lines.append(f'  base checkpoint: {result.get("checkpoint", "?")}')
    readiness = result.get('training_readiness', {})
    lines.append(f'  training_readiness: {readiness.get("state", "?")} ({readiness.get("reason", "")})')
    return '\n'.join(lines)


def format_prepare(result: dict) -> str:
    lines = ['prepare: ok']
    lines.append(f'  dataset_id: {result.get("dataset_id", "?")}')
    lines.append(f'  reused: {result.get("reused", False)}')
    lines.append(f'  selected: {result.get("selected", 0)} survivors: {result.get("survivors", 0)}')
    splits = result.get('splits', {})
    lines.append('  splits: ' + ', '.join(f'{k}={len(v)}' for k, v in splits.items()))
    lines.append(f'  dataset prefix: {result.get("dataset_prefix", "?")}')
    return '\n'.join(lines)


def format_train(result: dict) -> str:
    lines = ['train: ok']
    lines.append(f'  mode: {result.get("mode", "?")}  epochs: {result.get("epochs", "?")}')
    lines.append(f'  dataset_id: {result.get("dataset_id", "?")}')
    lines.append(f'  run prefix: {result.get("run_prefix", "?")}')
    lines.append(f'  model prefix: {result.get("model_prefix", "?")}')
    lines.append(f'  local run dir: {result.get("local_run_dir", "?")}')
    return '\n'.join(lines)


def format_visualize(result: dict) -> str:
    return f'visualize: ok\n  output: {result.get("output_dir", "?")}\n  images written: {result.get("count", 0)}'
