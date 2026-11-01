# embyads-yolopipe

A self-contained, open-source command-line pipeline that turns a fixed set of
SageMaker Ground Truth GTA 5 billboard segmentation manifests into a trained
[Ultralytics](https://docs.ultralytics.com/) YOLO segmentation model. It is
clone-and-run: there is nothing to publish or install globally. You clone the
repository, sync a frozen dependency lock, and drive everything through the
`yolopipe` command.

This document is the complete operator guide — architecture, setup, every
command, the data contracts, failure recovery, and licensing.

---

## 1. What it does

The pipeline runs one fixed path and nothing else:

```
fixed manifests → manifest validation + bounded selection
               → per-manifest perceptual image deduplication
               → mask → YOLO-segmentation label conversion
               → deterministic 75/15/10 split
               → fixed four-GPU YOLO training (Linux)
               → immutable publication to s3://emby-oss/yolopipe/
```

There is deliberately no orchestration framework, no distributed preprocessing,
no web UI, no monitoring or notifications, no ONNX/RF-DETR path, and no tuning
surface beyond the number of epochs. Everything the pipeline writes lives under
one S3 prefix and two local directories.

### Architecture

| Component | Responsibility |
|---|---|
| `catalog` | The fixed nine-job manifest list and its static SHA-256 fingerprint. |
| `manifest` / `selection` | Manifest-only validation, strict identity uniqueness, bounded group-aware selection. |
| `localize` | Atomic download of selected images (then survivor masks only) into the cache. |
| `dedup` / `image_matcher` | Per-manifest perceptual deduplication (wavelet-hash order + OpenCV correlation). |
| `conversion` | Canonical mask → YOLO polygon label bytes. |
| `split` | Deterministic 75/15/10 assignment with a fixed seed. |
| `dataset` / `dataset_publish` | Portable dataset materialization, content-addressed publication. |
| `checkpoint` / `preflight` / `trainer` | Base-weight integrity, GPU preflight, the fixed training recipe. |
| `run_publish` | Immutable run and model publication with completion markers. |
| `storage` / `paths` | The single S3 write boundary and local path guards. |

---

## 2. The fixed dataset

Nine completed GTA 5 billboard Ground Truth jobs form one immutable catalog
(six Gen1, three Gen2). The stopped `…-06-clone` job is intentionally excluded.
The manifests are read directly from their existing S3 locations; the tool never
calls the SageMaker API and needs no SageMaker IAM permission.

**Identity vs. content.** Two distinct rules apply:

- **Identity uniqueness** — every `source-ref` URI and every source filename
  across all nine manifests must be unique. Any duplicate fails validation.
- **Perceptual deduplication** — within each manifest (never across manifests),
  images are ordered by a 16×16 wavelet hash and each candidate is compared only
  to the last retained image using OpenCV `TM_CCOEFF_NORMED`; a correlation of
  at least `0.85` drops the candidate. This is a heuristic perceptual filter,
  not exact-pixel or byte equality, and it is not transitive.

Ground Truth rows that were skipped or expired during labeling (where the mask
ref is absent and the metadata carries a `failure-reason`) are counted and
skipped. Any other missing field is a validation error.

---

## 3. Storage layout

**S3 (writes confined to `s3://emby-oss/yolopipe/`):**

```
s3://emby-oss/yolopipe/
  base/yolo11n-seg.pt         # pinned base checkpoint (+ .sha256 sidecar)
  datasets/<dataset-id>/      # content-addressed prepared dataset
  runs/<run-id>/              # complete training run + metadata
  models/<run-id>/            # best.pt + checksum + linkage (model-id == run-id)
```

Input manifests, images, and masks are read from their existing buckets. The
pipeline never writes outside `s3://emby-oss/yolopipe/`.

**Local:**

```
./data/yolopipe/<run-id>/     # durable prepared data, run outputs (retained)
./cache/yolopipe/             # disposable downloads + Ultralytics config (YOLO_CONFIG_DIR)
```

`./data` and `./cache` must already exist. The pipeline never writes to your
home directory — Ultralytics is pointed at `./cache/yolopipe/ultralytics` so it
creates no `~/.config/Ultralytics` settings file.

---

## 4. IAM requirements

- **Read**: `s3:GetObject` on the input manifest/image/mask buckets.
- **Read/Write**: `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` scoped to
  `s3://emby-oss/yolopipe/*`.
- **Not required**: any SageMaker permission.

The default AWS profile is `embytest` and region `eu-west-1`; override with
`--aws-profile` / `--aws-region`. Credentials follow the normal AWS SDK chain
(`aws sso login`, environment variables, instance role) and are never logged.

---

## 5. Platforms

| OS | validate / prepare / visualize | train |
|---|---|---|
| Ubuntu Linux | yes | yes — production four-GPU and smoke |
| Windows 11 | yes | smoke only (single GPU, see below) |

Production training targets an Ubuntu EC2 `g5.12xlarge` with exactly four
NVIDIA A10G GPUs and uses NCCL DDP, which is Linux-only. Data preparation,
validation, and visualization need no GPU and run identically on both operating
systems; persisted dataset artifacts are byte-identical across platforms.

### Smoke-test exception

For quick local iteration, supplying **both** `--max-samples` and `--epochs`
puts `train` in **smoke mode**: a single-process, single-GPU run that works on
Windows or Linux. Smoke runs are labeled `mode: smoke` in their metadata and are
not production models. Example (Windows, one GPU):

```
uv run yolopipe train --run-id dev-smoke --max-samples 100 --epochs 3
```

---

## 6. Setup

```
git clone <this-repo> embyads-yolopipe
cd embyads-yolopipe
uv sync --frozen
mkdir data cache          # if they do not already exist
aws sso login --profile embytest
```

The base checkpoint must exist at `s3://emby-oss/yolopipe/base/yolo11n-seg.pt`
(a one-time operator bootstrap, verified against the pinned SHA-256). `validate`
tells you if it is missing.

---

## 7. Commands

### `validate`

```
uv run yolopipe validate
uv run yolopipe validate --require-gpu
```

Downloads and schema-validates all nine manifests (no image/mask download),
checks the S3 write boundary, local roots, and base-checkpoint integrity, and
reports `training_readiness`. Plain `validate` exits `0` on a CPU host and
reports `not_ready`; `--require-gpu` exits non-zero unless the Linux four-GPU
preflight passes.

### `prepare`

```
uv run yolopipe prepare --run-id my-run
uv run yolopipe prepare --run-id my-run --max-samples 100
```

Validates, selects, localizes, deduplicates, converts, splits, materializes, and
publishes the content-addressed dataset. Re-running with the same `--run-id` and
identical parameters reuses the local dataset without re-downloading manifests;
a finalized remote dataset with matching checksums is reused without re-upload.
`--max-samples` must be at least 3.

### `train`

```
uv run yolopipe train --run-id my-run                       # production (Linux, 4 GPU, 100 epochs)
uv run yolopipe train --run-id my-run --epochs 50           # production, fewer epochs
uv run yolopipe train --run-id dev --max-samples 100 --epochs 3   # smoke
```

Fails before any work if a `runs/<run-id>/` or `models/<run-id>/` prefix already
exists (no resume, no overwrite). Reuses a matching local dataset or prepares
one, runs the fixed recipe, and publishes the run and model with completion
markers written last. `--epochs` is the only training override and does not
change the dataset id.

### `visualize`

```
uv run yolopipe visualize --run-id my-run --output ./viz
```

Writes up to 16 overlay PNGs (`00_<id>.png` … `15_<id>.png`) of source image +
mask + YOLO polygons to `--output`. Requires a prepared dataset; never opens a
GUI or writes to S3.

---

## 8. Outputs and reproducibility

A prepared dataset is content-addressed: the same inputs and pinned libraries
produce the same `dataset-id`, byte-identical labels/splits/`data.yaml`, and the
same per-manifest dedup survivors, regardless of run id, host, or OS. Runs and
models are immutable, carry SHA-256 checksums, and gain a `_COMPLETE` marker only
after every object is uploaded — so an interrupted publish is always
distinguishable from a finished one. Determinism is pinned by the dependency
lock, a fixed split seed, deterministic Torch algorithms, and `CUBLAS_WORKSPACE_CONFIG`.

## 9. Failure recovery

- A failed run leaves no `_COMPLETE` marker; choose a **new** `--run-id` to retry
  (the pipeline never deletes or resumes an existing prefix).
- If `./cache/yolopipe/` is cleared between `prepare` and `train`/`visualize`,
  missing objects are re-downloaded and re-verified automatically.
- Very small `--max-samples` values can leave fewer than three deduplicated
  units, which fails the split precondition with an actionable message — raise
  the bound.

## 10. License

Licensed under **AGPL-3.0-only** (see `LICENSE`), consistent with the Ultralytics
dependency. Published models carry the license and a notice file. If you
distribute this software or offer it as a network service, your obligations
under the AGPL apply.
