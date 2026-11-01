"""Immutable repository-owned constants for the yolopipe pipeline."""

from __future__ import annotations

# S3 write confinement boundary. All pipeline-managed writes live under this.
EMBY_OSS_BUCKET = 'emby-oss'
EMBY_OSS_PREFIX = 'yolopipe/'

# S3 output sub-prefixes (relative to EMBY_OSS_PREFIX).
DATASETS_PREFIX = 'yolopipe/datasets/'
RUNS_PREFIX = 'yolopipe/runs/'
MODELS_PREFIX = 'yolopipe/models/'
BASE_CHECKPOINT_KEY = 'yolopipe/base/yolo11n-seg.pt'
BASE_CHECKPOINT_SHA256_KEY = 'yolopipe/base/yolo11n-seg.pt.sha256'

# Base checkpoint integrity: authoritative repository constant.
BASE_CHECKPOINT_SHA256 = '55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152'
BASE_CHECKPOINT_SIZE = 6182636

# Local working roots. ./data is durable, ./cache is disposable.
DATA_ROOT = 'data/yolopipe'
CACHE_ROOT = 'cache/yolopipe'
CACHE_OBJECTS_SUBDIR = 'objects'
ULTRALYTICS_CONFIG_SUBDIR = 'ultralytics'

# AWS defaults.
DEFAULT_AWS_PROFILE = 'embytest'
DEFAULT_AWS_REGION = 'eu-west-1'

# Class contract.
CLASS_NAME = 'gta5-billboard'
CLASS_ID = 0
GROUND_TRUTH_CLASS = 'billboard'
IGNORED_CLASSES = frozenset({'BACKGROUND'})

# Ground Truth annotation type.
ANNOTATION_TYPE = 'groundtruth/semantic-segmentation'

# Deterministic split.
SPLIT_SEED = 'yolopipe-gta5-v1'
SPLIT_RATIOS = {'train': 0.75, 'val': 0.15, 'minival': 0.10}
SPLIT_ORDER = ('train', 'val', 'minival')

# Conversion constants.
MIN_CONTOUR_AREA = 5.0
EPSILON_FACTOR = 0.005
COORD_PRECISION = 6
MIN_CONTOUR_POINTS = 4

# Deduplication (perceptual, per-manifest).
DEDUP_ALGORITHM_VERSION = 'whash16-ccoeff-v1'
DEDUP_HASH_SIZE = 16
DEDUP_THRESHOLD = 0.85
DEDUP_RESIZE = 256

# Algorithm versions (participate in dataset fingerprint).
SELECTION_ALGORITHM_VERSION = 'select-whole-unit-v1'
SPLIT_ALGORITHM_VERSION = 'largest-remainder-greedy-v1'
CONVERSION_ALGORITHM_VERSION = 'contour-yolo-seg-v1'

# Fixed training recipe (production).
BASE_MODEL = 'yolo11n-seg.pt'
DEFAULT_EPOCHS = 100
IMAGE_SIZE = 640
GLOBAL_BATCH = 16
WORKERS = 8
PRODUCTION_DEVICES = (0, 1, 2, 3)

# S3 transfer concurrency bound.
TRANSFER_WORKERS = 8

# Completion marker filename written last under each published prefix.
COMPLETION_MARKER = '_COMPLETE'
