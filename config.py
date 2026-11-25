"""
Configuration hub for 图像检测系统.
All paths, model configs, UI settings, and constants are defined here.
"""

from pathlib import Path
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ASSETS_DIR = ROOT / "assets"
IMAGES_DIR = ROOT / "images"

WEIGHTS_DIR = ROOT / "weights"

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(exist_ok=True)

# ─── App Metadata ────────────────────────────────────────────────────────────
APP_TITLE = "图像检测系统"
APP_ICON = None
APP_VERSION = "2.1.0"
APP_DESCRIPTION = (
    "支持图片目标检测、实例分割和姿态估计。"
    ""
)

TASK_DETECT = "目标检测"
TASK_SEGMENT = "实例分割"
TASK_WORLD = "YOLO World v2（文本检测）"
TASK_YOLOE = "YOLOE（文本分割）"
TASK_POSE = "姿态估计"
TASKS_LIST = [TASK_DETECT, TASK_SEGMENT, TASK_WORLD, TASK_YOLOE, TASK_POSE]

DETECTION_MODELS = {
    "YOLO26-微型 （轻量版）": "yolo26n.pt",
    "YOLO26-小型": "yolo26s.pt",
    "YOLO26-中型": "yolo26m.pt",
    "YOLO26-大型": "yolo26l.pt",
    "YOLO26-加大型 （加大版）": "yolo26x.pt",
    "RT-DETR-大型 （Transformer 架构）": "rtdetr-l.pt",
    "RT-DETR-加大型 （Transformer 架构）": "rtdetr-x.pt",
}

SEGMENTATION_MODELS = {
    "YOLO26-微型-seg （轻量版）": "yolo26n-seg.pt",
    "YOLO26-小型-seg": "yolo26s-seg.pt",
    "YOLO26-中型-seg": "yolo26m-seg.pt",
    "YOLO26-大型-seg": "yolo26l-seg.pt",
    "YOLO26-加大型-seg （加大版）": "yolo26x-seg.pt",
}

POSE_MODELS = {
    "YOLO26-微型-pose （轻量版）": "yolo26n-pose.pt",
    "YOLO26-小型-pose": "yolo26s-pose.pt",
    "YOLO26-中型-pose": "yolo26m-pose.pt",
    "YOLO26-大型-pose": "yolo26l-pose.pt",
    "YOLO26-加大型-pose （加大版）": "yolo26x-pose.pt",
}

WORLD_MODELS = {
    "YOLOv8-小型-worldv2": "yolov8s-worldv2.pt",
    "YOLOv8-中型-worldv2": "yolov8m-worldv2.pt",
    "YOLOv8-大型-worldv2 （推荐）": "yolov8l-worldv2.pt",
    "YOLOv8-加大型-worldv2 （加大版）": "yolov8x-worldv2.pt",
}

YOLOE_MODELS = {
    "YOLOE-26n-seg （轻量版）": "yoloe-26n-seg.pt",
    "YOLOE-26s-seg": "yoloe-26s-seg.pt",
    "YOLOE-26m-seg": "yoloe-26m-seg.pt",
    "YOLOE-26l-seg （推荐）": "yoloe-26l-seg.pt",
    "YOLOE-26x-seg （加大版）": "yoloe-26x-seg.pt",
}

# Defaults (first key in each dict)
DETECTION_MODEL = "yolo26n.pt"
SEGMENTATION_MODEL = "yolo26n-seg.pt"
POSE_MODEL = "yolo26n-pose.pt"

# YOLO World v2: open-vocabulary detection via natural language text prompts
YOLO_WORLD_MODEL = "yolov8l-worldv2.pt"

# YOLOE: open-vocabulary text-prompted detection + segmentation
YOLOE_MODEL = "yoloe-26l-seg.pt"

# ─── Default Assets ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"

DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50
MIN_CONFIDENCE = 10  # slider min (%)
MAX_CONFIDENCE = 100  # slider max (%)
DEFAULT_WORLD_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"

# ─── YOLOE Defaults ──────────────────────────────────────────────────────────
# YOLOE supports category-level text prompts (NOT descriptive phrases).
# Unlike YOLO World v2, YOLOE provides detection + segmentation masks.
DEFAULT_YOLOE_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"


def use_local_weights_dir() -> str | None:
    """Point Ultralytics' asset lookup at this repo's ``weights/``.

    Ultralytics resolves a missing asset against the current directory first and
    then ``SETTINGS["weights_dir"]``. Unless that setting points here, a weight we
    already have in ``weights/`` gets downloaded a second time into whatever
    directory the process happens to be running from.

    Returns the previous value if it was changed, else ``None``.
    """
    from ultralytics.utils import SETTINGS  # local import keeps config import cheap

    previous = SETTINGS["weights_dir"]
    if previous == str(WEIGHTS_DIR):
        return None
    SETTINGS["weights_dir"] = str(WEIGHTS_DIR)
    return previous


def resolve_model_path(model_name: str) -> str:
    """Return local weights path if it exists, else the bare name for auto-download.

    After auto-download, call ``sweep_stray_weights()`` to move any
    weight files that landed in the project root into ``weights/``.
    """
    local = WEIGHTS_DIR / model_name
    if local.exists():
        return str(local)
    # Not in weights/ yet — check project root (old download location)
    root_copy = ROOT / model_name
    if root_copy.exists():
        root_copy.rename(local)
        return str(local)
    # Will be auto-downloaded to CWD; return bare name
    return model_name


# What an auto-download can drop in the project root: ``.pt`` checkpoints and the
# ``.ts`` TorchScript CLIP text encoder that YOLO World v2 / YOLOE pull in.
_STRAY_WEIGHT_PATTERNS = ("*.pt", "*.ts")


def sweep_stray_weights() -> None:
    """Move any stray weight files from the project root into ``weights/``."""
    for pattern in _STRAY_WEIGHT_PATTERNS:
        for stray in ROOT.glob(pattern):
            dest = WEIGHTS_DIR / stray.name
            if not dest.exists():
                stray.rename(dest)


def get_model_catalog(task: str) -> dict[str, str]:
    """Return ``{display_label: filename}`` for the given *task*."""
    _CATALOGS = {
        TASK_DETECT: DETECTION_MODELS,
        TASK_SEGMENT: SEGMENTATION_MODELS,
        TASK_POSE: POSE_MODELS,
        TASK_WORLD: WORLD_MODELS,
        TASK_YOLOE: YOLOE_MODELS,
    }
    return _CATALOGS.get(task, DETECTION_MODELS)
