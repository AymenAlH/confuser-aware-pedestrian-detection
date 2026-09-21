from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_ROOT = ROOT / "YOLOv5"
DEFAULT_WEIGHTS = YOLOV5_ROOT / "runs/train/v2/weights/best.pt"
SETTINGS_FILE = Path(__file__).resolve().parent / ".settings.json"

CLASS_NAMES = {0: "person", 1: "person-like"}

CONFIDENCE_THRESH = 0.5
IOU_THRESH = 0.4
MAX_LOST_FRAMES = 5

WINDOW_SIZE = 10
CLASS_CONSISTENCY_THRESH = 0.8
HIGH_CONFIDENCE = 0.786
HYSTERESIS_FRAMES = 5
BBOX_MOVEMENT_THRESH = 50
STATIONARY_VELOCITY = 5
CONFIDENCE_DROP_THRESH = 0.3

BRAKE_MIN_HEIGHT_RATIO = 0.30   # bbox must be at least 30% of frame height
BRAKE_LATERAL_MARGIN = 0.35     # bbox foot-x must lie inside the central 30% of frame width
BRAKE_MIN_FOOT_RATIO = 0.55     # bbox bottom must be at least 55% down the frame

DECISION_COLORS = {
    "SUPPRESS": (0, 200, 0),
    "WARN": (0, 220, 255),
    "BRAKE": (0, 0, 230),
    "NORMAL": (160, 160, 160),
}
