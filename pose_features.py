"""
Shared arm-pose feature extraction, built on top of MediaPipe's pretrained
Pose Landmarker (models/pose_landmarker_lite.task).

We do NOT train the pose model itself — it's Google's pretrained landmark
detector. What we train (see train_gesture_classifier.py) is a small
classifier on top of a hand-crafted feature vector describing arm position,
built from that model's output.

Feature vector, per frame (14 numbers):
    - normalized (x, y) of: left_shoulder, right_shoulder, left_elbow,
      right_elbow, left_wrist, right_wrist       -> 12 values
    - left arm angle, right arm angle (degrees from vertical, upper arm
      shoulder->elbow direction)                  -> 2 values

Normalization: origin = shoulder midpoint, scale = shoulder width. This makes
the features roughly invariant to how far you stand from the camera and
where you stand in frame, but NOT invariant to camera rotation/tilt or which
way you're facing — see the README limitations section.
"""

import math
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model_downloaded(path):
    """Fetch Google's pretrained pose landmarker model on first run.

    It's ~6MB, so we don't commit it to git — every teammate's first run
    downloads it once instead.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained pose model to {path} ...")
    # macOS python.org builds don't always trust the system CA store by
    # default, so point at certifi's bundle explicitly rather than relying
    # on each teammate having run "Install Certificates.command".
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as response:
        path.write_bytes(response.read())

# MediaPipe Pose landmark indices we care about.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

FEATURE_NAMES = [
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_elbow_x", "left_elbow_y",
    "right_elbow_x", "right_elbow_y",
    "left_wrist_x", "left_wrist_y",
    "right_wrist_x", "right_wrist_y",
    "left_arm_angle", "right_arm_angle",
]


def _angle_from_vertical(dx, dy):
    """Angle in degrees between the vector (dx, dy) and straight up, signed."""
    return math.degrees(math.atan2(dx, -dy))


class PoseFeatureExtractor:
    """Wraps MediaPipe's pretrained Pose Landmarker for a live video stream."""

    def __init__(self, model_path=MODEL_PATH):
        _ensure_model_downloaded(model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, mp_image, timestamp_ms=None):
        """Run pose detection on one frame.

        Returns (features, landmarks) where features is a 14-element
        np.ndarray (or None if no person detected) and landmarks is the raw
        MediaPipe pose landmark list (or None), useful for drawing.
        """
        if timestamp_ms is None:
            self._timestamp_ms += 33  # assume ~30fps if caller doesn't track time
            timestamp_ms = self._timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None, None

        landmarks = result.pose_landmarks[0]
        return extract_features(landmarks), landmarks

    def close(self):
        self._landmarker.close()


def extract_features(landmarks):
    """Turn one frame's pose landmarks into our normalized feature vector."""
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    le = landmarks[LEFT_ELBOW]
    re = landmarks[RIGHT_ELBOW]
    lw = landmarks[LEFT_WRIST]
    rw = landmarks[RIGHT_WRIST]

    origin_x = (ls.x + rs.x) / 2.0
    origin_y = (ls.y + rs.y) / 2.0
    scale = math.hypot(ls.x - rs.x, ls.y - rs.y)
    if scale < 1e-6:
        return None  # shoulders too close together to normalize meaningfully

    def norm(pt):
        return (pt.x - origin_x) / scale, (pt.y - origin_y) / scale

    ls_n, rs_n = norm(ls), norm(rs)
    le_n, re_n = norm(le), norm(re)
    lw_n, rw_n = norm(lw), norm(rw)

    left_angle = _angle_from_vertical(le_n[0] - ls_n[0], le_n[1] - ls_n[1])
    right_angle = _angle_from_vertical(re_n[0] - rs_n[0], re_n[1] - rs_n[1])

    return np.array([
        *ls_n, *rs_n, *le_n, *re_n, *lw_n, *rw_n,
        left_angle, right_angle,
    ], dtype=np.float64)
