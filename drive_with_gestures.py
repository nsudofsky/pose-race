"""
Drive the LEGO double-motor car using your own trained arm-gesture classifier.

Requires models/gesture_classifier.joblib (see train_gesture_classifier.py).

Run yourself in a normal Terminal (needs camera access + a display, and the
double motor powered on and in Bluetooth range):
    source .venv/bin/activate
    python3 drive_with_gestures.py

Press 'q' in the video window to stop.
"""

import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from lelib import doubleMotor
from pose_features import PoseFeatureExtractor

MODEL_PATH = Path(__file__).parent / "models" / "gesture_classifier.joblib"

# How many recent predictions to vote across before acting — smooths out
# single-frame misclassifications so the car doesn't twitch.
SMOOTHING_WINDOW = 5
DRIVE_SPEED = 40


# action -> (left wheel speed %, right wheel speed %). Pivot turns in place;
# flip the signs below if "left"/"right" come out backwards on your build.
ACTION_TANK_SPEEDS = {
    "forward": (DRIVE_SPEED, DRIVE_SPEED),
    "left": (-DRIVE_SPEED, DRIVE_SPEED),
    "right": (DRIVE_SPEED, -DRIVE_SPEED),
    "stop": (0, 0),
}


def apply_action(dm, action, current_action):
    """Only send a new motor command when the smoothed action changes."""
    if action == current_action:
        return current_action

    speed_left, speed_right = ACTION_TANK_SPEEDS[action]
    dm.movement_move_tank(speed_left, speed_right)
    return action


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found — run collect_gesture_data.py then "
            "train_gesture_classifier.py first."
        )

    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    extractor = PoseFeatureExtractor()

    print("Connecting to double motor...")
    dm = doubleMotor()
    dm.connect(card_serial=None)
    print("Connected.")

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    current_action = "stop"
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            features, landmarks = extractor.process(mp_image, timestamp_ms)

            if features is not None:
                prediction = model.predict([features])[0]
                recent_predictions.append(prediction)
            else:
                recent_predictions.append("stop")  # no person -> stop, fail safe

            smoothed_action = Counter(recent_predictions).most_common(1)[0][0]
            current_action = apply_action(dm, smoothed_action, current_action)

            cv2.putText(frame, f"action: {current_action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if landmarks is None:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("drive_with_gestures - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        dm.movement_move_tank(0, 0)
        dm.movement_stop()
        dm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
