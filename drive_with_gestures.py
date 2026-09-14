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

import legoeducation as le
from lelib import doubleMotor, singleMotor
from pose_features import PoseFeatureExtractor, hand_to_head_distance

MODEL_PATH = Path(__file__).parent / "models" / "gesture_classifier.joblib"

# How many recent predictions to vote across before acting — smooths out
# single-frame misclassifications so the car doesn't twitch.
SMOOTHING_WINDOW = 5

# Speed scales with how far your hands are from your head (normalized by
# shoulder width, same units as hand_to_head_distance). Below NEAR_DIST ->
# MIN_SPEED; above FAR_DIST -> MAX_SPEED; linear in between. Tune these by
# watching the on-screen "hand_dist" readout while moving your hands.
MIN_SPEED = 15
MAX_SPEED = 100
NEAR_DIST = 0.8
FAR_DIST = 1.8

# Reverse ignores hand distance entirely and always drives at this fixed speed.
REVERSE_SPEED = 50

# left/right don't pivot-spin continuously anymore — each time the gesture is
# freshly detected, the car pivots at TURN_SPEED for TURN_DURATION seconds,
# then stops. Hold the gesture through "stop" and back to re-trigger another
# turn. (The legoeducation library has no turn-for-time primitive, only
# turn-for-degrees, so this is done manually with a tank pivot + sleep + stop.)
TURN_SPEED = 50
TURN_DURATION = 0.05

# Only re-send a tank command when the action changes or the speed drifts by
# more than this many percentage points — avoids flooding the BLE link every
# frame with near-identical speed values.
SPEED_CHANGE_THRESHOLD = 5

# Your "stop" pose (arms down at your sides) puts your wrists nearly as far
# from your nose as fully extended arms do, so hand_to_head_distance can
# briefly spike while you're lowering your hands — right before the gesture
# classifier (which lags a few frames) catches up and calls it "stop". This
# caps how many percentage points the speed is allowed to INCREASE per frame,
# so that spike can't punch the motors to full speed for an instant. Slowing
# down/stopping is never rate-limited — only speeding up is.
MAX_SPEED_INCREASE_PER_FRAME = 8

# action -> (left wheel sign, right wheel sign). "left"/"right" are handled
# separately as bounded turns (see TURN_DEGREES) and don't go through this.
ACTION_SIGNS = {
    "forward": (1, 1),
    "reverse": (-1, -1),
    "stop": (0, 0),
}


def speed_from_hand_distance(hand_dist):
    """Map a normalized hand-to-head distance to a motor speed percentage."""
    if hand_dist is None:
        return MIN_SPEED
    frac = (hand_dist - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
    frac = max(0.0, min(1.0, frac))
    return int(MIN_SPEED + frac * (MAX_SPEED - MIN_SPEED))


def rate_limit_speed(target_speed, previous_speed):
    """Cap how much speed can increase in one frame; decreases stay instant."""
    if target_speed > previous_speed:
        return min(target_speed, previous_speed + MAX_SPEED_INCREASE_PER_FRAME)
    return target_speed


def apply_action(dm, action, speed, current_action, current_speed):
    """Only send a new motor command when the action or speed meaningfully changes."""
    if action == current_action and abs(speed - current_speed) < SPEED_CHANGE_THRESHOLD:
        return current_action, current_speed

    sign_left, sign_right = ACTION_SIGNS[action]
    dm.movement_move_tank(sign_left * speed, sign_right * speed)
    return action, speed


def timed_turn(dm, direction):
    """Pivot left/right at TURN_SPEED for TURN_DURATION seconds, then stop."""
    sign_left, sign_right = (-1, 1) if direction == "left" else (1, -1)
    dm.movement_move_tank(sign_left * TURN_SPEED, sign_right * TURN_SPEED)
    time.sleep(TURN_DURATION)
    dm.movement_move_tank(0, 0)


def sync_single_motor(sm, action, speed, sm_speed):
    """Spin the single motor only while driving forward, at the same
    hand-distance speed as the drive wheels; stopped the rest of the time."""
    target = speed if action == "forward" else 0
    if target == sm_speed:
        return sm_speed
    if target != 0 and sm_speed != 0 and abs(target - sm_speed) < SPEED_CHANGE_THRESHOLD:
        return sm_speed

    if target == 0:
        sm.stop()
    else:
        sm.run(target)
    return target


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

    dm = None
    sm = None
    try:
        print("Connecting to double motor...")
        dm = doubleMotor()
        dm.connect(card_serial=None, card_color=le.LEGO_COLOR_ORANGE)  # only your orange-card hub
        print("Connected.")

        print("Connecting to single motor...")
        sm = singleMotor()
        sm.connect(card_serial=None, card_color=le.LEGO_COLOR_RED)  # only your red-card hub
        print("Connected.")
    except Exception:
        if dm is not None and dm.connected:
            dm.disconnect()
        raise

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    current_action = "stop"
    current_speed = 0
    sm_speed = 0
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

            hand_dist = hand_to_head_distance(landmarks) if landmarks is not None else None
            raw_speed = speed_from_hand_distance(hand_dist)
            speed = rate_limit_speed(raw_speed, current_speed)

            smoothed_action = Counter(recent_predictions).most_common(1)[0][0]

            if smoothed_action in ("left", "right"):
                if smoothed_action != current_action:
                    timed_turn(dm, smoothed_action)
                    current_action = smoothed_action
                    current_speed = 0
            else:
                effective_speed = REVERSE_SPEED if smoothed_action == "reverse" else speed
                current_action, current_speed = apply_action(
                    dm, smoothed_action, effective_speed, current_action, current_speed
                )

            sm_speed = sync_single_motor(sm, current_action, speed, sm_speed)

            cv2.putText(frame, f"action: {current_action}  speed: {current_speed}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"single motor: {sm_speed}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if hand_dist is not None:
                cv2.putText(frame, f"hand_dist: {hand_dist:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
        sm.stop()
        sm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
