"""
Record labeled arm-gesture training data from your webcam.

Run it, stand where the camera can see your shoulders/elbows/wrists, and
press a label key to start recording that gesture continuously (every frame
while the key's label is active gets appended as one training example).
Switch labels or pause anytime; quit with 'q' to save everything to
gesture_data.csv.

Keys:
    1 = forward   (both arms raised out to the sides / up)
    2 = left      (left arm raised, right arm down)
    3 = right     (right arm raised, left arm down)
    4 = stop      (both arms down / neutral)
    0 = pause (stop recording, no label)
    q = quit and save

Run this yourself in a normal Terminal (not through Claude Code) so macOS can
prompt you for camera access and a live preview window can open:
    source .venv/bin/activate
    python3 collect_gesture_data.py
"""

import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

from pose_features import PoseFeatureExtractor, FEATURE_NAMES

LABELS = {
    ord("1"): "forward",
    ord("2"): "left",
    ord("3"): "right",
    ord("4"): "stop",
}

CSV_PATH = Path(__file__).parent / "gesture_data.csv"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    extractor = PoseFeatureExtractor()
    current_label = None
    counts = {name: 0 for name in LABELS.values()}

    rows = []
    start_time = time.time()

    print("Recording controls: 1=forward 2=left 3=right 4=stop 0=pause q=quit+save")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror, more intuitive to control against
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            features, landmarks = extractor.process(mp_image, timestamp_ms)

            if features is not None and current_label is not None:
                rows.append([*features, current_label])
                counts[current_label] += 1

            # --- overlay ---
            status = f"label: {current_label or '(paused)'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if current_label else (0, 0, 255), 2)
            y = 60
            for name, count in counts.items():
                cv2.putText(frame, f"{name}: {count}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 22
            if landmarks is None:
                cv2.putText(frame, "NO PERSON DETECTED", (10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("collect_gesture_data - q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("0"):
                current_label = None
            elif key in LABELS:
                current_label = LABELS[key]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    if not rows:
        print("No data recorded — nothing saved.")
        return

    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([*FEATURE_NAMES, "label"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {CSV_PATH} (counts: {counts})")


if __name__ == "__main__":
    main()
