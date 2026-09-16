# pose-race

Control a LEGO Education car with arm gestures read from a webcam via
MediaPipe.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

## Hardware bring-up

`setup_two_motors.py` is a smoke test for the two-motor drivetrain (the
"Double Motor" hub, with a LEFT and RIGHT motor port). Power it on, keep it
in Bluetooth range, then:

```bash
python3 setup_two_motors.py
```

It spins the left motor, spins the right motor, drives forward for a second,
then turns — printing the IMU heading along the way so you can confirm both
motors and the gyro are working before wiring up gesture control.

`lelib.py` is the shared wrapper (from the class's `legoeducation` package)
that everything else in this project builds on — use its `doubleMotor` /
`singleMotor` / `controller` / `colorSensor` classes rather than the raw
`legoeducation` API directly.

## How Python talks to the LEGO hardware

The `legoeducation` package talks to the hub over Bluetooth Low Energy (BLE)
using an RPC protocol, with `bleak` as the BLE transport. Internally it runs
an `asyncio` event loop on a background thread (see `background_worker.py` /
`ble_transport.py` in the package) to handle BLE I/O, but the public API
(`SingleMotor`, `DoubleMotor`, etc., wrapped here by `lelib.py`) exposes
**synchronous, blocking calls** — e.g. `dm.run_time(1000)` blocks the calling
thread until that command finishes. So from the app code's point of view it
is synchronous; underneath, the BLE communication itself is asynchronous.

## Gesture control

Three scripts, run in order, all needing a normal Terminal window (not run
through an AI assistant's sandboxed shell) since they need webcam + display
access, and the last one needs the double motor powered on nearby:

1. **`collect_gesture_data.py`** — opens the webcam, shows a live preview.
   Press `1`/`2`/`3`/`4`/`5` to start recording examples of `forward`/`left`/
   `right`/`stop`/`reverse` (arms crossed in front of your face), `0` to
   pause, `q` to save everything to `gesture_data.csv`. Move around while
   recording each gesture — vary your distance from the camera and position
   in frame — so the classifier doesn't just memorize one exact spot. Aim
   for at least ~30 seconds (a few hundred frames) per gesture.
2. **`train_gesture_classifier.py`** — loads `gesture_data.csv`, splits it
   80/25 into train/test, fits a logistic regression classifier, prints test
   accuracy and a confusion matrix, and saves the model to
   `models/gesture_classifier.joblib`.
3. **`drive_with_gestures.py`** — runs the same webcam + pose pipeline live,
   feeds each frame's features through the trained classifier, smooths
   predictions over a 5-frame rolling window (to avoid flicker), and drives
   the double motor with `movement_move_tank(left%, right%)` accordingly.
   Speed is set independently every frame from how far your hands are from
   your head (wrist-to-nose distance, normalized by shoulder width) — arms
   extended drives faster, hands near your head drives slower. Press `q` to
   stop; it disconnects the motor cleanly.

### How it works / how it was trained

We do **not** train the underlying pose model — `pose_features.py` uses
MediaPipe's pretrained **Pose Landmarker** (`pose_landmarker_lite.task`,
downloaded automatically on first run) to find 33 body keypoints per frame.
From those we hand-pick the shoulders, elbows, and wrists, normalize them
(origin = shoulder midpoint, scale = shoulder width, so it doesn't matter how
far you stand from the camera) and compute each arm's angle from vertical —
14 numbers per frame total.

What **is** trained is a small `scikit-learn` **logistic regression**
classifier on top of those 14 features, fit on data we record ourselves with
`collect_gesture_data.py`. It's a standard supervised-learning setup: labeled
examples, a held-out test split, and a reported accuracy — not just
hand-written if/else thresholds.

### Limitations

- **Small, personal dataset.** It's trained on whoever recorded the data,
  in one room, under one lighting setup. It won't generalize well to a
  different person's body proportions, a different camera angle, or a much
  darker/brighter room without collecting more data there.
- **Only 2D image-plane features.** Normalizing by shoulder width makes it
  roughly distance-invariant, but not invariant to camera rotation/tilt or to
  facing a different direction — turning sideways to the camera changes the
  apparent arm geometry.
- **Single person, single pose per frame.** `num_poses=1`; a second person in
  frame is ignored or confuses shoulder-width normalization if they're closer
  to the camera.
- **Latency.** Each frame: pose inference → feature extraction → classifier
  predict → 5-frame smoothing → BLE command. Smoothing intentionally trades
  a small delay for fewer spurious direction changes.
- **Small gesture vocabulary.** Five classes (forward/left/right/stop/
  reverse) plus a continuous hand-distance speed control; still no combined
  turn-while-moving gesture.
- **Fail-safe:** if no person is detected in frame, it defaults to `stop`
  rather than continuing the last command.
