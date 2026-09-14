# lego-arm-racer

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

## Training / limitations

(Fill in once gesture recognition is built: what MediaPipe model is used —
e.g. Hand Landmarker / Pose Landmarker — whether it's used out-of-the-box or
fine-tuned, what gestures map to what car behavior, and where it breaks down:
lighting sensitivity, occlusion, latency between camera frame and motor
command, limited gesture vocabulary, single-person tracking, etc.)
