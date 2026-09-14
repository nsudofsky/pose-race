"""
Bring-up test for the two-motor LEGO Education drivetrain (the "Double Motor"
device — a single BLE hub with a LEFT and RIGHT motor port, used to drive a car).

What it does:
    1. Connects to the nearest double motor over Bluetooth.
    2. Spins each side individually so you can confirm wiring/direction.
    3. Drives forward briefly, then turns, using the IMU to report heading.

Run it with the double motor powered on and within Bluetooth range:
    source .venv/bin/activate
    python3 setup_two_motors.py

If you have more than one double motor nearby, find its card_serial/card_color
printed on the connection card that ships with the hardware and pass them to
connect() instead of card_serial=None (see main() below).
"""

import time

from lelib import doubleMotor


def main():
    dm = doubleMotor()

    print("Connecting to double motor...")
    dm.connect(card_serial=None)  # grabs whichever double motor is nearby
    print("Connected.")

    try:
        print("Spinning LEFT motor forward for 1 rotation...")
        dm.run_left(degrees=360)
        time.sleep(1)

        print("Spinning RIGHT motor forward for 1 rotation...")
        dm.run_right(degrees=360)
        time.sleep(1)

        dm.reset_heading()
        print("Driving forward for 1 second...")
        dm.run_time(1000)
        print(f"Heading after forward drive: {dm.yaw():.1f} degrees")

        print("Turning right 90 degrees...")
        dm.turn_right(90)
        print(f"Heading after turn: {dm.yaw():.1f} degrees")

    finally:
        dm.stop()
        dm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
