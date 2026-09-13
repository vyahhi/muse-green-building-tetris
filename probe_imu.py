#!/usr/bin/env python3
"""Print one-second Muse IMU summaries for orientation debugging."""

import time

import numpy as np
from MuseLSL2.muse import Muse

from tetris_muselsl import DEFAULT_ADDRESS


def main() -> int:
    accel: list[np.ndarray] = []
    gyro: list[np.ndarray] = []

    def on_acc(samples, _timestamps):
        accel.extend(np.asarray(samples, dtype=float).T)

    def on_gyro(samples, _timestamps):
        gyro.extend(np.asarray(samples, dtype=float).T)

    muse = Muse(
        address=DEFAULT_ADDRESS,
        callback_acc=on_acc,
        callback_gyro=on_gyro,
        preset="p50",
    )
    print("Connecting; keep head centered and still.", flush=True)
    muse.connect()
    muse.start()
    try:
        for second in range(1, 7):
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                muse.adapter.pump(0.02)
            acc = np.asarray(accel)
            gyr = np.asarray(gyro)
            print(
                f"{second}s accel median={np.median(acc, axis=0).round(4).tolist()} "
                f"range={(np.ptp(acc, axis=0)).round(4).tolist()} "
                f"gyro median={np.median(gyr, axis=0).round(2).tolist()} "
                f"range={(np.ptp(gyr, axis=0)).round(2).tolist()}",
                flush=True,
            )
            accel.clear()
            gyro.clear()
    finally:
        muse.stop()
        muse.adapter.pump(0.1)
        muse.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
