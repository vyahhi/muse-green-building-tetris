#!/usr/bin/env python3
"""Run Green Building Tetris through MuseLSL2's macOS BLE backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from MuseLSL2.muse import Muse


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """Load simple KEY=VALUE configuration without adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env()

from tetris import API_ROOT, DISPLAY_ID, HEIGHT, WIDTH, Tetris, WebDisplay
from advanced_winks import OnlineWinkDetector, WinkProfile
from head_orientation import HeadOrientation, RollColumnMapper
from motion_controls import AccelNodDetector


DEFAULT_ADDRESS = os.environ.get("MUSE_ADDRESS")
SAMPLE_RATE = 256

POSE_SYMBOLS = {
    "REST": (
        "001100110", "001100110", "001100110", "001100110",
        "001100110", "001100110", "001100110",
    ),
    "LEFT": (
        "000100000", "001100000", "011111110", "111111111",
        "011111110", "001100000", "000100000",
    ),
    "RIGHT": (
        "000001000", "000001100", "011111110", "111111111",
        "011111110", "000001100", "000001000",
    ),
    "NOD": (
        "000010000", "000010000", "000010000", "001111100",
        "000111000", "000010000", "000000000",
    ),
    "SWEEP": (
        "000000000", "001000100", "011000110", "111111111",
        "011000110", "001000100", "000000000",
    ),
}


def pose_frame(symbol: str) -> list[list[list[int]]]:
    background = [28, 31, 39]
    colors = {
        "REST": [105, 115, 130],
        "LEFT": [35, 95, 210],
        "RIGHT": [205, 55, 50],
        "NOD": [175, 75, 205],
        "SWEEP": [35, 165, 205],
    }
    frame = [[background.copy() for _ in range(WIDTH)] for _ in range(HEIGHT)]
    pattern = POSE_SYMBOLS[symbol]
    y0 = (HEIGHT - len(pattern)) // 2
    for y, row in enumerate(pattern):
        for x, bit in enumerate(row):
            if bit == "1":
                frame[y0 + y][x] = colors[symbol].copy()
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--display", default=DISPLAY_ID)
    parser.add_argument("--api-root", default=API_ROOT)
    parser.add_argument("--swap-eyes", action="store_true")
    parser.add_argument("--drop-seconds", type=float, default=0.92)
    parser.add_argument(
        "--rotation", choices=("nod", "blink", "off"), default="nod",
        help="Rotation gesture (default: deliberate down-and-back nod)",
    )
    parser.add_argument(
        "--no-blink-rotate", action="store_true", help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    display = WebDisplay(args.display, args.api_root)
    game = Tetris(args.seed)
    profile_path = Path(__file__).with_name("wink_profile.json")
    profile = WinkProfile.load(profile_path) if args.rotation == "blink" else None
    orientation = HeadOrientation()
    nod_detector: AccelNodDetector | None = None
    eeg_chunks: list[np.ndarray] = []
    accel_chunks: list[np.ndarray] = []
    gyro_chunks: list[np.ndarray] = []
    running = True

    def on_eeg(data: np.ndarray, _timestamps: np.ndarray) -> None:
        # MuseLSL2 order is TP9, AF7, AF8, TP10, AUX.
        chunk = np.array(data[:4], copy=True)
        # MuseLSL2 initializes a packet block with zeros. If a channel packet
        # is late/missing, forwarding that block creates a false ~2,000 µV
        # blink edge. Only classify complete AF7/AF8 blocks.
        if np.all(chunk[1] == 0) or np.all(chunk[2] == 0):
            return
        eeg_chunks.append(chunk)

    def on_acc(samples: np.ndarray, _timestamps: np.ndarray) -> None:
        # MuseLSL2 0.3.0 reshapes the packet with Fortran ordering, yielding
        # axes × samples despite documenting samples × axes.
        accel_chunks.append(np.array(samples, copy=True).T)

    def on_gyro(samples: np.ndarray, _timestamps: np.ndarray) -> None:
        gyro_chunks.append(np.array(samples, copy=True).T)

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    muse = Muse(
        address=args.address,
        callback_eeg=on_eeg,
        callback_acc=on_acc,
        callback_gyro=on_gyro,
        # MuseLSL2's callback is synchronized on its fifth EEG/AUX packet.
        # p50 supplies that packet reliably; p21 produced zero-filled blocks.
        preset="p50",
        disable_light=False,
    )
    connected = False
    started = False
    try:
        print(f"Connecting to Muse at {args.address} via MuseLSL2…", flush=True)
        connected = muse.connect()
        if not connected:
            raise RuntimeError("MuseLSL2 could not connect")
        muse.start()
        started = True
        latest_roll = 0.0
        latest_gyro_mag = 0.0

        def process_imu() -> list[tuple[float, float, float]]:
            nonlocal latest_roll, latest_gyro_mag
            values: list[tuple[float, float, float]] = []
            while accel_chunks and gyro_chunks:
                accel = accel_chunks.pop(0)
                gyro = gyro_chunks.pop(0)
                for accel_sample, gyro_sample in zip(accel, gyro):
                    latest_gyro_mag = float(np.linalg.norm(gyro_sample))
                    latest_roll = orientation.update(accel_sample, gyro_sample)
                    if orientation.ready:
                        forward = float(accel_sample[0])
                        values.append((latest_roll, orientation.pitch, forward))
                        if nod_detector is not None:
                            nod_detector.add(forward)
            return values

        def collect(seconds: float, component: int = 0) -> list[float]:
            values: list[float] = []
            deadline = time.monotonic() + seconds
            while running and time.monotonic() < deadline:
                muse.adapter.pump(0.018)
                values.extend(value[component] for value in process_imu())
            return values

        print("Connected. Hold your head centered and still for 5 seconds.", flush=True)
        display.send(pose_frame("REST"))
        settle_deadline = time.monotonic() + 9.0
        while running and not orientation.ready and time.monotonic() < settle_deadline:
            muse.adapter.pump(0.018)
            process_imu()
        if not orientation.ready:
            raise RuntimeError("not enough synchronized IMU samples to calibrate")

        print(
            "TILT CALIBRATION: slowly sweep left-center-right-center twice.",
            flush=True,
        )
        display.send(pose_frame("SWEEP"))
        roll_sweep = collect(8.0)
        if len(roll_sweep) < 80:
            raise RuntimeError("too few tilt calibration samples")
        left_roll = float(np.percentile(roll_sweep, 5))
        right_roll = float(np.percentile(roll_sweep, 95))
        if right_roll - left_roll < 8.0:
            raise RuntimeError("tilt sweep was too small; please retry")

        # Measure neutral immediately before the down pose. Unlike Euler pitch,
        # forward acceleration does not wrap or couple strongly with head roll.
        print("CENTER", flush=True)
        display.send(pose_frame("REST"))
        center_forward_samples = collect(2.5, component=2)

        nod_forward = 0.0
        if args.rotation == "nod":
            print("NOD CALIBRATION: perform three slow down-and-back nods.", flush=True)
            display.send(pose_frame("NOD"))
            nod_samples = collect(8.0, component=2)
            if len(nod_samples) < 80:
                raise RuntimeError("too few nod-pose samples; please retry")
            center_forward = float(np.median(center_forward_samples))
            low = float(np.percentile(nod_samples, 5))
            high = float(np.percentile(nod_samples, 95))
            nod_forward = (
                low if abs(low - center_forward) > abs(high - center_forward) else high
            )

        print("CENTER — calibration complete.", flush=True)
        display.send(pose_frame("REST"))
        collect(2.0)
        mapper = RollColumnMapper(left_roll, right_roll)
        if args.rotation == "nod":
            nod_detector = AccelNodDetector(center_forward, nod_forward)
        print(
            f"Fused roll range: left={left_roll:+.1f}° right={right_roll:+.1f}°",
            flush=True,
        )
        if nod_detector is not None:
            print(
                f"Nod accel: center={nod_detector.center:+.3f}g "
                f"down={nod_forward:+.3f}g threshold={nod_detector.activation:.3f}g",
                flush=True,
            )
        print(
            "Lean continuously to choose a column | Quick nod down and back = rotate"
            if args.rotation == "nod" else
            "Lean continuously to choose a column | Double blink = rotate"
            if args.rotation == "blink" and not args.no_blink_rotate else
            "Lean continuously to choose a column | Rotation disabled",
            flush=True,
        )
        # Rotation is binary, so retain the profile's artifact-distance gate
        # but relax left/right/both separation: fit asymmetry often makes a
        # two-eye blink resemble a unilateral wink.
        detector: OnlineWinkDetector | None = None
        if profile is not None:
            profile.max_distance = max(profile.max_distance, 8.5)
            detector = OnlineWinkDetector(
                profile,
                confidence=0.0,
                refractory_seconds=0.30,
                enforce_profile_distance=False,
            )
        eeg_chunks.clear()
        display.send(game.frame())

        next_drop = time.monotonic() + args.drop_seconds
        last_send = 0.0
        dirty = True
        game_over_at: float | None = None
        last_control = 0.0
        last_target: int | None = None
        blink_flash_until = 0.0
        blink_flash_color = [220, 170, 35]
        pending_blink_at: float | None = None

        while running:
            # Pump CoreBluetooth's asyncio loop; callbacks populate eeg_chunks.
            muse.adapter.pump(0.018)
            now = time.monotonic()
            while eeg_chunks:
                chunk = eeg_chunks.pop(0)
                if detector is not None:
                    detector.add(chunk[1:3])
            process_imu()

            if not game.game_over:
                piece_width = max(x for x, _ in game.piece.cells) + 1
                target = mapper.target(latest_roll, WIDTH - piece_width + 1)
                if target != last_target:
                    print(
                        f"COLUMN {target + 1} roll={latest_roll:+.1f}°",
                        flush=True,
                    )
                    last_target = target
                while game.piece.x < target and game.move(1):
                    dirty = True
                while game.piece.x > target and game.move(-1):
                    dirty = True

            wink, distances, confidence = (
                detector.detect() if detector is not None else (None, None, 0.0)
            )
            control: str | None = None
            source = ""
            nod, nod_forward_now = (
                nod_detector.detect() if nod_detector is not None else (False, 0.0)
            )
            if (
                nod and args.rotation == "nod"
                and now - last_control >= 0.70
            ):
                control = "rotate"
                source = f"nod return ({nod_forward_now:+.3f}g)"
            if wink and latest_gyro_mag < 30.0 and args.rotation == "blink":
                if args.swap_eyes and wink in ("left", "right"):
                    wink = "right" if wink == "left" else "left"
                separation = (
                    now - pending_blink_at if pending_blink_at is not None else None
                )
                if (
                    separation is not None
                    and 0.32 <= separation <= 1.00
                    and now - last_control >= 0.70
                    and not args.no_blink_rotate
                ):
                    control = "rotate"
                    source = f"double blink ({separation:.2f}s)"
                    pending_blink_at = None
                elif separation is None or separation > 1.00:
                    pending_blink_at = now
                    blink_flash_until = now + 0.16
                    blink_flash_color = [220, 170, 35]
                    dirty = True
                    print(
                        f"BLINK 1/2 confidence={confidence:.0%}", flush=True
                    )

            if control:
                print(
                    f"{control.upper():6s} {source}",
                    flush=True,
                )
                last_control = now
                rotated = game.rotate()
                # Always acknowledge a valid blink, including on an O-piece
                # whose appearance cannot change when rotated.
                blink_flash_until = now + 0.20
                blink_flash_color = [35, 220, 150]
                dirty = True
                print("  rotated" if rotated else "  acknowledged; shape unchanged", flush=True)

            if blink_flash_until and now >= blink_flash_until:
                blink_flash_until = 0.0
                dirty = True

            if now >= next_drop and not game.game_over:
                dirty |= game.tick()
                next_drop = now + max(0.18, args.drop_seconds - game.lines * 0.018)
            if game.game_over and game_over_at is None:
                game_over_at = now
                dirty = True
                print(f"Game over — score {game.score}, lines {game.lines}", flush=True)
            if game_over_at is not None and now - game_over_at >= 2.5:
                game = Tetris(args.seed)
                game_over_at = None
                next_drop = now + args.drop_seconds
                dirty = True
                last_target = None
                print("New game", flush=True)
            if dirty and now - last_send >= 0.075:
                frame = game.frame()
                if blink_flash_until:
                    frame[0][0] = blink_flash_color.copy()
                    frame[0][-1] = blink_flash_color.copy()
                display.send(frame)
                last_send = now
                dirty = False
        return 0
    except Exception as exc:
        print(f"Tetris stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if started:
            try:
                muse.stop()
                muse.adapter.pump(0.1)
            except Exception:
                pass
        if connected:
            try:
                muse.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
