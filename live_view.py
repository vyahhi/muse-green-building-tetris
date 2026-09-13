#!/usr/bin/env python3
"""Connect to a Muse 2 over BLE and display its four live EEG channels."""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams


BOARD_ID = BoardIds.MUSE_2_BOARD.value
CHANNEL_NAMES = ("TP9", "AF7", "AF8", "TP10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        help="Optional Muse device name/serial when more than one is nearby",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Seconds of EEG to show (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Bluetooth discovery timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--y-limit",
        type=float,
        default=500.0,
        help="Fixed symmetric EEG Y-axis limit in µV (default: 500)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = BrainFlowInputParams()
    params.timeout = args.timeout
    if args.serial:
        params.serial_number = args.serial

    BoardShim.enable_dev_board_logger()
    board = BoardShim(BOARD_ID, params)
    prepared = False
    streaming = False

    try:
        print("Scanning for Muse 2. Keep it powered on and close to the Mac…")
        board.prepare_session()
        prepared = True
        board.start_stream(450000)
        streaming = True
        print("Connected. Waiting for EEG samples…")

        sample_rate = BoardShim.get_sampling_rate(BOARD_ID)
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        sample_count = max(int(sample_rate * args.seconds), sample_rate)

        deadline = time.monotonic() + 10
        while board.get_board_data_count() < min(sample_rate, sample_count):
            if time.monotonic() >= deadline:
                raise RuntimeError("Connected, but no EEG samples arrived within 10 seconds")
            plt.pause(0.05)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
        fig.canvas.manager.set_window_title("Muse 2 — Live EEG")
        fig.suptitle("Muse 2 — Raw EEG (µV)")
        lines = []
        for axis, name in zip(axes, CHANNEL_NAMES):
            (line,) = axis.plot([], [], linewidth=0.8)
            axis.set_ylabel(name)
            axis.set_ylim(-args.y_limit, args.y_limit)
            axis.grid(alpha=0.2)
            lines.append(line)
        axes[-1].set_xlabel("Seconds ago")

        def update(_frame: int):
            data = board.get_current_board_data(sample_count)
            if data.shape[1] < 2:
                return lines
            seconds = np.arange(data.shape[1]) / sample_rate
            x = seconds - seconds[-1]
            for axis, line, channel in zip(axes, lines, eeg_channels):
                y = data[channel]
                line.set_data(x, y)
                axis.set_xlim(-args.seconds, 0)
            return lines

        animation = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
        fig._muse_animation = animation  # Keep it alive for the lifetime of the window.
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Muse connection failed: {exc}", file=sys.stderr)
        print(
            "Check that the headband is on, not connected to a phone, and that "
            "Bluetooth access is enabled for the app running this command.",
            file=sys.stderr,
        )
        return 1
    finally:
        if streaming:
            board.stop_stream()
        if prepared:
            board.release_session()


if __name__ == "__main__":
    raise SystemExit(main())
