#!/usr/bin/env python3
"""Measure personal Muse AF7/AF8 wink signatures using simulator color cues."""

from __future__ import annotations

import time
import json

import numpy as np
from MuseLSL2.muse import Muse

from tetris import HEIGHT, WIDTH, WebDisplay
from tetris_muselsl import DEFAULT_ADDRESS
from advanced_winks import train_profile


def solid(rgb: tuple[int, int, int]) -> list[list[list[int]]]:
    return [[list(rgb) for _ in range(WIDTH)] for _ in range(HEIGHT)]


GLYPHS = {
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
}

# Icons are intentionally static and fill the narrow 9 × 17 façade. They are
# understandable before the scrolling text has had time to cross the display.
SYMBOLS = {
    "REST": (
        "001100110",
        "001100110",
        "001100110",
        "001100110",
        "001100110",
        "001100110",
        "001100110",
    ),
    "LEFT": (
        "000100000",
        "001100000",
        "011111110",
        "111111111",
        "011111110",
        "001100000",
        "000100000",
    ),
    "RIGHT": (
        "000001000",
        "000001100",
        "011111110",
        "111111111",
        "011111110",
        "000001100",
        "000001000",
    ),
    "BOTH": (
        "000000000",
        "011101110",
        "100010001",
        "101010101",
        "100010001",
        "011101110",
        "000000000",
    ),
}

BACKGROUND = (28, 31, 39)
CUE_COLORS = {
    "REST": (105, 115, 130),
    "LEFT": (35, 95, 210),
    "RIGHT": (205, 55, 50),
    "BOTH": (145, 155, 170),
}


def symbol_frame(symbol: str, pulse: float = 1.0) -> list[list[list[int]]]:
    """Render one large semantic cue; pulse demonstrates the blink rhythm."""
    frame = solid(BACKGROUND)
    color = tuple(round(channel * pulse) for channel in CUE_COLORS[symbol])
    pattern = SYMBOLS[symbol]
    x0 = (WIDTH - len(pattern[0])) // 2
    y0 = (HEIGHT - len(pattern)) // 2
    for y, row in enumerate(pattern):
        for x, bit in enumerate(row):
            if bit == "1":
                frame[y0 + y][x0 + x] = list(color)
    return frame


def label_frame(
    word: str, rgb: tuple[int, int, int], progress: float = 0.0
) -> list[list[list[int]]]:
    """Scroll a conventional horizontal 5x7 word across the narrow facade."""
    frame = solid((0, 0, 0))
    canvas_width = len(word) * 6 - 1
    offset = round(WIDTH - min(max(progress, 0.0), 1.0) * (canvas_width + WIDTH))
    y0 = (HEIGHT - 7) // 2
    for letter_index, letter in enumerate(word):
        for gy, row in enumerate(GLYPHS[letter]):
            for gx, bit in enumerate(row):
                x = offset + letter_index * 6 + gx
                if bit == "1" and 0 <= x < WIDTH:
                    frame[y0 + gy][x] = list(rgb)
    return frame


def analyze(name: str, chunks: list[np.ndarray]) -> tuple[float, float]:
    data = np.concatenate(chunks, axis=1)
    # First differences suppress slow electrode/contact drift and emphasize
    # the rapid EOG/EMG transient created by a wink.
    left = np.diff(data[1], prepend=data[1, 0])
    right = np.diff(data[2], prepend=data[2, 0])
    width = 128  # 0.5-second non-overlapping windows
    observations: list[tuple[float, float, float]] = []
    for start in range(0, data.shape[1] - width + 1, width):
        left_amp = float(np.percentile(np.abs(left[start:start + width]), 97))
        right_amp = float(np.percentile(np.abs(right[start:start + width]), 97))
        strength = max(left_amp, right_amp)
        score = float(np.log((left_amp + 20.0) / (right_amp + 20.0)))
        observations.append((strength, score, left_amp, right_amp))
    strongest = sorted(observations, reverse=True)[:5]
    scores = [item[1] for item in strongest]
    strengths = [item[0] for item in strongest]
    print(f"\n{name} strongest windows:")
    for strength, score, left_amp, right_amp in strongest:
        print(
            f"  AF7={left_amp:6.0f}  AF8={right_amp:6.0f}  "
            f"log-ratio={score:+.3f}  peak={strength:6.0f}"
        )
    return float(np.median(scores)), float(np.median(strengths))


def main() -> int:
    display = WebDisplay("hazel-yak", "https://sundai.willsarg.com/api")
    samples: dict[str, list[np.ndarray]] = {
        "baseline": [], "left": [], "right": [], "both": []
    }
    active_phase = "baseline"
    recording = False

    def on_eeg(data: np.ndarray, _timestamps: np.ndarray) -> None:
        if not recording:
            return
        chunk = np.array(data[:4], copy=True)
        if np.all(chunk[1] == 0) or np.all(chunk[2] == 0):
            return
        samples[active_phase].append(chunk)

    muse = Muse(
        address=DEFAULT_ADDRESS,
        callback_eeg=on_eeg,
        preset="p50",
    )
    print("Connecting for wink calibration…", flush=True)
    muse.connect()
    muse.start()
    try:
        phases = (
            ("baseline", "REST", 3.0, "HOLD STILL"),
            ("left", "LEFT", 7.0, "WINK LEFT 4-6 TIMES — ABOUT ONCE PER SECOND"),
            ("baseline", "REST", 2.0, "REST"),
            ("right", "RIGHT", 7.0, "WINK RIGHT 4-6 TIMES — ABOUT ONCE PER SECOND"),
            ("baseline", "REST", 2.0, "REST"),
            ("both", "BOTH", 7.0, "BLINK BOTH 4-6 TIMES — ABOUT ONCE PER SECOND"),
        )
        for name, symbol, duration, label in phases:
            # Preview gesture cues without recording, so the participant has
            # time to understand the symbol before their first required blink.
            if name != "baseline":
                recording = False
                print(f"GET READY: {symbol}", flush=True)
                display.send(symbol_frame(symbol, 0.72))
                preview_deadline = time.monotonic() + 2.0
                while time.monotonic() < preview_deadline:
                    muse.adapter.pump(0.04)
            active_phase = name
            recording = True
            print(label, flush=True)
            started = time.monotonic()
            deadline = started + duration
            next_frame = 0.0
            while (now := time.monotonic()) < deadline:
                if now >= next_frame:
                    # A gentle one-Hz pulse models the requested cadence while
                    # keeping the symbol continuously visible.
                    phase = (now - started) % 1.0
                    pulse = 1.0 if phase < 0.42 else 0.66
                    display.send(symbol_frame(symbol, pulse))
                    next_frame = now + 0.14
                muse.adapter.pump(0.02)
            recording = False
        display.send(solid((25, 135, 55)))
    finally:
        muse.stop()
        muse.adapter.pump(0.1)
        muse.disconnect()

    left_score, left_peak = analyze("LEFT", samples["left"])
    right_score, right_peak = analyze("RIGHT", samples["right"])
    both_score, both_peak = analyze("BOTH", samples["both"])
    _, baseline_peak = analyze("BASELINE", samples["baseline"])
    print("\nRecommended calibration:")
    print(f"  left_score={left_score:+.4f}")
    print(f"  right_score={right_score:+.4f}")
    print(f"  both_score={both_score:+.4f}")
    print(
        f"  threshold={max(15.0, min(left_peak, right_peak, both_peak) * 0.45, baseline_peak * 1.35):.1f}"
    )
    labeled = {
        name: np.concatenate(samples[name], axis=1)[1:3]
        for name in ("left", "right", "both")
    }
    baseline = np.concatenate(samples["baseline"], axis=1)[1:3]
    profile, counts = train_profile(baseline, labeled)
    print("\nAdvanced filtered epoch model:")
    print(f"  accepted_events={counts}")
    print(f"  training_accuracy={profile.accuracy:.1%}")
    print(f"  normalized_event_threshold={profile.event_threshold:.3f}")
    print(f"  classifier_max_distance={profile.max_distance:.3f}")
    print("PROFILE_JSON=" + json.dumps({
        "scales": profile.scales.tolist(),
        "feature_mean": profile.feature_mean.tolist(),
        "feature_scale": profile.feature_scale.tolist(),
        "centroids": profile.centroids.tolist(),
        "event_threshold": profile.event_threshold,
        "accuracy": profile.accuracy,
        "max_distance": profile.max_distance,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
