#!/usr/bin/env python3
"""Muse 2 wink-controlled Tetris for the 17x9 MIT Green Building simulator."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams


WIDTH = 9
HEIGHT = 17
BOARD_ID = BoardIds.MUSE_2_BOARD.value
API_ROOT = os.environ.get(
    "GREEN_BUILDING_API_ROOT", "https://sundai.willsarg.com/api"
)
DISPLAY_ID = os.environ.get("GREEN_BUILDING_DISPLAY_ID", "hazel-yak")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = {
    "I": (0, 230, 255),
    "O": (255, 220, 0),
    "T": (190, 70, 255),
    "S": (60, 235, 90),
    "Z": (255, 55, 70),
    "J": (40, 100, 255),
    "L": (255, 145, 25),
}
SHAPES = {
    "I": ((0, 1), (1, 1), (2, 1), (3, 1)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rotated = tuple((-y, x) for x, y in cells)
    min_x = min(x for x, _ in rotated)
    min_y = min(y for _, y in rotated)
    return tuple((x - min_x, y - min_y) for x, y in rotated)


@dataclass
class Piece:
    kind: str
    cells: tuple[tuple[int, int], ...]
    x: int
    y: int = 0


class Tetris:
    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.board: list[list[str | None]] = [[None] * WIDTH for _ in range(HEIGHT)]
        self.score = 0
        self.lines = 0
        self.game_over = False
        self.piece = self._new_piece()

    def _new_piece(self) -> Piece:
        kind = self.random.choice(tuple(SHAPES))
        cells = SHAPES[kind]
        piece_width = max(x for x, _ in cells) + 1
        return Piece(kind, cells, (WIDTH - piece_width) // 2)

    def _valid(self, piece: Piece, dx: int = 0, dy: int = 0,
               cells: tuple[tuple[int, int], ...] | None = None) -> bool:
        for px, py in cells or piece.cells:
            x, y = piece.x + dx + px, piece.y + dy + py
            if x < 0 or x >= WIDTH or y >= HEIGHT:
                return False
            if y >= 0 and self.board[y][x] is not None:
                return False
        return True

    def move(self, dx: int) -> bool:
        if not self.game_over and self._valid(self.piece, dx=dx):
            self.piece.x += dx
            return True
        return False

    def rotate(self) -> bool:
        if self.game_over or self.piece.kind == "O":
            return False
        cells = rotate_cells(self.piece.cells)
        for kick in (0, -1, 1, -2, 2):
            if self._valid(self.piece, dx=kick, cells=cells):
                self.piece.cells = cells
                self.piece.x += kick
                return True
        return False

    def tick(self) -> bool:
        if self.game_over:
            return False
        if self._valid(self.piece, dy=1):
            self.piece.y += 1
            return True
        for px, py in self.piece.cells:
            x, y = self.piece.x + px, self.piece.y + py
            if y < 0:
                self.game_over = True
                return True
            self.board[y][x] = self.piece.kind
        full = [row for row in self.board if all(cell is not None for cell in row)]
        if full:
            count = len(full)
            self.board = [[None] * WIDTH for _ in range(count)] + [
                row for row in self.board if not all(cell is not None for cell in row)
            ]
            self.lines += count
            self.score += (0, 100, 300, 500, 800)[count]
        self.piece = self._new_piece()
        if not self._valid(self.piece):
            self.game_over = True
        return True

    def frame(self) -> list[list[list[int]]]:
        frame = [[list(BLACK) for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for y, row in enumerate(self.board):
            for x, kind in enumerate(row):
                if kind:
                    frame[y][x] = list(COLORS[kind])
        if not self.game_over:
            for px, py in self.piece.cells:
                x, y = self.piece.x + px, self.piece.y + py
                if 0 <= y < HEIGHT and 0 <= x < WIDTH:
                    frame[y][x] = list(COLORS[self.piece.kind])
        else:
            # Red X: visible game-over state on a very low-resolution facade.
            for y in range(HEIGHT):
                x = round(y * (WIDTH - 1) / (HEIGHT - 1))
                frame[y][x] = [255, 0, 0]
                frame[y][WIDTH - 1 - x] = [255, 0, 0]
        return frame


class WebDisplay:
    def __init__(self, display_id: str, api_root: str) -> None:
        self.url = f"{api_root.rstrip('/')}/i/{display_id}/frame"

    def send(self, frame: list[list[list[int]]]) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(frame, separators=(",", ":")).encode(),
            headers={
                "Content-Type": "application/json",
                # The simulator's Cloudflare policy rejects Python's default UA.
                "User-Agent": "curl/8.7.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status >= 300:
                raise RuntimeError(f"display returned HTTP {response.status}")


class WinkDetector:
    """Classify frontal EOG artifacts using AF7/AF8 amplitude asymmetry."""

    def __init__(self, sample_rate: int, threshold_uv: float,
                 dominance: float, swap_eyes: bool) -> None:
        self.sample_rate = sample_rate
        self.threshold_uv = threshold_uv
        self.dominance = dominance
        self.swap_eyes = swap_eyes
        self.left = np.empty(0)
        self.right = np.empty(0)
        self.armed = True
        self.last_event = 0.0

    def add(self, left: np.ndarray, right: np.ndarray) -> None:
        keep = int(self.sample_rate * 1.5)
        self.left = np.concatenate((self.left, left))[-keep:]
        self.right = np.concatenate((self.right, right))[-keep:]

    def detect(self, now: float) -> tuple[str | None, float, float]:
        window = int(self.sample_rate * 0.32)
        baseline = int(self.sample_rate * 1.2)
        if len(self.left) < baseline:
            return None, 0.0, 0.0
        left_base = np.median(self.left[-baseline:-window])
        right_base = np.median(self.right[-baseline:-window])
        left_amp = float(np.max(np.abs(self.left[-window:] - left_base)))
        right_amp = float(np.max(np.abs(self.right[-window:] - right_base)))
        peak = max(left_amp, right_amp)

        if peak < self.threshold_uv * 0.28:
            self.armed = True
        if (not self.armed or peak < self.threshold_uv
                or now - self.last_event < 0.55):
            return None, left_amp, right_amp

        self.armed = False
        self.last_event = now
        asymmetry = (left_amp - right_amp) / max(left_amp + right_amp, 1.0)
        if asymmetry >= self.dominance:
            event = "left"
        elif asymmetry <= -self.dominance:
            event = "right"
        else:
            event = "both"
        if self.swap_eyes and event in ("left", "right"):
            event = "right" if event == "left" else "left"
        return event, left_amp, right_amp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", default=DISPLAY_ID)
    parser.add_argument("--api-root", default=API_ROOT)
    parser.add_argument("--serial", default="Muse-2D36")
    parser.add_argument("--threshold", type=float, default=180.0,
                        help="Minimum blink amplitude in µV (default: 180)")
    parser.add_argument("--dominance", type=float, default=0.16,
                        help="AF7/AF8 asymmetry for a unilateral wink")
    parser.add_argument("--swap-eyes", action="store_true",
                        help="Swap left/right classifications")
    parser.add_argument("--drop-seconds", type=float, default=0.72)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    display = WebDisplay(args.display, args.api_root)
    game = Tetris(args.seed)
    params = BrainFlowInputParams()
    params.serial_number = args.serial
    params.timeout = 20
    board = BoardShim(BOARD_ID, params)
    eeg = BoardShim.get_eeg_channels(BOARD_ID)
    rate = BoardShim.get_sampling_rate(BOARD_ID)
    detector = WinkDetector(rate, args.threshold, args.dominance, args.swap_eyes)
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    BoardShim.enable_board_logger()
    prepared = False
    streaming = False
    try:
        print(f"Connecting to {args.serial}…", flush=True)
        board.prepare_session()
        prepared = True
        board.start_stream(450000)
        streaming = True
        print("Connected. Hold still for 1.2 seconds, then wink to play.", flush=True)
        print("Left wink = left | Right wink = right | Both eyes = rotate", flush=True)
        display.send(game.frame())
        next_drop = time.monotonic() + args.drop_seconds
        last_send = 0.0
        dirty = True
        game_over_at: float | None = None

        while running:
            now = time.monotonic()
            data = board.get_board_data()
            if data.shape[1]:
                detector.add(data[eeg[1]], data[eeg[2]])  # AF7, AF8
                event, left_amp, right_amp = detector.detect(now)
                if event:
                    print(
                        f"{event.upper():5s}  AF7={left_amp:6.0f}µV "
                        f"AF8={right_amp:6.0f}µV",
                        flush=True,
                    )
                    if event == "left":
                        dirty |= game.move(-1)
                    elif event == "right":
                        dirty |= game.move(1)
                    else:
                        dirty |= game.rotate()

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
                print("New game", flush=True)

            if dirty and now - last_send >= 0.075:
                display.send(game.frame())
                last_send = now
                dirty = False
            time.sleep(0.018)
        return 0
    except Exception as exc:
        print(f"Tetris stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if streaming:
            try:
                board.stop_stream()
            except Exception:
                pass
        if prepared:
            try:
                board.release_session()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
