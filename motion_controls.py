"""Reliable gesture controls from the Muse accelerometer."""

from __future__ import annotations

from collections import deque

import numpy as np


class HeadGestureDetector:
    """Detect latched head tilts relative to a short neutral calibration.

    Muse acceleration axes are X=forward, Y=rightward, Z=downward. A gesture
    fires once, then the head must return near neutral before another fires.
    """

    def __init__(
        self,
        threshold_g: float = 0.20,
        neutral_g: float = 0.075,
        calibration_samples: int = 156,
        smooth_samples: int = 7,
    ) -> None:
        self.threshold_g = threshold_g
        self.neutral_g = neutral_g
        self.calibration_samples = calibration_samples
        self.samples: deque[np.ndarray] = deque(maxlen=calibration_samples)
        self.smooth: deque[np.ndarray] = deque(maxlen=smooth_samples)
        self.baseline: np.ndarray | None = None
        self.armed = True

    @property
    def ready(self) -> bool:
        return self.baseline is not None

    def add(self, samples: np.ndarray) -> None:
        for sample in np.asarray(samples, dtype=float):
            if sample.shape != (3,) or not np.all(np.isfinite(sample)):
                continue
            self.smooth.append(sample)
            if self.baseline is None:
                self.samples.append(sample)
                if len(self.samples) >= self.calibration_samples:
                    self.baseline = np.median(np.asarray(self.samples), axis=0)

    def detect(self) -> tuple[str | None, np.ndarray | None]:
        if self.baseline is None or len(self.smooth) < self.smooth.maxlen:
            return None, None
        current = np.median(np.asarray(self.smooth), axis=0)
        delta = current - self.baseline
        forward, rightward = float(delta[0]), float(delta[1])

        if abs(forward) < self.neutral_g and abs(rightward) < self.neutral_g:
            self.armed = True
            # Slowly follow headband slippage, but only while clearly neutral.
            self.baseline = self.baseline * 0.998 + current * 0.002
            return None, delta
        if not self.armed:
            return None, delta

        event: str | None = None
        if abs(rightward) >= self.threshold_g:
            event = "right" if rightward > 0 else "left"
        if event is not None:
            self.armed = False
        return event, delta


class YawRotateDetector:
    """Turn either way to rotate once, ignoring the return motion."""

    def __init__(
        self,
        threshold_dps: float = 90.0,
        cooldown_samples: int = 55,
    ) -> None:
        self.threshold_dps = threshold_dps
        self.cooldown_samples = cooldown_samples
        self.cooldown = 0
        self.pending = False
        self.last_peak = 0.0

    def add(self, samples: np.ndarray) -> None:
        for sample in np.asarray(samples, dtype=float):
            if sample.shape != (3,) or not np.all(np.isfinite(sample)):
                continue
            self.cooldown = max(0, self.cooldown - 1)
            yaw = float(sample[2])
            if self.cooldown == 0 and abs(yaw) >= self.threshold_dps:
                self.pending = True
                self.last_peak = yaw
                # About one second at the Muse IMU's ~52 Hz rate. This masks
                # the opposite angular velocity when the head returns center.
                self.cooldown = self.cooldown_samples

    def detect(self) -> tuple[bool, float]:
        if not self.pending:
            return False, self.last_peak
        self.pending = False
        return True, self.last_peak


class NodRotateDetector:
    """Detect one deliberate down-and-back nod from gyroscope samples.

    A rotation is emitted only after angular velocity crosses a strong pitch
    threshold and then reverses direction within a short window. Requiring the
    complete gesture avoids triggering on ordinary posture changes and masks
    the return motion as part of the same nod.
    """

    def __init__(
        self,
        threshold_dps: float = 55.0,
        return_threshold_dps: float = 28.0,
        max_gap_samples: int = 42,
        cooldown_samples: int = 36,
        dominance: float = 1.10,
    ) -> None:
        self.threshold_dps = threshold_dps
        self.return_threshold_dps = return_threshold_dps
        self.max_gap_samples = max_gap_samples
        self.cooldown_samples = cooldown_samples
        self.dominance = dominance
        self.cooldown = 0
        self.window = 0
        self.first_sign = 0
        self.pending = False
        self.last_peak = 0.0

    def add(self, samples: np.ndarray) -> None:
        for sample in np.asarray(samples, dtype=float):
            if sample.shape != (3,) or not np.all(np.isfinite(sample)):
                continue
            self.cooldown = max(0, self.cooldown - 1)
            if self.cooldown:
                continue

            roll, pitch, yaw = (float(value) for value in sample)
            if self.window:
                self.window -= 1
                if pitch * self.first_sign <= -self.return_threshold_dps:
                    self.pending = True
                    self.cooldown = self.cooldown_samples
                    self.window = 0
                    self.first_sign = 0
                elif self.window == 0:
                    self.first_sign = 0
                continue

            # A nod should be primarily pitch, not the roll used to select a
            # column or a quick horizontal head turn.
            other_axis = max(abs(roll), abs(yaw))
            if (
                abs(pitch) >= self.threshold_dps
                and abs(pitch) >= other_axis * self.dominance
            ):
                self.first_sign = 1 if pitch > 0 else -1
                self.last_peak = pitch
                self.window = self.max_gap_samples

    def detect(self) -> tuple[bool, float]:
        if not self.pending:
            return False, self.last_peak
        self.pending = False
        return True, self.last_peak


class PitchNodDetector:
    """Recognize a personalized look-down pose followed by return to center."""

    def __init__(
        self,
        calibrated_pitch: float,
        activation_fraction: float = 0.58,
        hold_samples: int = 3,
        neutral_samples: int = 3,
        cooldown_samples: int = 34,
    ) -> None:
        if abs(calibrated_pitch) < 6.0:
            raise ValueError("calibrated nod pitch must be at least 6 degrees")
        self.direction = 1.0 if calibrated_pitch > 0 else -1.0
        self.activation = max(5.0, abs(calibrated_pitch) * activation_fraction)
        self.neutral = min(4.0, self.activation * 0.35)
        self.hold_samples = hold_samples
        self.neutral_samples = neutral_samples
        self.cooldown_samples = cooldown_samples
        self.above = 0
        self.centered = 0
        self.dipped = False
        self.cooldown = 0
        self.pending = False
        self.last_pitch = 0.0

    def add(self, pitch: float) -> None:
        if not np.isfinite(pitch):
            return
        self.last_pitch = float(pitch)
        directed = self.last_pitch * self.direction
        if self.cooldown:
            self.cooldown -= 1
            return
        if not self.dipped:
            self.above = self.above + 1 if directed >= self.activation else 0
            if self.above >= self.hold_samples:
                self.dipped = True
                self.centered = 0
            return
        self.centered = self.centered + 1 if abs(self.last_pitch) <= self.neutral else 0
        if self.centered >= self.neutral_samples:
            self.pending = True
            self.dipped = False
            self.above = 0
            self.centered = 0
            self.cooldown = self.cooldown_samples

    def detect(self) -> tuple[bool, float]:
        if not self.pending:
            return False, self.last_pitch
        self.pending = False
        return True, self.last_pitch


class AccelNodDetector:
    """Detect a calibrated look-down-and-return using forward acceleration."""

    def __init__(
        self,
        center_forward: float,
        down_forward: float,
        activation_fraction: float = 0.68,
        hold_samples: int = 6,
        neutral_samples: int = 5,
        cooldown_samples: int = 70,
    ) -> None:
        self.center = float(center_forward)
        self.delta = float(down_forward - center_forward)
        if abs(self.delta) < 0.06:
            raise ValueError("calibrated down pose differs too little from center")
        self.direction = 1.0 if self.delta > 0 else -1.0
        self.activation = abs(self.delta) * activation_fraction
        self.neutral = max(0.025, abs(self.delta) * 0.20)
        self.hold_samples = hold_samples
        self.neutral_samples = neutral_samples
        self.cooldown_samples = cooldown_samples
        self.above = 0
        self.centered = 0
        self.dipped = False
        self.cooldown = 0
        self.pending = False
        self.last_forward = self.center
        self.smooth: deque[float] = deque(maxlen=5)

    def add(self, forward: float) -> None:
        if not np.isfinite(forward):
            return
        self.smooth.append(float(forward))
        if len(self.smooth) < self.smooth.maxlen:
            return
        self.last_forward = float(np.median(self.smooth))
        offset = self.last_forward - self.center
        directed = offset * self.direction
        if self.cooldown:
            self.cooldown -= 1
            return
        if not self.dipped:
            self.above = self.above + 1 if directed >= self.activation else 0
            if self.above >= self.hold_samples:
                self.dipped = True
                self.centered = 0
            return
        self.centered = self.centered + 1 if abs(offset) <= self.neutral else 0
        if self.centered >= self.neutral_samples:
            self.pending = True
            self.dipped = False
            self.above = 0
            self.centered = 0
            self.cooldown = self.cooldown_samples

    def detect(self) -> tuple[bool, float]:
        if not self.pending:
            return False, self.last_forward
        self.pending = False
        return True, self.last_forward
