"""Fused Muse IMU orientation and continuous roll-to-column mapping.

The processing architecture is adapted from Ludentes/muse-vtuber (MIT):
https://github.com/Ludentes/muse-vtuber
"""

from __future__ import annotations

import math

import numpy as np


Quat = tuple[float, float, float, float]  # x, y, z, w


def _multiply(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _conjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def _normalize(q: Quat) -> Quat:
    norm = math.sqrt(sum(value * value for value in q))
    if norm == 0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def _slerp(a: Quat, b: Quat, amount: float) -> Quat:
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        return _normalize(tuple(
            a[index] + amount * (b[index] - a[index]) for index in range(4)
        ))  # type: ignore[arg-type]
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    left = math.sin((1.0 - amount) * angle) / sin_angle
    right = math.sin(amount * angle) / sin_angle
    return tuple(left * a[i] + right * b[i] for i in range(4))  # type: ignore[return-value]


class OneEuroQuaternion:
    """Speed-adaptive quaternion low-pass filter."""

    def __init__(self, min_cutoff: float = 0.3, beta: float = 1.5) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.previous: Quat | None = None
        self.previous_raw: Quat | None = None
        self.previous_time = 0.0

    def filter(self, value: Quat, timestamp: float) -> Quat:
        if self.previous is None or self.previous_raw is None:
            self.previous = value
            self.previous_raw = value
            self.previous_time = timestamp
            return value
        elapsed = timestamp - self.previous_time
        if elapsed <= 0:
            return self.previous
        self.previous_time = timestamp
        if sum(a * b for a, b in zip(value, self.previous_raw)) < 0:
            value = tuple(-item for item in value)  # type: ignore[assignment]
        dot = min(1.0, abs(sum(
            a * b for a, b in zip(value, self.previous_raw)
        )))
        speed = 2.0 * math.acos(dot) / elapsed
        self.previous_raw = value
        effective_speed = 0.0 if speed < 0.15 else speed
        cutoff = self.min_cutoff + self.beta * effective_speed
        tau = 1.0 / (2.0 * math.pi * cutoff)
        amount = 1.0 / (1.0 + tau / elapsed)
        self.previous = _slerp(self.previous, value, amount)
        return self.previous


class Madgwick6:
    """Minimal six-axis Madgwick AHRS."""

    def __init__(self, sample_rate: float = 52.0, beta: float = 0.8) -> None:
        self.sample_rate = sample_rate
        self.beta = beta
        self.q: Quat = (0.0, 0.0, 0.0, 1.0)

    def update(self, gyro_dps: np.ndarray, accel_g: np.ndarray) -> None:
        qx, qy, qz, qw = self.q
        gx, gy, gz = np.radians(gyro_dps)
        qdw = 0.5 * (-qx * gx - qy * gy - qz * gz)
        qdx = 0.5 * (qw * gx + qy * gz - qz * gy)
        qdy = 0.5 * (qw * gy - qx * gz + qz * gx)
        qdz = 0.5 * (qw * gz + qx * gy - qy * gx)

        ax, ay, az = (float(value) for value in accel_g)
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if accel_norm > 0.001:
            ax, ay, az = ax / accel_norm, ay / accel_norm, az / accel_norm
            f1 = 2 * (qx * qz - qw * qy) - ax
            f2 = 2 * (qw * qx + qy * qz) - ay
            f3 = 2 * (0.5 - qx * qx - qy * qy) - az
            # Jᵀf for q=(w,x,y,z). Keeping these quaternion components
            # aligned is essential; the reference implementation mixed the
            # rows and could converge to the equivalent-looking 180° branch.
            sw = (-2 * qy) * f1 + (2 * qx) * f2
            sx = (2 * qz) * f1 + (2 * qw) * f2 + (-4 * qx) * f3
            sy = (-2 * qw) * f1 + (2 * qz) * f2 + (-4 * qy) * f3
            sz = (2 * qx) * f1 + (2 * qy) * f2
            gradient_norm = math.sqrt(sw * sw + sx * sx + sy * sy + sz * sz)
            if gradient_norm:
                sw, sx, sy, sz = (
                    value / gradient_norm for value in (sw, sx, sy, sz)
                )
                qdw -= self.beta * sw
                qdx -= self.beta * sx
                qdy -= self.beta * sy
                qdz -= self.beta * sz

        step = 1.0 / self.sample_rate
        self.q = _normalize((
            qx + qdx * step,
            qy + qdy * step,
            qz + qdz * step,
            qw + qdw * step,
        ))


class HeadOrientation:
    """Estimate physical roll relative to a settled center pose."""

    def __init__(self, sample_rate: float = 52.0, settle_frames: int = 260) -> None:
        self.sample_rate = sample_rate
        self.settle_frames = settle_frames
        self.ahrs = Madgwick6(sample_rate)
        self.smoother = OneEuroQuaternion()
        self.home_inverse: Quat | None = None
        self.frames = 0
        self.roll = 0.0
        self.pitch = 0.0

    @property
    def ready(self) -> bool:
        return self.home_inverse is not None

    @property
    def progress(self) -> float:
        return min(1.0, self.frames / self.settle_frames)

    def update(self, accel: np.ndarray, gyro: np.ndarray) -> float:
        gyro = np.where(np.abs(gyro) < 2.0, 0.0, gyro)
        self.ahrs.update(gyro, accel)
        self.frames += 1
        if not self.ready and self.frames >= self.settle_frames:
            self.recenter()
        if not self.ready:
            return 0.0
        relative = _multiply(self.home_inverse, self.ahrs.q)
        filtered = self.smoother.filter(relative, self.frames / self.sample_rate)
        x, y, z, w = filtered
        self.roll = math.degrees(math.atan2(
            2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
        ))
        pitch_sine = 2 * (w * y - z * x)
        self.pitch = math.degrees(math.asin(float(np.clip(pitch_sine, -1.0, 1.0))))
        return self.roll

    def recenter(self) -> None:
        self.home_inverse = _conjugate(self.ahrs.q)
        self.smoother = OneEuroQuaternion()
        self.roll = 0.0
        self.pitch = 0.0


class RollColumnMapper:
    """Map personalized held roll poses to a stable discrete column."""

    def __init__(
        self, left_roll: float, right_roll: float, hysteresis: float = 0.68
    ) -> None:
        if abs(right_roll - left_roll) < 8.0:
            raise ValueError("left/right poses must differ by at least 8 degrees")
        self.left_roll = left_roll
        self.right_roll = right_roll
        self.hysteresis = hysteresis
        self.column: int | None = None

    def target(self, roll: float, columns: int) -> int:
        if columns <= 1:
            return 0
        position = (
            (roll - self.left_roll)
            / (self.right_roll - self.left_roll)
            * (columns - 1)
        )
        position = float(np.clip(position, 0, columns - 1))
        if self.column is None:
            self.column = int(round(position))
        elif position > self.column + self.hysteresis:
            self.column = min(columns - 1, int(math.floor(position + 0.32)))
        elif position < self.column - self.hysteresis:
            self.column = max(0, int(math.ceil(position - 0.32)))
        return self.column
