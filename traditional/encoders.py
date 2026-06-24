from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TAU = 2.0 * np.pi


@dataclass(frozen=True)
class EncoderCalibration:
    cart_counts_per_meter: float
    cart_sign: int
    cart_zero_count: int
    joint_counts_per_rev: np.ndarray
    joint_signs: np.ndarray
    joint_zero_counts: np.ndarray


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % TAU - np.pi


def cart_count_to_meters(count: int, cal: EncoderCalibration) -> float:
    return cal.cart_sign * (float(count - cal.cart_zero_count) / cal.cart_counts_per_meter)


def joint_counts_to_radians(counts: np.ndarray, cal: EncoderCalibration) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    raw = cal.joint_signs * (counts - cal.joint_zero_counts) * TAU / cal.joint_counts_per_rev
    return wrap_to_pi(raw)


class FiniteDifferenceVelocity:
    """Estimate velocity from position samples."""

    def __init__(self, state_dim: int):
        self._last_position: np.ndarray | None = None
        self._last_time_s: float | None = None
        self._state_dim = state_dim

    def update(self, position: np.ndarray, time_s: float) -> np.ndarray:
        position = np.asarray(position, dtype=float)
        if self._last_position is None or self._last_time_s is None:
            self._last_position = position.copy()
            self._last_time_s = time_s
            return np.zeros(self._state_dim, dtype=float)

        dt = time_s - self._last_time_s
        if dt <= 0.0:
            return np.zeros(self._state_dim, dtype=float)

        delta = position - self._last_position
        delta[1:] = wrap_to_pi(delta[1:])
        velocity = delta / dt

        self._last_position = position.copy()
        self._last_time_s = time_s
        return velocity
