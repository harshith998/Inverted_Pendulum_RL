from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from encoders import EncoderCalibration, FiniteDifferenceVelocity, cart_count_to_meters, joint_counts_to_radians
from hardware import SensorPacket


@dataclass(frozen=True)
class SafetyLimits:
    rail_limit_m: float
    angle_limit_rad: float
    velocity_limit_mps: float
    angular_velocity_limit_radps: float


@dataclass(frozen=True)
class ActuatorCalibration:
    max_force_n: float
    pwm_per_newton: float
    max_pwm: float


class LQRRobotController:
    def __init__(
        self,
        gain: np.ndarray,
        encoder_calibration: EncoderCalibration,
        actuator_calibration: ActuatorCalibration,
        safety_limits: SafetyLimits,
    ):
        self._gain = np.asarray(gain, dtype=float)
        self._encoder_calibration = encoder_calibration
        self._actuator_calibration = actuator_calibration
        self._safety_limits = safety_limits
        self._velocity_estimator = FiniteDifferenceVelocity(state_dim=4)

    def state_from_packet(self, packet: SensorPacket) -> np.ndarray:
        position = np.zeros(4, dtype=float)
        position[0] = cart_count_to_meters(packet.cart_count, self._encoder_calibration)
        position[1:] = joint_counts_to_radians(np.array(packet.joint_counts), self._encoder_calibration)
        velocity = self._velocity_estimator.update(position, packet.time_s)
        return np.concatenate([position, velocity])

    def force_command(self, state: np.ndarray) -> float:
        force = float(-(self._gain @ state)[0])
        return float(np.clip(force, -self._actuator_calibration.max_force_n, self._actuator_calibration.max_force_n))

    def pwm_from_force(self, force_n: float) -> float:
        pwm = force_n * self._actuator_calibration.pwm_per_newton
        return float(np.clip(pwm, -self._actuator_calibration.max_pwm, self._actuator_calibration.max_pwm))

    def check_safety(self, packet: SensorPacket, state: np.ndarray) -> None:
        if not packet.estop_ok:
            raise RuntimeError("emergency stop is open")
        if packet.left_limit or packet.right_limit:
            raise RuntimeError("cart limit switch is active")
        if abs(state[0]) > self._safety_limits.rail_limit_m:
            raise RuntimeError("cart exceeded software rail limit")
        if np.any(np.abs(state[1:4]) > self._safety_limits.angle_limit_rad):
            raise RuntimeError("joint angle exceeded balance limit")
        if abs(state[4]) > self._safety_limits.velocity_limit_mps:
            raise RuntimeError("cart velocity exceeded safety limit")
        if np.any(np.abs(state[5:8]) > self._safety_limits.angular_velocity_limit_radps):
            raise RuntimeError("joint angular velocity exceeded safety limit")

    def command_from_packet(self, packet: SensorPacket) -> tuple[np.ndarray, float, float]:
        state = self.state_from_packet(packet)
        self.check_safety(packet, state)
        force = self.force_command(state)
        pwm = self.pwm_from_force(force)
        return state, force, pwm
