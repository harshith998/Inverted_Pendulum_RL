from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from controller import ActuatorCalibration, LQRRobotController, SafetyLimits
from encoders import EncoderCalibration
from hardware import SerialRobotIO
from model import PhysicalParams, bryson_q_r, continuous_state_space, discretize_euler
from riccati import lqr_gain, solve_discrete_riccati_iteration


def load_controller(config_path: str) -> LQRRobotController:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    robot = cfg["robot"]
    lqr = cfg["lqr"]

    params = PhysicalParams(
        cart_mass_kg=float(robot["cart_mass_kg"]),
        link_lengths_m=np.array(robot["link_lengths_m"], dtype=float),
        link_masses_kg=np.array(robot["link_masses_kg"], dtype=float),
        gravity=float(robot["gravity"]),
    )

    a, b = continuous_state_space(params)
    ad, bd = discretize_euler(a, b, float(robot["control_dt"]))

    max_state_cfg = lqr["max_state"]
    max_state = np.array(
        [max_state_cfg["cart_position_m"]]
        + list(max_state_cfg["joint_angle_rad"])
        + [max_state_cfg["cart_velocity_mps"]]
        + list(max_state_cfg["joint_velocity_radps"]),
        dtype=float,
    )
    q, r = bryson_q_r(max_state, float(lqr["max_force_n"]))
    p = solve_discrete_riccati_iteration(
        ad,
        bd,
        q,
        r,
        tolerance=float(lqr["riccati_tolerance"]),
        max_iterations=int(lqr["riccati_max_iterations"]),
    )
    k = lqr_gain(ad, bd, p, r)

    enc = cfg["encoders"]
    encoder_cal = EncoderCalibration(
        cart_counts_per_meter=float(enc["cart_counts_per_meter"]),
        cart_sign=int(enc["cart_sign"]),
        cart_zero_count=int(enc["cart_zero_count"]),
        joint_counts_per_rev=np.array(enc["joint_counts_per_rev"], dtype=float),
        joint_signs=np.array(enc["joint_signs"], dtype=float),
        joint_zero_counts=np.array(enc["joint_zero_counts"], dtype=float),
    )

    actuator = cfg["actuator"]
    actuator_cal = ActuatorCalibration(
        max_force_n=float(actuator["max_force_n"]),
        pwm_per_newton=float(actuator["pwm_per_newton"]),
        max_pwm=float(actuator["max_pwm"]),
    )

    safety = cfg["safety"]
    safety_limits = SafetyLimits(
        rail_limit_m=float(safety["rail_limit_m"]),
        angle_limit_rad=float(safety["angle_limit_rad"]),
        velocity_limit_mps=float(safety["velocity_limit_mps"]),
        angular_velocity_limit_radps=float(safety["angular_velocity_limit_radps"]),
    )

    print("Computed LQR gain K:")
    print(k)

    return LQRRobotController(k, encoder_cal, actuator_cal, safety_limits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-robot traditional LQR controller.")
    parser.add_argument("--config", default="traditional/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="read sensors and print commands without sending PWM")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    controller = load_controller(args.config)
    serial_cfg = cfg["serial"]
    safety_cfg = cfg["safety"]

    io = SerialRobotIO(
        port=serial_cfg["port"],
        baudrate=int(serial_cfg["baudrate"]),
        timeout_s=float(serial_cfg["read_timeout_s"]),
    )

    last_packet_wall_time = time.monotonic()
    try:
        while True:
            packet = io.read_sensor_packet()
            now = time.monotonic()
            if now - last_packet_wall_time > float(safety_cfg["stale_packet_timeout_s"]):
                raise RuntimeError("sensor stream went stale")
            last_packet_wall_time = now

            state, force, pwm = controller.command_from_packet(packet)
            if args.dry_run:
                io.command_pwm(0.0)
                print(f"x={np.array2string(state, precision=4)} force={force:+.3f}N pwm={pwm:+.3f}")
            else:
                io.command_pwm(pwm)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        io.stop()
        io.close()


if __name__ == "__main__":
    main()
