# Traditional LQR Controller

This folder is for the real robotic cart + 3-link pendulum controller.
It is not a simulator.

The intended split is:

- STM32 Nucleo firmware reads encoders, drives the VNH5019 motor driver, checks
  limit switches, and streams one sensor packet per control tick.
- This Python controller computes the physical model, solves the discrete LQR
  gain by Riccati iteration, checks safety limits, and sends a force command.

The code is deliberately written without a black-box `lqr()` call. The only
linear algebra primitive used for the gain is `numpy.linalg.solve`.

## State Convention

The controller state is:

```text
[p, theta1, theta2, theta3, p_dot, theta1_dot, theta2_dot, theta3_dot]
```

where the joint angles are relative encoder angles measured from the upright
zero position.

## Files

- `config.yaml` - physical parameters, encoder calibration, LQR weights, safety
  limits, and serial settings.
- `model.py` - ground-up mass matrix, gravity stiffness matrix, continuous
  state-space model, and Euler discretization.
- `riccati.py` - discrete Riccati iteration and LQR gain computation.
- `encoders.py` - count-to-position and count-to-angle conversion.
- `hardware.py` - serial protocol boundary for the STM32/Nucleo.
- `controller.py` - safety checks, state assembly, and force calculation.
- `run_controller.py` - real-time control loop entry point.
- `firmware_protocol.md` - packet format the STM32 firmware should implement.

## First Bring-Up Order

1. Run the motor with the pendulum removed and low PWM limits.
2. Confirm cart encoder direction and meters-per-count.
3. Confirm each joint encoder direction and zero offset.
4. Confirm endstop polarity and emergency stop behavior.
5. Run `run_controller.py --dry-run` to read sensors without commanding force.
6. Enable low force limits first, then tune upward carefully.

Do not close the LQR loop until encoder signs are verified. Wrong signs turn
stabilizing feedback into destabilizing feedback.
