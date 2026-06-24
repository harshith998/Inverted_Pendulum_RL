# STM32 Firmware Protocol

The Python controller expects the Nucleo firmware to run the real hardware
loop. The firmware should:

- Decode the motor encoder for cart position.
- Decode three joint encoders.
- Read left and right limit switches.
- Read emergency stop state.
- Apply signed PWM to the VNH5019 motor driver.
- Immediately command zero PWM if communication is lost.

## Sensor Packet

Send one ASCII line every control tick:

```text
S,<time_s>,<cart_count>,<j1_count>,<j2_count>,<j3_count>,<left_limit>,<right_limit>,<estop_ok>
```

Example:

```text
S,12.345000,1024,-15,7,31,0,0,1
```

Boolean fields are `0` or `1`.

## Command Packet

Python sends:

```text
C,<pwm>
```

where `pwm` is a signed value in `[-1, 1]`.

Example:

```text
C,-0.125000
```

## Firmware Safety Requirements

The STM32 should not blindly trust the Python process. It should independently
enforce:

- zero PWM when e-stop is open,
- zero or inward-only PWM when a limit switch is active,
- zero PWM if no command packet arrives within a short timeout,
- a conservative maximum PWM during early testing,
- valid direction/PWM pin states for the VNH5019.

## VNH5019 Control

A typical signed-PWM mapping is:

- positive command: INA high, INB low, PWM duty = abs(command)
- negative command: INA low, INB high, PWM duty = abs(command)
- zero command: PWM duty = 0

Confirm the exact VNH5019 carrier pin names against the Pololu documentation
before wiring.
