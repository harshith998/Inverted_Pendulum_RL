# NUCLEO-F446RE Pin Plan

This is the intended real-hardware pinout for the cart + 3-link pendulum.
Confirm every assignment in STM32CubeMX before wiring, then label the physical
headers on the Nucleo board.

Sources used for the plan:

- ST UM1724 Nucleo-64 user manual: Nucleo board connector/power/debug layout.
- ST STM32F446xC/E datasheet: STM32F446RE alternate-function pin table.
- ST RM0390 reference manual: timer encoder mode, PWM mode, USART, GPIO.

## Peripheral Allocation

| Function | STM32 peripheral | MCU pins | Notes |
|---|---:|---|---|
| Cart motor encoder A/B | TIM2 encoder mode | PA0 / PA1 | 32-bit timer, good for cart count range |
| Joint 1 encoder A/B | TIM3 encoder mode | PA6 / PA7 | quadrature x4 |
| Joint 2 encoder A/B | TIM4 encoder mode | PB6 / PB7 | quadrature x4 |
| Joint 3 encoder A/B | TIM1 encoder mode | PA8 / PA9 | TIM1 also supports encoder mode |
| Motor PWM | TIM10 CH1 PWM | PB8 | VNH5019 PWM input |
| Motor direction INA | GPIO output | PC8 | VNH5019 INA |
| Motor direction INB | GPIO output | PC9 | VNH5019 INB |
| Motor enable/diagnostic EN/DIAG | GPIO input/output | PC6 | pull high to enable, read fault if wired that way |
| Left limit switch | GPIO input pullup | PC10 | active-low recommended |
| Right limit switch | GPIO input pullup | PC11 | active-low recommended |
| E-stop OK signal | GPIO input pullup | PC12 | active-low/open means unsafe |
| Debug/telemetry UART TX/RX | USART2 | PA2 / PA3 | ST-LINK virtual COM port |

## Wiring Notes

- Encoder outputs must be pulled up to an MCU-safe voltage. Use 3.3 V pullups
  unless you have verified the exact Nucleo pins and encoder outputs are safe at
  5 V.
- Tie grounds at one star point: motor supply negative, 5 V supply negative,
  VNH5019 GND, Nucleo GND, encoder GND, and slip-ring signal GND.
- Endstops should be wired so broken/disconnected wiring reads unsafe if
  practical.
- The STM32 should enforce zero PWM on e-stop, limit switch, stale command, or
  over-angle even if the host computer is still sending commands.

## Timer Setup in CubeMX

- TIM2, TIM3, TIM4, TIM1: Combined Channels -> Encoder Mode.
- Encoder polarity: start with rising edge, no inversion. Flip sign in software
  if a count increases in the wrong direction.
- TIM10 CH1: PWM Generation CH1.
- USART2: asynchronous, 115200 baud.
- GPIO limit/e-stop pins: input with pullup if switches pull to ground.
- GPIO VNH5019 direction pins: output push-pull.

## Why This Pin Plan Fits the Parts

- The Pololu 37D motor encoder gives A/B quadrature for cart position.
- Each LPD3806 joint encoder gives A/B quadrature.
- The VNH5019 can be driven by one PWM signal plus two direction inputs.
- The Nucleo-F446RE has enough general-purpose timers to decode four encoders
  in hardware, avoiding missed counts from GPIO interrupts.
