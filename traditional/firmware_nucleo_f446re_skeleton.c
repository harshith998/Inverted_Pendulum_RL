/*
 * STM32 NUCLEO-F446RE firmware skeleton for the real cart-pendulum controller.
 *
 * This file is intentionally hardware-facing, not a simulator. It shows the
 * control-loop structure expected by traditional/hardware.py. Generate the
 * concrete CubeMX project separately, then move this logic into your main.c.
 *
 * Required peripherals:
 * - 4 quadrature encoder timers: cart motor encoder + 3 joint encoders
 * - 1 PWM timer channel for VNH5019 PWM
 * - 2 GPIO outputs for VNH5019 direction inputs
 * - 2 GPIO inputs for limit switches
 * - 1 GPIO input for e-stop OK
 * - UART connected to the Python host
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CubeMX should provide these. Replace with your generated includes/handles. */
/* #include "main.h" */
/* extern TIM_HandleTypeDef htim_cart_encoder; */
/* extern TIM_HandleTypeDef htim_j1_encoder; */
/* extern TIM_HandleTypeDef htim_j2_encoder; */
/* extern TIM_HandleTypeDef htim_j3_encoder; */
/* extern TIM_HandleTypeDef htim_pwm; */
/* extern UART_HandleTypeDef huart2; */

#define CONTROL_PERIOD_MS 5U
#define COMMAND_TIMEOUT_MS 50U
#define MAX_PWM_ABS 0.35f

static uint32_t last_command_ms = 0;
static float commanded_pwm = 0.0f;

static float clamp_float(float value, float lo, float hi)
{
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static int32_t read_encoder_count_cart(void)
{
    /* return (int32_t)__HAL_TIM_GET_COUNTER(&htim_cart_encoder); */
    return 0;
}

static int32_t read_encoder_count_j1(void)
{
    /* return (int32_t)__HAL_TIM_GET_COUNTER(&htim_j1_encoder); */
    return 0;
}

static int32_t read_encoder_count_j2(void)
{
    /* return (int32_t)__HAL_TIM_GET_COUNTER(&htim_j2_encoder); */
    return 0;
}

static int32_t read_encoder_count_j3(void)
{
    /* return (int32_t)__HAL_TIM_GET_COUNTER(&htim_j3_encoder); */
    return 0;
}

static bool read_left_limit(void)
{
    /* return HAL_GPIO_ReadPin(LEFT_LIMIT_GPIO_Port, LEFT_LIMIT_Pin) == GPIO_PIN_SET; */
    return false;
}

static bool read_right_limit(void)
{
    /* return HAL_GPIO_ReadPin(RIGHT_LIMIT_GPIO_Port, RIGHT_LIMIT_Pin) == GPIO_PIN_SET; */
    return false;
}

static bool read_estop_ok(void)
{
    /* Prefer fail-safe wiring where open circuit means not OK. */
    /* return HAL_GPIO_ReadPin(ESTOP_OK_GPIO_Port, ESTOP_OK_Pin) == GPIO_PIN_SET; */
    return true;
}

static void motor_set_pwm(float pwm)
{
    pwm = clamp_float(pwm, -MAX_PWM_ABS, MAX_PWM_ABS);

    if (!read_estop_ok()) {
        pwm = 0.0f;
    }
    if (read_left_limit() && pwm < 0.0f) {
        pwm = 0.0f;
    }
    if (read_right_limit() && pwm > 0.0f) {
        pwm = 0.0f;
    }

    /*
     * VNH5019 signed PWM mapping:
     *   pwm > 0: INA=1, INB=0
     *   pwm < 0: INA=0, INB=1
     *   pwm = 0: duty=0
     *
     * Replace these comments with HAL_GPIO_WritePin and TIM compare writes.
     */
    float duty = pwm >= 0.0f ? pwm : -pwm;
    (void)duty;
}

static void send_sensor_packet(uint32_t now_ms)
{
    char line[160];
    float time_s = 0.001f * (float)now_ms;
    int n = snprintf(
        line,
        sizeof(line),
        "S,%.6f,%ld,%ld,%ld,%ld,%d,%d,%d\n",
        time_s,
        (long)read_encoder_count_cart(),
        (long)read_encoder_count_j1(),
        (long)read_encoder_count_j2(),
        (long)read_encoder_count_j3(),
        read_left_limit() ? 1 : 0,
        read_right_limit() ? 1 : 0,
        read_estop_ok() ? 1 : 0
    );

    if (n > 0) {
        /* HAL_UART_Transmit(&huart2, (uint8_t *)line, (uint16_t)n, 5); */
    }
}

static void handle_command_line(const char *line, uint32_t now_ms)
{
    if (line[0] != 'C' || line[1] != ',') {
        return;
    }

    commanded_pwm = clamp_float(strtof(&line[2], NULL), -MAX_PWM_ABS, MAX_PWM_ABS);
    last_command_ms = now_ms;
}

void control_loop_tick(uint32_t now_ms)
{
    if ((now_ms - last_command_ms) > COMMAND_TIMEOUT_MS) {
        commanded_pwm = 0.0f;
    }

    motor_set_pwm(commanded_pwm);
    send_sensor_packet(now_ms);
}

int main(void)
{
    /*
     * HAL_Init();
     * SystemClock_Config();
     * MX_GPIO_Init();
     * MX_USART2_UART_Init();
     * MX_TIMx_Init();
     * HAL_TIM_Encoder_Start(... all encoder timers ...);
     * HAL_TIM_PWM_Start(&htim_pwm, PWM_CHANNEL);
     */

    uint32_t last_tick_ms = 0;
    last_command_ms = 0;

    while (1) {
        /* uint32_t now_ms = HAL_GetTick(); */
        uint32_t now_ms = 0;

        /*
         * Poll or interrupt-fill a UART line buffer, then call:
         * handle_command_line(rx_line, now_ms);
         */

        if ((now_ms - last_tick_ms) >= CONTROL_PERIOD_MS) {
            last_tick_ms = now_ms;
            control_loop_tick(now_ms);
        }
    }
}
