/*
 * Integration notes for CubeMX-generated main.c.
 *
 * This is not meant to compile by itself. The real CubeMX project will provide
 * the HAL includes, peripheral handles, clock setup, and generated init code.
 */

#include "robot_control.h"

static RobotController controller;
static uint32_t previous_tick_ms = 0;

static RobotSensors read_robot_sensors_from_hal(void)
{
    RobotSensors s = {0};

    /*
     * s.cart_count = (int32_t)__HAL_TIM_GET_COUNTER(&htim2);
     * s.joint_count[0] = (int32_t)__HAL_TIM_GET_COUNTER(&htim3);
     * s.joint_count[1] = (int32_t)__HAL_TIM_GET_COUNTER(&htim4);
     * s.joint_count[2] = (int32_t)__HAL_TIM_GET_COUNTER(&htim1);
     *
     * With active-low switch wiring:
     * s.left_limit_active = HAL_GPIO_ReadPin(LEFT_LIMIT_GPIO_Port, LEFT_LIMIT_Pin) == GPIO_PIN_RESET;
     * s.right_limit_active = HAL_GPIO_ReadPin(RIGHT_LIMIT_GPIO_Port, RIGHT_LIMIT_Pin) == GPIO_PIN_RESET;
     * s.estop_ok = HAL_GPIO_ReadPin(ESTOP_OK_GPIO_Port, ESTOP_OK_Pin) == GPIO_PIN_SET;
     */

    return s;
}

static void set_vnh5019_pwm_from_hal(float pwm)
{
    /*
     * float abs_pwm = pwm >= 0.0f ? pwm : -pwm;
     * uint32_t compare = (uint32_t)(abs_pwm * (float)__HAL_TIM_GET_AUTORELOAD(&htim10));
     *
     * if (pwm > 0.0f) {
     *     HAL_GPIO_WritePin(MOTOR_INA_GPIO_Port, MOTOR_INA_Pin, GPIO_PIN_SET);
     *     HAL_GPIO_WritePin(MOTOR_INB_GPIO_Port, MOTOR_INB_Pin, GPIO_PIN_RESET);
     * } else if (pwm < 0.0f) {
     *     HAL_GPIO_WritePin(MOTOR_INA_GPIO_Port, MOTOR_INA_Pin, GPIO_PIN_RESET);
     *     HAL_GPIO_WritePin(MOTOR_INB_GPIO_Port, MOTOR_INB_Pin, GPIO_PIN_SET);
     * } else {
     *     HAL_GPIO_WritePin(MOTOR_INA_GPIO_Port, MOTOR_INA_Pin, GPIO_PIN_RESET);
     *     HAL_GPIO_WritePin(MOTOR_INB_GPIO_Port, MOTOR_INB_Pin, GPIO_PIN_RESET);
     * }
     *
     * __HAL_TIM_SET_COMPARE(&htim10, TIM_CHANNEL_1, compare);
     */
    (void)pwm;
}

void user_setup_after_cube_init(void)
{
    /*
     * HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);
     * HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL);
     * HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL);
     * HAL_TIM_Encoder_Start(&htim1, TIM_CHANNEL_ALL);
     * HAL_TIM_PWM_Start(&htim10, TIM_CHANNEL_1);
     */

    if (!robot_controller_init(&controller)) {
        while (1) {
            set_vnh5019_pwm_from_hal(0.0f);
        }
    }
}

void user_loop_tick_from_main_while(void)
{
    /*
     * uint32_t now_ms = HAL_GetTick();
     * if ((now_ms - previous_tick_ms) < 5U) {
     *     return;
     * }
     */
    uint32_t now_ms = previous_tick_ms + 5U;

    float dt_s = 0.001f * (float)(now_ms - previous_tick_ms);
    previous_tick_ms = now_ms;

    RobotSensors sensors = read_robot_sensors_from_hal();
    float force_n = 0.0f;
    float pwm = 0.0f;

    bool ok = robot_controller_tick(&controller, &sensors, dt_s, &force_n, &pwm);
    if (!ok) {
        pwm = 0.0f;
    }
    set_vnh5019_pwm_from_hal(pwm);
}
