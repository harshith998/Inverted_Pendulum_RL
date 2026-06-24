#include "robot_control.h"

#include <math.h>
#include <string.h>

#include "lqr_math.h"

#define PI_F 3.14159265358979323846f
#define TAU_F (2.0f * PI_F)

static float clamp(float value, float lo, float hi)
{
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float wrap_to_pi(float angle)
{
    while (angle > PI_F) angle -= TAU_F;
    while (angle < -PI_F) angle += TAU_F;
    return angle;
}

static void sensors_to_q(const RobotSensors *sensors, float q[NQ])
{
    q[0] = CART_SIGN * ((float)(sensors->cart_count - CART_ZERO_COUNT)) / CART_COUNTS_PER_METER;
    for (int i = 0; i < N_LINKS; ++i) {
        float revolutions =
            JOINT_SIGN[i] *
            ((float)(sensors->joint_count[i] - JOINT_ZERO_COUNT[i])) /
            JOINT_COUNTS_PER_REV[i];
        q[i + 1] = wrap_to_pi(TAU_F * revolutions);
    }
}

static bool state_is_safe(const RobotSensors *sensors, const float x[NX])
{
    if (!sensors->estop_ok) return false;
    if (sensors->left_limit_active || sensors->right_limit_active) return false;
    if (fabsf(x[0]) > RAIL_LIMIT_M) return false;
    if (fabsf(x[NQ]) > CART_VEL_LIMIT_MPS) return false;

    for (int i = 0; i < N_LINKS; ++i) {
        if (fabsf(x[i + 1]) > ANGLE_LIMIT_RAD) return false;
        if (fabsf(x[NQ + i + 1]) > ANG_VEL_LIMIT_RADPS) return false;
    }
    return true;
}

bool robot_controller_init(RobotController *controller)
{
    memset(controller, 0, sizeof(*controller));

    float ad[NX][NX];
    float bd[NX];
    lqr_build_model(ad, bd);
    return lqr_solve_gain(ad, bd, controller->k);
}

bool robot_controller_tick(
    RobotController *controller,
    const RobotSensors *sensors,
    float dt_s,
    float *force_n,
    float *pwm
)
{
    float q[NQ];
    sensors_to_q(sensors, q);

    for (int i = 0; i < NQ; ++i) {
        controller->x[i] = q[i];
    }

    if (!controller->has_previous_q || dt_s <= 0.0f) {
        for (int i = 0; i < NQ; ++i) {
            controller->x[NQ + i] = 0.0f;
        }
        controller->has_previous_q = true;
    } else {
        controller->x[NQ] = (q[0] - controller->previous_q[0]) / dt_s;
        for (int i = 0; i < N_LINKS; ++i) {
            controller->x[NQ + i + 1] = wrap_to_pi(q[i + 1] - controller->previous_q[i + 1]) / dt_s;
        }
    }

    memcpy(controller->previous_q, q, sizeof(q));

    if (!state_is_safe(sensors, controller->x)) {
        *force_n = 0.0f;
        *pwm = 0.0f;
        return false;
    }

    *force_n = lqr_force_command(controller->k, controller->x);
    *pwm = clamp((*force_n) * PWM_PER_NEWTON, -MAX_PWM_ABS, MAX_PWM_ABS);

    if (sensors->left_limit_active && *pwm < 0.0f) *pwm = 0.0f;
    if (sensors->right_limit_active && *pwm > 0.0f) *pwm = 0.0f;

    return true;
}
