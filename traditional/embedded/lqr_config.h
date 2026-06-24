#ifndef LQR_CONFIG_H
#define LQR_CONFIG_H

#define N_LINKS 3
#define NQ (N_LINKS + 1)
#define NX (2 * NQ)

#define CONTROL_DT_S 0.005f
#define GRAVITY_MPS2 9.81f

/* Replace these after weighing/measuring the real build. */
#define CART_MASS_KG 1.0f

static const float LINK_LENGTH_M[N_LINKS] = {0.30f, 0.30f, 0.30f};
static const float LINK_MASS_KG[N_LINKS] = {0.10f, 0.10f, 0.10f};

/* Encoder calibration placeholders. */
#define CART_COUNTS_PER_METER 30000.0f
#define CART_SIGN 1.0f
#define CART_ZERO_COUNT 0

static const float JOINT_COUNTS_PER_REV[N_LINKS] = {2400.0f, 2400.0f, 2400.0f};
static const float JOINT_SIGN[N_LINKS] = {1.0f, 1.0f, 1.0f};
static const int JOINT_ZERO_COUNT[N_LINKS] = {0, 0, 0};

/* Bryson-rule LQR starting point. */
static const float MAX_STATE[NX] = {
    0.40f,
    0.10f, 0.10f, 0.10f,
    1.50f,
    3.0f, 3.0f, 3.0f
};

#define MAX_LQR_FORCE_N 12.0f
#define MAX_MOTOR_FORCE_N 20.0f
#define PWM_PER_NEWTON 0.04f
#define MAX_PWM_ABS 0.35f

#define RICCATI_MAX_ITERS 10000
#define RICCATI_TOLERANCE 1.0e-5f

#define RAIL_LIMIT_M 0.25f
#define ANGLE_LIMIT_RAD 0.35f
#define CART_VEL_LIMIT_MPS 3.0f
#define ANG_VEL_LIMIT_RADPS 12.0f

#endif
