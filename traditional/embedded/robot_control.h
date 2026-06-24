#ifndef ROBOT_CONTROL_H
#define ROBOT_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

#include "lqr_config.h"

typedef struct {
    int32_t cart_count;
    int32_t joint_count[N_LINKS];
    bool left_limit_active;
    bool right_limit_active;
    bool estop_ok;
} RobotSensors;

typedef struct {
    float x[NX];
    float k[NX];
    float previous_q[NQ];
    bool has_previous_q;
} RobotController;

bool robot_controller_init(RobotController *controller);
bool robot_controller_tick(
    RobotController *controller,
    const RobotSensors *sensors,
    float dt_s,
    float *force_n,
    float *pwm
);

#endif
