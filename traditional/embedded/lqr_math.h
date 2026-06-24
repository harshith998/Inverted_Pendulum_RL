#ifndef LQR_MATH_H
#define LQR_MATH_H

#include <stdbool.h>

#include "lqr_config.h"

void lqr_build_model(float ad[NX][NX], float bd[NX]);
bool lqr_solve_gain(const float ad[NX][NX], const float bd[NX], float k[NX]);
float lqr_force_command(const float k[NX], const float x[NX]);

#endif
