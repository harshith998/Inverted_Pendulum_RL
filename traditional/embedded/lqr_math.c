#include "lqr_math.h"

#include <math.h>
#include <string.h>

static void mat_zero_nq(float a[NQ][NQ])
{
    memset(a, 0, sizeof(float) * NQ * NQ);
}

static void mat_zero_nx(float a[NX][NX])
{
    memset(a, 0, sizeof(float) * NX * NX);
}

static void solve_nq(const float a_in[NQ][NQ], const float b_in[NQ], float x[NQ])
{
    float a[NQ][NQ];
    float b[NQ];
    memcpy(a, a_in, sizeof(a));
    memcpy(b, b_in, sizeof(b));

    for (int col = 0; col < NQ; ++col) {
        int pivot = col;
        float best = fabsf(a[col][col]);
        for (int row = col + 1; row < NQ; ++row) {
            float candidate = fabsf(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }

        if (pivot != col) {
            for (int j = col; j < NQ; ++j) {
                float tmp = a[col][j];
                a[col][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
            float tmp_b = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp_b;
        }

        float diag = a[col][col];
        for (int row = col + 1; row < NQ; ++row) {
            float factor = a[row][col] / diag;
            for (int j = col; j < NQ; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    for (int row = NQ - 1; row >= 0; --row) {
        float sum = b[row];
        for (int j = row + 1; j < NQ; ++j) {
            sum -= a[row][j] * x[j];
        }
        x[row] = sum / a[row][row];
    }
}

static void build_mass_matrix(float m[NQ][NQ])
{
    mat_zero_nq(m);
    m[0][0] = CART_MASS_KG;

    for (int i = 0; i < N_LINKS; ++i) {
        float jv[NQ] = {0.0f};
        jv[0] = 1.0f;
        for (int r = 0; r <= i; ++r) {
            jv[r + 1] = (r < i) ? LINK_LENGTH_M[r] : 0.5f * LINK_LENGTH_M[i];
        }

        for (int r = 0; r < NQ; ++r) {
            for (int c = 0; c < NQ; ++c) {
                m[r][c] += LINK_MASS_KG[i] * jv[r] * jv[c];
            }
        }

        float inertia = LINK_MASS_KG[i] * LINK_LENGTH_M[i] * LINK_LENGTH_M[i] / 12.0f;
        float jw[NQ] = {0.0f};
        for (int r = 0; r <= i; ++r) {
            jw[r + 1] = 1.0f;
        }
        for (int r = 0; r < NQ; ++r) {
            for (int c = 0; c < NQ; ++c) {
                m[r][c] += inertia * jw[r] * jw[c];
            }
        }
    }
}

static void build_gravity_stiffness(float s[N_LINKS][N_LINKS])
{
    memset(s, 0, sizeof(float) * N_LINKS * N_LINKS);

    float coeff[N_LINKS];
    for (int r = 0; r < N_LINKS; ++r) {
        float distal = 0.0f;
        for (int i = r + 1; i < N_LINKS; ++i) {
            distal += LINK_MASS_KG[i];
        }
        coeff[r] = GRAVITY_MPS2 * LINK_LENGTH_M[r] * (distal + 0.5f * LINK_MASS_KG[r]);
    }

    for (int i = 0; i < N_LINKS; ++i) {
        for (int j = 0; j < N_LINKS; ++j) {
            float sum = 0.0f;
            int start = (i > j) ? i : j;
            for (int r = start; r < N_LINKS; ++r) {
                sum += coeff[r];
            }
            s[i][j] = sum;
        }
    }
}

void lqr_build_model(float ad[NX][NX], float bd[NX])
{
    float m[NQ][NQ];
    float s[N_LINKS][N_LINKS];
    float a[NX][NX];
    float b[NX];

    build_mass_matrix(m);
    build_gravity_stiffness(s);
    mat_zero_nx(a);
    memset(b, 0, sizeof(float) * NX);

    for (int i = 0; i < NQ; ++i) {
        a[i][NQ + i] = 1.0f;
    }

    for (int col = 0; col < NQ; ++col) {
        float rhs[NQ] = {0.0f};
        if (col > 0) {
            for (int r = 0; r < N_LINKS; ++r) {
                rhs[r + 1] = s[r][col - 1];
            }
        }

        float sol[NQ] = {0.0f};
        solve_nq(m, rhs, sol);
        for (int row = 0; row < NQ; ++row) {
            a[NQ + row][col] = sol[row];
        }
    }

    float rhs_b[NQ] = {0.0f};
    rhs_b[0] = 1.0f;
    float sol_b[NQ] = {0.0f};
    solve_nq(m, rhs_b, sol_b);
    for (int row = 0; row < NQ; ++row) {
        b[NQ + row] = sol_b[row];
    }

    for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
            ad[r][c] = (r == c ? 1.0f : 0.0f) + CONTROL_DT_S * a[r][c];
        }
        bd[r] = CONTROL_DT_S * b[r];
    }
}

bool lqr_solve_gain(const float ad[NX][NX], const float bd[NX], float k[NX])
{
    float q[NX][NX] = {{0.0f}};
    float p[NX][NX] = {{0.0f}};
    float p_next[NX][NX] = {{0.0f}};
    float r = 1.0f / (MAX_LQR_FORCE_N * MAX_LQR_FORCE_N);

    for (int i = 0; i < NX; ++i) {
        q[i][i] = 1.0f / (MAX_STATE[i] * MAX_STATE[i]);
        p[i][i] = q[i][i];
    }

    for (int iter = 0; iter < RICCATI_MAX_ITERS; ++iter) {
        float pb[NX] = {0.0f};
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                pb[i] += p[i][j] * bd[j];
            }
        }

        float denom = r;
        for (int i = 0; i < NX; ++i) {
            denom += bd[i] * pb[i];
        }

        float bpa[NX] = {0.0f};
        for (int c = 0; c < NX; ++c) {
            for (int i = 0; i < NX; ++i) {
                bpa[c] += pb[i] * ad[i][c];
            }
        }

        for (int r_i = 0; r_i < NX; ++r_i) {
            for (int c_i = 0; c_i < NX; ++c_i) {
                float atpa = 0.0f;
                for (int i = 0; i < NX; ++i) {
                    for (int j = 0; j < NX; ++j) {
                        atpa += ad[i][r_i] * p[i][j] * ad[j][c_i];
                    }
                }
                p_next[r_i][c_i] = q[r_i][c_i] + atpa - bpa[r_i] * bpa[c_i] / denom;
            }
        }

        float diff = 0.0f;
        for (int r_i = 0; r_i < NX; ++r_i) {
            for (int c_i = 0; c_i < NX; ++c_i) {
                float sym = 0.5f * (p_next[r_i][c_i] + p_next[c_i][r_i]);
                float d = sym - p[r_i][c_i];
                diff += d * d;
                p[r_i][c_i] = sym;
            }
        }

        if (sqrtf(diff) < RICCATI_TOLERANCE) {
            float pb_final[NX] = {0.0f};
            for (int i = 0; i < NX; ++i) {
                for (int j = 0; j < NX; ++j) {
                    pb_final[i] += p[i][j] * bd[j];
                }
            }
            float denom_final = r;
            for (int i = 0; i < NX; ++i) {
                denom_final += bd[i] * pb_final[i];
            }
            for (int c = 0; c < NX; ++c) {
                float num = 0.0f;
                for (int i = 0; i < NX; ++i) {
                    num += pb_final[i] * ad[i][c];
                }
                k[c] = num / denom_final;
            }
            return true;
        }
    }

    return false;
}

float lqr_force_command(const float k[NX], const float x[NX])
{
    float u = 0.0f;
    for (int i = 0; i < NX; ++i) {
        u -= k[i] * x[i];
    }
    if (u > MAX_MOTOR_FORCE_N) return MAX_MOTOR_FORCE_N;
    if (u < -MAX_MOTOR_FORCE_N) return -MAX_MOTOR_FORCE_N;
    return u;
}
