from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalParams:
    cart_mass_kg: float
    link_lengths_m: np.ndarray
    link_masses_kg: np.ndarray
    gravity: float = 9.81

    @property
    def n_links(self) -> int:
        return int(len(self.link_lengths_m))


def absolute_angle_matrix(n_links: int) -> np.ndarray:
    """Map relative joint angles to absolute link angles."""
    return np.tril(np.ones((n_links, n_links), dtype=float))


def mass_matrix_upright(params: PhysicalParams) -> np.ndarray:
    """Return the linearized generalized-coordinate mass matrix at upright.

    Coordinates are q = [cart_position, theta_1, ..., theta_n].
    """
    n = params.n_links
    lengths = np.asarray(params.link_lengths_m, dtype=float)
    masses = np.asarray(params.link_masses_kg, dtype=float)

    if n != len(masses):
        raise ValueError("link_lengths_m and link_masses_kg must have same length")

    size = n + 1
    mass_matrix = np.zeros((size, size), dtype=float)
    mass_matrix[0, 0] = params.cart_mass_kg

    for i in range(n):
        # Horizontal COM velocity near upright:
        # xdot_i = pdot + sum_r coeff[r] * theta_dot_r.
        j_v = np.zeros(size, dtype=float)
        j_v[0] = 1.0
        for r in range(i + 1):
            j_v[r + 1] = lengths[r] if r < i else 0.5 * lengths[i]
        mass_matrix += masses[i] * np.outer(j_v, j_v)

        # Rod rotational kinetic energy around its own center of mass.
        inertia_i = masses[i] * lengths[i] ** 2 / 12.0
        j_w = np.zeros(size, dtype=float)
        j_w[1 : i + 2] = 1.0
        mass_matrix += inertia_i * np.outer(j_w, j_w)

    return mass_matrix


def gravity_stiffness_upright(params: PhysicalParams) -> np.ndarray:
    """Return S where dV/dtheta ~= -S theta near upright."""
    n = params.n_links
    lengths = np.asarray(params.link_lengths_m, dtype=float)
    masses = np.asarray(params.link_masses_kg, dtype=float)

    coefficients = np.zeros(n, dtype=float)
    for r in range(n):
        distal_mass = float(np.sum(masses[r + 1 :]))
        coefficients[r] = params.gravity * lengths[r] * (distal_mass + 0.5 * masses[r])

    e = absolute_angle_matrix(n)
    return e.T @ np.diag(coefficients) @ e


def continuous_state_space(params: PhysicalParams) -> tuple[np.ndarray, np.ndarray]:
    """Build xdot = A x + B u for x=[q, qdot]."""
    n = params.n_links
    nq = n + 1
    state_dim = 2 * nq

    m = mass_matrix_upright(params)
    s = gravity_stiffness_upright(params)

    kq = np.zeros((nq, nq), dtype=float)
    kq[1:, 1:] = s

    bq = np.zeros((nq, 1), dtype=float)
    bq[0, 0] = 1.0

    lower_left = np.linalg.solve(m, kq)
    lower_b = np.linalg.solve(m, bq)

    a = np.zeros((state_dim, state_dim), dtype=float)
    a[:nq, nq:] = np.eye(nq)
    a[nq:, :nq] = lower_left

    b = np.zeros((state_dim, 1), dtype=float)
    b[nq:, :] = lower_b

    return a, b


def discretize_euler(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple small-dt discretization: x[k+1] = Ad x[k] + Bd u[k]."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    ad = np.eye(a.shape[0]) + dt * a
    bd = dt * b
    return ad, bd


def bryson_q_r(max_state: np.ndarray, max_force: float) -> tuple[np.ndarray, np.ndarray]:
    """Build diagonal Q and scalar R from acceptable maximum deviations."""
    max_state = np.asarray(max_state, dtype=float)
    if np.any(max_state <= 0.0):
        raise ValueError("all max_state entries must be positive")
    if max_force <= 0.0:
        raise ValueError("max_force must be positive")
    q = np.diag(1.0 / (max_state ** 2))
    r = np.array([[1.0 / (max_force ** 2)]], dtype=float)
    return q, r
