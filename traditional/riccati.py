from __future__ import annotations

import numpy as np


def solve_discrete_riccati_iteration(
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 10000,
) -> np.ndarray:
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""
    p = q.copy()

    for _ in range(max_iterations):
        rbpb = r + bd.T @ p @ bd
        gain_middle = np.linalg.solve(rbpb, bd.T @ p @ ad)
        p_next = q + ad.T @ p @ ad - ad.T @ p @ bd @ gain_middle
        p_next = 0.5 * (p_next + p_next.T)

        if np.linalg.norm(p_next - p, ord="fro") < tolerance:
            return p_next
        p = p_next

    raise RuntimeError("Riccati iteration did not converge")


def lqr_gain(ad: np.ndarray, bd: np.ndarray, p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return K for u = -Kx using the converged Riccati matrix P."""
    return np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
