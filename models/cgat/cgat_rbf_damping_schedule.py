"""RBF force schedule plus a compact learned damping residual.

The residual is a low-dimensional physical-state feedback term optimized from
environment returns only. It is not an LQR gain, does not solve Riccati
equations, and does not recompute a controller at eval time.
"""

import torch
import torch.nn as nn

from .cgat_rbf_force_schedule import CGATRBFForceSchedulePPOPolicy


class CGATRBFDampingSchedulePPOPolicy(CGATRBFForceSchedulePPOPolicy):
    """RBF force-scheduled CGAT plus bounded learned damping residual."""

    VARIANT = "rbf_damping_schedule"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.damping_coeffs = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        self.damping_limit = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))

    def _damping_features(self, obs: dict) -> torch.Tensor:
        state, params = self._physical_state_and_params(obs)
        cart_x = state[:, 0:1] / 2.5
        cart_v = state[:, 1:2] / 5.0
        theta = state[:, 2:2 + self.max_links]
        theta_dot = state[:, 2 + self.max_links:2 + 2 * self.max_links]
        lengths = params[:, :self.max_links].clamp_min(0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp_min(0.02)
        inertia = masses * lengths.square()
        inertia_w = inertia / inertia.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mass_w = masses / masses.sum(dim=1, keepdim=True).clamp_min(1e-6)
        theta_mean = theta.mean(dim=1, keepdim=True)
        vel_mean = (theta_dot / 10.0).mean(dim=1, keepdim=True)
        theta_mass = (mass_w * theta).sum(dim=1, keepdim=True)
        vel_inertia = (inertia_w * (theta_dot / 10.0)).sum(dim=1, keepdim=True)
        sin_mass = (mass_w * torch.sin(theta)).sum(dim=1, keepdim=True)
        phase = (mass_w * torch.sin(theta) * torch.tanh(theta_dot / 10.0)).sum(
            dim=1, keepdim=True
        )
        return torch.cat([
            cart_x,
            cart_v,
            theta_mean,
            vel_mean,
            theta_mass,
            vel_inertia,
            sin_mass,
            phase,
        ], dim=1).clamp(-3.0, 3.0)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        residual = (self._damping_features(obs) * self.damping_coeffs.view(1, -1)).sum(
            dim=1, keepdim=True
        )
        limit = self.damping_limit.abs().clamp(0.05, 2.5)
        return mean + limit * torch.tanh(residual)
