"""RBF parameter schedule plus a small state-dependent rescue scale.

This remains a fixed learned policy at evaluation time. The extra rescue terms
only use the current observed state and are optimized from environment returns;
there are no LQR actions, gains, solves, or controller recomputations.
"""

import torch
import torch.nn as nn

from .cgat_rbf_force_schedule import CGATRBFForceSchedulePPOPolicy


class CGATRBFStateForceSchedulePPOPolicy(CGATRBFForceSchedulePPOPolicy):
    """RBF force schedule with bounded angle/velocity-dependent scale."""

    VARIANT = "rbf_state_force_schedule"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.state_angle_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.state_vel_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.state_angle_threshold = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))
        self.state_log_clip = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def _state_rescue_scale(self, obs: dict) -> torch.Tensor:
        state, _ = self._physical_state_and_params(obs)
        theta = state[:, 2:2 + self.max_links]
        theta_dot = state[:, 2 + self.max_links:2 + 2 * self.max_links]
        max_theta = theta.abs().max(dim=1, keepdim=True).values
        max_vel = theta_dot.abs().max(dim=1, keepdim=True).values
        threshold = self.state_angle_threshold.clamp(0.02, 0.65)
        angle_excess = torch.relu(max_theta - threshold)
        vel_norm = (max_vel / 10.0).clamp(0.0, 2.0)
        log_scale = self.state_angle_gain * angle_excess + self.state_vel_gain * vel_norm
        clip = self.state_log_clip.abs().clamp(0.02, 0.90)
        return torch.exp(log_scale.clamp(-clip, clip))

    def _force_scale(self, obs: dict) -> torch.Tensor:
        return super()._force_scale(obs) * self._state_rescue_scale(obs)
