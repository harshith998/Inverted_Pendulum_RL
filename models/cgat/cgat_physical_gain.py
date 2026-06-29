"""CGAT with neural physical-unit feedback gains.

The actor adds a learned gain map K_phi(params) over the physical state order
used by classical linear controllers:

    [x, theta_1, theta_2, theta_3, xdot, theta_dot_1, theta_dot_2, theta_dot_3]

K_phi is a neural network evaluated once per observation from physical
parameters. There is no LQR solve, oracle gain, or controller recomputation in
the policy.
"""

import torch
import torch.nn as nn

from .cgat_base import CGATBasePPOPolicy
from .cgat_param_residual import (
    _ANG_VEL_MAX,
    _CART_MASS_MIN,
    _CART_MASS_RANGE,
    _CART_VEL_MAX,
    _LEN_MIN,
    _LEN_RANGE,
    _MASS_MIN,
    _MASS_RANGE,
    _RAIL_LIMIT,
)


class CGATPhysicalGainPPOPolicy(CGATBasePPOPolicy):
    """Base CGAT actor plus learned physical-unit feedback gains."""

    VARIANT = "physical_gain"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.max_links = max_links
        self.state_dim = 2 + 2 * max_links
        enhanced_param_dim = 1 + 7 * max_links
        self.physical_gain_net = nn.Sequential(
            nn.Linear(enhanced_param_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.state_dim + 1),
        )
        nn.init.zeros_(self.physical_gain_net[-1].weight)
        nn.init.zeros_(self.physical_gain_net[-1].bias)

    def _state_and_enhanced_params(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node = obs["node_features"].float()
        edge = obs["edge_features"].float()

        cart = node[:, 0]
        x = cart[:, 6:7] * _RAIL_LIMIT
        xdot = cart[:, 7:8] * _CART_VEL_MAX
        cart_mass = cart[:, 8:9] * _CART_MASS_RANGE + _CART_MASS_MIN

        joints = node[:, 1:self.max_links + 1]
        theta = torch.atan2(joints[:, :, 3], joints[:, :, 4])
        theta_dot = joints[:, :, 5] * _ANG_VEL_MAX

        rods = edge[:, 0:2 * self.max_links:2]
        lengths = rods[:, :, 0] * _LEN_RANGE + _LEN_MIN
        masses = rods[:, :, 1] * _MASS_RANGE + _MASS_MIN

        state = torch.cat([x, theta, xdot, theta_dot], dim=1)

        lengths_c = lengths.clamp(min=0.03)
        masses_c = masses.clamp(min=0.02)
        cart_mass_c = cart_mass.clamp(min=0.05)
        params = torch.cat([
            lengths_c,
            masses_c,
            1.0 / lengths_c,
            torch.log(lengths_c),
            torch.log(masses_c),
            masses_c / lengths_c,
            masses_c / (cart_mass_c + masses_c.sum(dim=1, keepdim=True)),
            cart_mass_c,
        ], dim=1)
        return state, params

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        base_mean = self.mean_head(actor_h)
        state, params = self._state_and_enhanced_params(obs)
        gains_and_bias = self.physical_gain_net(params)
        gains = gains_and_bias[:, :self.state_dim]
        bias = gains_and_bias[:, self.state_dim:self.state_dim + 1]
        feedback = (gains * state).sum(dim=1, keepdim=True) + bias
        return base_mean + feedback
