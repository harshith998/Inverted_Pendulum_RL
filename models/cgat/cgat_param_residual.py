"""CGAT gain-feedback policy with enhanced parameter-feature residual.

The current best learned policy fails hardest on very short links. Raw length
coordinates are a poor extrapolation basis there, so this variant adds a
zero-initialized learned residual whose gains are conditioned on reciprocal and
log length features, mass/length ratios, and the original physical parameters.

This is still a fixed neural policy at evaluation time: no LQR solve, no oracle
controller, and no runtime controller recomputation.
"""

import torch
import torch.nn as nn

from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy

_RAIL_LIMIT = 2.5
_CART_VEL_MAX = 5.0
_ANG_VEL_MAX = 10.0
_LEN_MIN = 0.3
_LEN_RANGE = 0.9
_MASS_MIN = 0.1
_MASS_RANGE = 1.9
_CART_MASS_MIN = 0.5
_CART_MASS_RANGE = 2.5


class CGATParamResidualPPOPolicy(CGATGainFeedbackPPOPolicy):
    """Gain-feedback CGAT plus enhanced-parameter learned feedback residual."""

    VARIANT = "param_residual"

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
        self.nonlinear_feature_dim = 2 + 6 * max_links
        enhanced_param_dim = 1 + 7 * max_links
        residual_dim = self.state_dim + self.nonlinear_feature_dim

        self.param_residual_net = nn.Sequential(
            nn.Linear(enhanced_param_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, residual_dim + 1),
        )
        nn.init.zeros_(self.param_residual_net[-1].weight)
        nn.init.zeros_(self.param_residual_net[-1].bias)

    def _physical_state_and_params(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state_norm, params_norm = self._state_and_params(obs)

        cart_x = state_norm[:, 0:1] * _RAIL_LIMIT
        cart_v = state_norm[:, 1:2] * _CART_VEL_MAX
        theta = state_norm[:, 2:2 + self.max_links]
        theta_dot = state_norm[:, 2 + self.max_links:] * _ANG_VEL_MAX
        state = torch.cat([cart_x, cart_v, theta, theta_dot], dim=1)

        lengths = params_norm[:, :self.max_links] * _LEN_RANGE + _LEN_MIN
        masses = params_norm[:, self.max_links:2 * self.max_links] * _MASS_RANGE + _MASS_MIN
        cart_mass = (
            params_norm[:, 2 * self.max_links:2 * self.max_links + 1]
            * _CART_MASS_RANGE
            + _CART_MASS_MIN
        )
        params = torch.cat([lengths, masses, cart_mass], dim=1)
        return state, params

    def _enhanced_params(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links].clamp(min=0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp(min=0.02)
        cart_mass = params[:, 2 * self.max_links:2 * self.max_links + 1].clamp(min=0.05)

        inv_lengths = 1.0 / lengths
        log_lengths = torch.log(lengths)
        log_masses = torch.log(masses)
        mass_per_length = masses / lengths
        mass_fraction = masses / (cart_mass + masses.sum(dim=1, keepdim=True))

        return torch.cat([
            lengths,
            masses,
            inv_lengths,
            log_lengths,
            log_masses,
            mass_per_length,
            mass_fraction,
            cart_mass,
        ], dim=1)

    def _nonlinear_features(self, state: torch.Tensor) -> torch.Tensor:
        cart = state[:, :2]
        theta = state[:, 2:2 + self.max_links]
        theta_dot = state[:, 2 + self.max_links:2 + 2 * self.max_links]

        return torch.cat([
            cart,
            torch.sin(theta),
            1.0 - torch.cos(theta),
            theta * theta.abs(),
            theta_dot,
            theta_dot * theta_dot.abs(),
            theta * theta_dot,
        ], dim=1)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_and_params(obs)
        enhanced_params = self._enhanced_params(params)
        feedback_features = torch.cat([state, self._nonlinear_features(state)], dim=1)
        gains_and_bias = self.param_residual_net(enhanced_params)
        gains = gains_and_bias[:, :feedback_features.shape[1]]
        bias = gains_and_bias[:, feedback_features.shape[1]:feedback_features.shape[1] + 1]
        residual = (gains * feedback_features).sum(dim=1, keepdim=True) + bias
        return mean + residual
