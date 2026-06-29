"""Action-scale CGAT plus a learned energy-feature residual.

This is a pure neural policy. The residual uses hand-built physical features,
but all coefficients are produced by learned networks trained through PPO. It
does not solve LQR, use LQR actions, or recompute any controller at evaluation.
"""

import torch
import torch.nn as nn

from .cgat_action_scale import CGATActionScalePPOPolicy


class CGATEnergyResidualPPOPolicy(CGATActionScalePPOPolicy):
    """Action-scale CGAT with parameter-conditioned energy residual."""

    VARIANT = "energy_residual"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.energy_feature_dim = 2 + 8 * max_links + 5
        self.energy_param_dim = 1 + 7 * max_links
        self.energy_limit = 3.0
        self.energy_gain_net = nn.Sequential(
            nn.Linear(self.energy_param_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.energy_feature_dim + 1),
        )
        nn.init.zeros_(self.energy_gain_net[-1].weight)
        nn.init.zeros_(self.energy_gain_net[-1].bias)

    def _physical_state_params(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node = obs["node_features"].float()
        edge = obs["edge_features"].float()

        cart = node[:, 0]
        cart_x = cart[:, 6:7] * 2.5
        cart_v = cart[:, 7:8] * 5.0

        joints = node[:, 1:self.max_links + 1]
        sin_t = joints[:, :, 3]
        cos_t = joints[:, :, 4]
        theta = torch.atan2(sin_t, cos_t)
        theta_dot = joints[:, :, 5] * 10.0

        rods = edge[:, 0:2 * self.max_links:2]
        lengths = rods[:, :, 0] * 0.9 + 0.3
        masses = rods[:, :, 1] * 1.9 + 0.1
        cart_mass = cart[:, 8:9] * 2.5 + 0.5
        state = torch.cat([cart_x, cart_v, theta, theta_dot], dim=1)
        params = torch.cat([lengths, masses, cart_mass], dim=1)
        return state, params

    def _energy_features(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        cart = state[:, :2]
        theta = state[:, 2:2 + self.max_links]
        theta_dot = state[:, 2 + self.max_links:]
        lengths = params[:, :self.max_links].clamp_min(0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp_min(0.02)
        cart_mass = params[:, -1:].clamp_min(0.05)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        potential = masses * 9.81 * lengths * (1.0 - cos_t)
        kinetic = 0.5 * masses * (lengths * theta_dot).pow(2)
        angular_momentum = masses * lengths.pow(2) * theta_dot
        total_link_mass = masses.sum(dim=1, keepdim=True)
        inertia_sum = (masses * lengths.pow(2)).sum(dim=1, keepdim=True)
        energy_sum = (potential + kinetic).sum(dim=1, keepdim=True)
        mass_ratio = total_link_mass / (cart_mass + total_link_mass)
        avg_length = lengths.mean(dim=1, keepdim=True)

        return torch.cat(
            [
                cart,
                sin_t,
                1.0 - cos_t,
                theta,
                torch.tanh(theta_dot / 12.0),
                potential / 20.0,
                kinetic / 20.0,
                angular_momentum / 10.0,
                sin_t * torch.tanh(theta_dot / 12.0),
                energy_sum / 40.0,
                inertia_sum,
                mass_ratio,
                avg_length,
                total_link_mass / 6.0,
            ],
            dim=1,
        ).clamp(-10.0, 10.0)

    def _energy_params(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links].clamp_min(0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp_min(0.02)
        cart_mass = params[:, -1:].clamp_min(0.05)
        total_mass = masses.sum(dim=1, keepdim=True).clamp_min(0.05)
        return torch.cat(
            [
                torch.log(cart_mass / 1.75),
                lengths,
                masses,
                torch.log(lengths / 0.6),
                torch.log(masses / 1.0),
                1.0 / lengths,
                masses / lengths,
                masses / (cart_mass + total_mass),
            ],
            dim=1,
        ).clamp(-10.0, 10.0)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        raw = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_params(obs)
        features = self._energy_features(state, params)
        gains_and_bias = self.energy_gain_net(self._energy_params(params))
        gains = gains_and_bias[:, :self.energy_feature_dim]
        bias = gains_and_bias[:, self.energy_feature_dim:self.energy_feature_dim + 1]
        residual = (gains * features).sum(dim=1, keepdim=True) + bias
        return raw + self.energy_limit * torch.tanh(residual)
