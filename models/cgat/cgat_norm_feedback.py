"""Action-scale CGAT plus normalized learned feedback residual.

This is a pure learned controller. It does not compute LQR gains, does not use
oracle actions, and does not update a classical controller at evaluation time.

The existing gain-feedback path uses raw physical state values directly. This
variant adds a second residual path with bounded, normalized state inputs and
log/ratio parameter features so PPO sees a better-conditioned regression
problem across short/long and light/heavy 3-link systems.
"""

import torch
import torch.nn as nn

from .cgat_action_scale import CGATActionScalePPOPolicy


class CGATNormFeedbackPPOPolicy(CGATActionScalePPOPolicy):
    """Action-scale CGAT with bounded normalized parameter feedback."""

    VARIANT = "norm_feedback"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.norm_state_dim = 2 + 2 * max_links
        self.norm_param_dim = 4 * max_links + 3
        self.feedback_limit = 2.0
        self.norm_gain_net = nn.Sequential(
            nn.Linear(self.norm_param_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.norm_state_dim + 1),
        )
        nn.init.zeros_(self.norm_gain_net[-1].weight)
        nn.init.zeros_(self.norm_gain_net[-1].bias)

    def _normalized_state_params(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state, params = self._state_and_params(obs)
        cart_x = (state[:, 0:1] / 2.5).clamp(-2.0, 2.0)
        cart_v = torch.tanh(state[:, 1:2] / 5.0)
        theta = torch.tanh(state[:, 2:2 + self.max_links] / 0.7854)
        theta_dot = torch.tanh(state[:, 2 + self.max_links:] / 10.0)
        norm_state = torch.cat([cart_x, cart_v, theta, theta_dot], dim=1)

        lengths = params[:, :self.max_links].clamp_min(0.05)
        masses = params[:, self.max_links:2 * self.max_links].clamp_min(0.05)
        cart_mass = params[:, -1:].clamp_min(0.05)
        total_link_mass = masses.sum(dim=1, keepdim=True).clamp_min(0.05)
        length_mid = lengths.mean(dim=1, keepdim=True).clamp_min(0.05)
        mass_mid = masses.mean(dim=1, keepdim=True).clamp_min(0.05)
        norm_params = torch.cat(
            [
                torch.log(lengths / 0.6),
                torch.log(masses / 1.0),
                lengths / length_mid,
                masses / mass_mid,
                torch.log(cart_mass / 1.75),
                torch.log(total_link_mass / cart_mass),
                torch.log((lengths * masses).sum(dim=1, keepdim=True) / 1.8),
            ],
            dim=1,
        )
        return norm_state, norm_params.clamp(-5.0, 5.0)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        raw = super().actor_mean(obs, emb, actor_h)
        norm_state, norm_params = self._normalized_state_params(obs)
        gains_and_bias = self.norm_gain_net(norm_params)
        gains = torch.tanh(gains_and_bias[:, :self.norm_state_dim])
        bias = torch.tanh(gains_and_bias[:, self.norm_state_dim:self.norm_state_dim + 1])
        residual = (gains * norm_state).sum(dim=1, keepdim=True) + bias
        return raw + self.feedback_limit * torch.tanh(residual)
