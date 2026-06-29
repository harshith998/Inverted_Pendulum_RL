"""CGAT with learned nonlinear parameter-conditioned feedback residual.

This stays learned-only: the residual features are hand-chosen state features,
but all gains are produced by a neural net and trained by PPO. There is no LQR
solve, oracle gain, or per-environment controller recomputation.
"""

import torch
import torch.nn as nn

from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy


class CGATNonlinearFeedbackPPOPolicy(CGATGainFeedbackPPOPolicy):
    """Gain-feedback CGAT plus zero-init nonlinear state-feedback residual."""

    VARIANT = "nonlinear_feedback"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        param_dim = 2 * max_links + 1
        self.nonlinear_feature_dim = 2 + 6 * max_links
        self.nonlinear_gain_net = nn.Sequential(
            nn.Linear(param_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.nonlinear_feature_dim + 1),
        )
        nn.init.zeros_(self.nonlinear_gain_net[-1].weight)
        nn.init.zeros_(self.nonlinear_gain_net[-1].bias)

    def _nonlinear_features(self, state: torch.Tensor) -> torch.Tensor:
        cart = state[:, :2]
        theta = state[:, 2:2 + self.max_links]
        theta_dot = state[:, 2 + self.max_links:2 + 2 * self.max_links]

        features = [
            cart,
            torch.sin(theta),
            1.0 - torch.cos(theta),
            theta * theta.abs(),
            theta_dot,
            theta_dot * theta_dot.abs(),
            theta * theta_dot,
        ]
        return torch.cat(features, dim=1)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._state_and_params(obs)
        features = self._nonlinear_features(state)
        gains_and_bias = self.nonlinear_gain_net(params)
        gains = gains_and_bias[:, :self.nonlinear_feature_dim]
        bias = gains_and_bias[:, self.nonlinear_feature_dim:self.nonlinear_feature_dim + 1]
        residual = (gains * features).sum(dim=1, keepdim=True) + bias
        return mean + residual
