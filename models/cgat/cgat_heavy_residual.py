"""Param-residual CGAT with a learned heavy-mass correction head.

This is still a pure learned policy. It does not solve LQR, store LQR gains, or
recompute a controller at evaluation time. The extra head is zero-initialized so
non-strict loading from a param_residual checkpoint preserves the original
actions before PPO fine-tuning.
"""

import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy


class CGATHeavyResidualPPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus a smooth heavy-link residual expert."""

    VARIANT = "heavy_residual"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        enhanced_param_dim = 1 + 7 * max_links
        residual_dim = self.state_dim + self.nonlinear_feature_dim
        self.heavy_residual_net = nn.Sequential(
            nn.Linear(enhanced_param_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, residual_dim + 1),
        )
        nn.init.zeros_(self.heavy_residual_net[-1].weight)
        nn.init.zeros_(self.heavy_residual_net[-1].bias)

    def _heavy_gate(self, params: torch.Tensor) -> torch.Tensor:
        masses = params[:, self.max_links:2 * self.max_links]
        max_mass = masses.max(dim=1, keepdim=True).values
        mean_mass = masses.mean(dim=1, keepdim=True)
        # Smoothly activates beyond the old in-distribution upper mass range.
        return torch.sigmoid(3.0 * (0.65 * max_mass + 0.35 * mean_mass - 2.1))

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_and_params(obs)
        enhanced_params = self._enhanced_params(params)
        feedback_features = torch.cat([state, self._nonlinear_features(state)], dim=1)
        gains_and_bias = self.heavy_residual_net(enhanced_params)
        gains = gains_and_bias[:, :feedback_features.shape[1]]
        bias = gains_and_bias[:, feedback_features.shape[1]:feedback_features.shape[1] + 1]
        residual = (gains * feedback_features).sum(dim=1, keepdim=True) + bias
        return mean + self._heavy_gate(params) * residual
