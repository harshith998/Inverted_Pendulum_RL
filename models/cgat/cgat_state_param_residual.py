"""CGAT gain-feedback policy with direct physical state/parameter residual.

This variant adds a zero-initialized MLP residual over physical-unit state and
enhanced physical parameter features. It is more expressive than the linear
param-conditioned residual, while remaining a fixed neural policy at eval time:
no LQR gains, no oracle action, and no controller recomputation.
"""

import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy


class CGATStateParamResidualPPOPolicy(CGATParamResidualPPOPolicy):
    """Gain-feedback CGAT plus direct MLP residual over physical features."""

    VARIANT = "state_param_residual"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        state_dim = 2 + 2 * max_links
        nonlinear_dim = 2 + 6 * max_links
        enhanced_param_dim = 1 + 7 * max_links
        feature_dim = state_dim + nonlinear_dim + enhanced_param_dim

        self.state_param_residual_net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.state_param_residual_net[-1].weight)
        nn.init.zeros_(self.state_param_residual_net[-1].bias)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_and_params(obs)
        features = torch.cat([
            state,
            self._nonlinear_features(state),
            self._enhanced_params(params),
        ], dim=1)
        return mean + self.state_param_residual_net(features)
