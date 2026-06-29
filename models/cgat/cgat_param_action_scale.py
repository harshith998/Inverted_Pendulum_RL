"""Param-residual CGAT with a learned parameter-conditioned action scale.

This keeps the best learned param-residual controller as the base action and
learns only a bounded multiplicative correction from physical parameters. It is
purely learned by PPO: no LQR labels, gains, Riccati solve, or controller
recomputation are used.
"""

import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy


class CGATParamActionScalePPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus learned bounded post-action scaling."""

    VARIANT = "param_action_scale"

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
        self.scale_limit = 0.60
        self.param_action_scale_net = nn.Sequential(
            nn.Linear(enhanced_param_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.param_action_scale_net[-1].weight)
        nn.init.zeros_(self.param_action_scale_net[-1].bias)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        _, params = self._physical_state_and_params(obs)
        enhanced_params = self._enhanced_params(params)
        log_scale = self.param_action_scale_net(enhanced_params).clamp(
            -self.scale_limit, self.scale_limit
        )
        return mean * torch.exp(log_scale)
