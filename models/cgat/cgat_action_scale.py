"""CGAT gain-feedback policy with a learned raw-action scale.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

The fixed action-scale sweep improved the best learned checkpoint, so this
variant lets PPO learn a small state/parameter-conditioned multiplicative
correction around the existing actor mean:

    raw_mean = raw_base * exp(clamp(s_phi(features), -limit, limit))

The scale head is zero-initialized, so non-strict loading from a gain-feedback
checkpoint preserves its initial action exactly.
"""

import torch
import torch.nn as nn

from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy


class CGATActionScalePPOPolicy(CGATGainFeedbackPPOPolicy):
    """Gain-feedback CGAT plus learned bounded raw-action scale."""

    VARIANT = "action_scale"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.scale_limit = 0.75
        # state_dim + param_dim from CGATGainFeedbackPPOPolicy.
        feature_dim = self.state_dim + (2 * max_links + 1)
        self.action_scale_net = nn.Sequential(
            nn.Linear(feature_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.action_scale_net[-1].weight)
        nn.init.zeros_(self.action_scale_net[-1].bias)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        raw = super().actor_mean(obs, emb, actor_h)
        state, params = self._state_and_params(obs)
        features = torch.cat([state, params], dim=1)
        log_scale = self.action_scale_net(features).clamp(
            -self.scale_limit, self.scale_limit
        )
        return raw * torch.exp(log_scale)
