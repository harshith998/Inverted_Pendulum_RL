"""CGAT gain-feedback policy with a tiny learned linear raw-action scale.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

This is a lower-capacity version of ``action_scale``. The fixed scale sweep
showed that action magnitude matters; this variant gives PPO only a small
linear log-scale over structured state/parameter features, initialized at zero
so loading an existing gain-feedback checkpoint preserves its action exactly.
"""

import torch
import torch.nn as nn

from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy


class CGATLinearActionScalePPOPolicy(CGATGainFeedbackPPOPolicy):
    """Gain-feedback CGAT plus a learned linear raw-action scale."""

    VARIANT = "linear_action_scale"

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
        feature_dim = self.state_dim + (2 * max_links + 1)
        self.linear_action_scale = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.linear_action_scale.weight)
        nn.init.zeros_(self.linear_action_scale.bias)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        raw = super().actor_mean(obs, emb, actor_h)
        state, params = self._state_and_params(obs)
        features = torch.cat([state, params], dim=1)
        log_scale = self.linear_action_scale(features).clamp(
            -self.scale_limit, self.scale_limit
        )
        return raw * torch.exp(log_scale)
