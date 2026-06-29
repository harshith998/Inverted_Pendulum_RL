"""CGAT with a pure learned linear feedback head.

No LQR gains, no oracle controller, no controller recomputation.

This variant keeps the standard CGAT encoder and actor/critic MLPs, then adds a
trainable linear feedback path over a compact fixed-3-link physical state. The
feedback weights are initialized at zero, so loading a base CGAT checkpoint
starts with identical behavior and PPO can learn the extra control structure.
"""

import torch
import torch.nn as nn

from .cgat_base import CGATBasePPOPolicy


class CGATFeedbackPPOPolicy(CGATBasePPOPolicy):
    """Base CGAT actor plus learned linear feedback features."""

    VARIANT = "feedback"

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
        state_dim = 2 + 2 * max_links + 2 * max_links + 1
        self.feedback_head = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.feedback_head.weight)
        nn.init.zeros_(self.feedback_head.bias)

    def _structured_state(self, obs: dict) -> torch.Tensor:
        node = obs["node_features"].float()
        edge = obs["edge_features"].float()

        cart = node[:, 0]
        cart_x = cart[:, 6:7]
        cart_v = cart[:, 7:8]
        cart_mass = cart[:, 8:9]

        joints = node[:, 1:self.max_links + 1]
        theta = torch.atan2(joints[:, :, 3], joints[:, :, 4])
        theta_dot = joints[:, :, 5]

        rods = edge[:, 0:2 * self.max_links:2]
        lengths = rods[:, :, 0]
        masses = rods[:, :, 1]

        return torch.cat([
            cart_x,
            cart_v,
            theta,
            theta_dot,
            lengths,
            masses,
            cart_mass,
        ], dim=1)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        return self.mean_head(actor_h) + self.feedback_head(self._structured_state(obs))
