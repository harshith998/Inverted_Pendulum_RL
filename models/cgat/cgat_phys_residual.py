"""CGAT with a learned physical-feature residual action head.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

The residual path receives compact fixed-3-link physical features plus simple
nonlinear parameter features. It is zero-initialized, so a base checkpoint starts
with identical actions and PPO can add only the useful residual correction.
"""

import torch
import torch.nn as nn

from .cgat_base import CGATBasePPOPolicy


class CGATPhysResidualPPOPolicy(CGATBasePPOPolicy):
    """Base CGAT actor plus zero-init physical residual MLP."""

    VARIANT = "phys_residual"

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
        # x, xdot, theta, thetadot, length, mass, cart_mass,
        # inverse/positive scale features, and parameter-state products.
        feature_dim = 2 + 2 * max_links + 2 * max_links + 1 + 2 * max_links + 2 * max_links
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _features(self, obs: dict) -> torch.Tensor:
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

        inv_lengths = 1.0 / (lengths.abs() + 0.25)
        inv_masses = 1.0 / (masses.abs() + 0.25)
        theta_len = theta * lengths
        vel_mass = theta_dot * masses

        return torch.cat([
            cart_x,
            cart_v,
            theta,
            theta_dot,
            lengths,
            masses,
            cart_mass,
            inv_lengths,
            inv_masses,
            theta_len,
            vel_mass,
        ], dim=1)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        return self.mean_head(actor_h) + self.residual(self._features(obs))
