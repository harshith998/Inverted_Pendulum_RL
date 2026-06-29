"""CGAT with learned parameter-conditioned feedback gains.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

For the fixed 3-link setting, this actor adds a classical-controller-shaped
path without hard-coding the controller:

    u_raw = u_cgat_raw + K_phi(lengths, masses, cart_mass) @ state

The gain vector K_phi is produced by a neural network and trained only through
PPO. The final gain layer is zero-initialized, so loading a base CGAT checkpoint
with non-strict weights starts from identical actions.
"""

import torch
import torch.nn as nn

from .cgat_base import CGATBasePPOPolicy


class CGATGainFeedbackPPOPolicy(CGATBasePPOPolicy):
    """Base CGAT actor plus learned parameter-conditioned state feedback."""

    VARIANT = "gain_feedback"

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
        self.state_dim = 2 + 2 * max_links
        param_dim = 2 * max_links + 1

        self.gain_net = nn.Sequential(
            nn.Linear(param_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.state_dim + 1),
        )
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)

    def _state_and_params(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
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

        state = torch.cat([cart_x, cart_v, theta, theta_dot], dim=1)
        params = torch.cat([lengths, masses, cart_mass], dim=1)
        return state, params

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        base_mean = self.mean_head(actor_h)
        state, params = self._state_and_params(obs)
        gains_and_bias = self.gain_net(params)
        gains = gains_and_bias[:, :self.state_dim]
        bias = gains_and_bias[:, self.state_dim:self.state_dim + 1]
        feedback = (gains * state).sum(dim=1, keepdim=True) + bias
        return base_mean + feedback
