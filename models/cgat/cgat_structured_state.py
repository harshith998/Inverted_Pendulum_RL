"""CGAT policy with a structured fixed-3-link state path.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

For the 3-link-only setting, the controller can benefit from seeing the control
state in the same compact order a classical controller would use. This keeps the
CGAT physics-attention encoder, but fuses it with

    [x, xdot, theta_i, theta_dot_i, length_i, mass_i, cart_mass]

where theta_i is recovered from sin/cos. The path is learned end-to-end by PPO.
"""

import torch
import torch.nn as nn

from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from .cgat_base import ICGALayer
from ._icga_base import CGATEncoderBase


class CGATStructuredStatePPOPolicy(BasePPOPolicy):
    """CGAT encoder fused with a compact physical state vector."""

    VARIANT = "structured_state"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.max_links = max_links

        structured_dim = 2 + 2 * max_links + 2 * max_links + 1
        self.encoder = CGATEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)
        self.state_embed = nn.Sequential(
            nn.Linear(structured_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def _structured_state(self, obs: dict) -> torch.Tensor:
        node = obs["node_features"].float()
        edge = obs["edge_features"].float()

        cart = node[:, 0]
        cart_x = cart[:, 6:7]
        cart_v = cart[:, 7:8]
        cart_mass = cart[:, 8:9]

        joints = node[:, 1:self.max_links + 1]
        sin_theta = joints[:, :, 3]
        cos_theta = joints[:, :, 4]
        theta = torch.atan2(sin_theta, cos_theta)
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

    def encode(self, obs: dict) -> torch.Tensor:
        graph_emb = self.encoder(obs, compute_inertia_coupling(obs))
        state_emb = self.state_embed(self._structured_state(obs))
        return self.fuse(torch.cat([graph_emb, state_emb], dim=1))
