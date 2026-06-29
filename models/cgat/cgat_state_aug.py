"""CGAT State-Augmented PPO Policy.

Pure learned policy: no LQR action, no oracle gain, no controller recomputation.

The base CGAT encoder pools graph node embeddings into one latent vector. For
fixed 3-link control, some stabilizing information is very coordinate-sensitive
(cart x/xdot, joint sin/cos/vel, link length/mass). This variant keeps the CGAT
physics-attention encoder, but also gives the actor/critic a learned skip path
over the raw padded graph features so PPO does not have to reconstruct the flat
control state entirely through message passing.
"""

import torch
import torch.nn as nn

from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from .cgat_base import ICGALayer
from ._icga_base import CGATEncoderBase


class CGATStateAugPPOPolicy(BasePPOPolicy):
    """CGAT encoder fused with raw graph state/parameter features."""

    VARIANT = "state_aug"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.max_links = max_links
        self.max_nodes = max_links + 1
        self.max_edges = 2 * max_links
        raw_dim = self.max_nodes * 9 + self.max_edges * 2 + 2

        self.encoder = CGATEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)
        self.raw_embed = nn.Sequential(
            nn.Linear(raw_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

    def encode(self, obs: dict) -> torch.Tensor:
        graph_emb = self.encoder(obs, compute_inertia_coupling(obs))
        node = obs["node_features"].float().flatten(start_dim=1)
        edge = obs["edge_features"].float().flatten(start_dim=1)
        counts = torch.cat([
            obs["n_nodes"].float() / float(self.max_nodes),
            obs["n_edges"].float() / float(max(self.max_edges, 1)),
        ], dim=1)
        raw_emb = self.raw_embed(torch.cat([node, edge, counts], dim=1))
        return self.fuse(torch.cat([graph_emb, raw_emb], dim=1))
