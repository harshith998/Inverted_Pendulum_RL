"""
CGAT Gravity — scalar β·M̃ + analytic gravity torque injected into node embeddings.

After the initial node embedding, each node receives an additive residual
from its current gravitational torque τ_g_i = g·Lᵢ·sin(θᵢ)·(Σₖ≥ᵢmₖ − mᵢ/2).

M̃ encodes HOW joints couple to each other (structure).
τ_g   encodes WHAT gravity is currently doing to each joint (dynamics).
Together they give the attention a complete physics picture per timestep.

The gravity projection is a single Linear(1→hidden) with no bias, keeping
the contribution zero-initialised and easy to ablate.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy
from ._physics import NODE_FEAT_DIM, compute_inertia_coupling, compute_gravity_torques
from ._icga_base import ICGALayerBase, CGATEncoderBase


class ICGALayer(ICGALayerBase):
    """Same scalar β as base variant."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        self.physics_beta = nn.Parameter(torch.zeros(1))

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        return logits + torch.tanh(self.physics_beta) * M_edge.unsqueeze(-1)


class CGATGravityEncoder(CGATEncoderBase):
    """
    Adds a gravity torque residual to node embeddings before attention:
        h = node_embed(x)  +  gravity_proj(τ_g)
    Then runs standard ICGA layers.
    """

    def __init__(self, hidden: int, n_icga_layers: int, n_heads: int):
        super().__init__(hidden, n_icga_layers, n_heads, icga_cls=ICGALayer)
        self.gravity_proj = nn.Linear(1, hidden, bias=False)
        nn.init.zeros_(self.gravity_proj.weight)

    def forward(self, obs: dict, M_tilde: torch.Tensor) -> torch.Tensor:
        node_features = obs["node_features"].float()
        edge_index    = obs["edge_index"].long()
        edge_features = obs["edge_features"].float()
        n_nodes       = obs["n_nodes"].long()
        n_edges       = obs["n_edges"].long()

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)

        # Inject gravity torques as additive residual on node embeddings
        g_torques = compute_gravity_torques(obs).clamp(-5.0, 5.0)  # (B, max_nodes)
        h = h + self.gravity_proj(g_torques.unsqueeze(-1))  # (B, max_nodes, hidden)

        for layer in self.icga_layers:
            h = layer(h, edge_features, edge_index, n_edges, M_tilde)

        node_mask = (torch.arange(max_nodes, device=h.device).unsqueeze(0)
                     < n_nodes.squeeze(-1).unsqueeze(-1))
        h = h * node_mask.unsqueeze(-1).float()
        return h.sum(dim=1) / n_nodes.float()


class CGATGravityPPOPolicy(BasePPOPolicy):
    """Transformer + scalar β·M̃ + gravity torque node injection."""

    VARIANT = "gravity"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = CGATGravityEncoder(hidden, n_icga_layers, n_heads)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs, compute_inertia_coupling(obs))
