# Run (via train_ppo.py): python3.12 training/train_ppo.py --policy mlp

"""
Flat MLP actor-critic for PPO.

Flattens the padded graph observation (node features + edge features)
into one vector and passes through a 4-layer MLP trunk, then splits
into actor and critic heads via the shared base class.

Wider than the DQN MLP (hidden=128, 4 layers) to be fair to the GNN
models which also use hidden=128.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 2


class MLPPPOPolicy(BasePPOPolicy):

    def __init__(self, hidden: int = 128, max_links: int = 4,
                 dropout: float = 0.1, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)

        max_nodes = max_links + 1
        max_edges = max_links * 2
        flat_dim  = max_nodes * NODE_FEAT_DIM + max_edges * EDGE_FEAT_DIM

        self.backbone = nn.Sequential(
            nn.Linear(flat_dim, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

    def encode(self, obs: dict) -> torch.Tensor:
        B = obs["node_features"].shape[0]
        node_flat = obs["node_features"].float().reshape(B, -1)
        edge_flat = obs["edge_features"].float().reshape(B, -1)
        x = torch.cat([node_flat, edge_flat], dim=-1)
        return self.backbone(x)                        # (B, hidden)
