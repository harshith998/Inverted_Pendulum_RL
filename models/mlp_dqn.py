#MLP baseline DQN. Flattens the padded graph observation and passes through a standard MLP.

import torch
import torch.nn as nn
from models.base_dqn import BaseDQNPolicy

NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 2


class MLPDQNPolicy(BaseDQNPolicy):
    #Flat MLP: concatenate padded node + edge features → hidden layers → Q-values."""

    def __init__(self, n_action_bins: int, max_links: int = 4, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__(n_action_bins)

        max_nodes = max_links + 1
        max_edges = max_links * 2
        flat_dim = max_nodes * NODE_FEAT_DIM + max_edges * EDGE_FEAT_DIM  # all padded features

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_action_bins), #output is number of action discrete vals
        )

    def get_q_values(self, obs: dict) -> torch.Tensor:
        B = obs["node_features"].shape[0]
        # flatten padded node and edge features into one vector
        node_flat = obs["node_features"].float().reshape(B, -1)   # (B, max_nodes * 8)
        edge_flat = obs["edge_features"].float().reshape(B, -1)   # (B, max_edges * 2)
        x = torch.cat([node_flat, edge_flat], dim=-1)             # (B, flat_dim)
        return self.net(x)                                         # (B, n_action_bins)
