# Run (via train_ppo.py): python3.12 training/train_ppo.py --policy gnn_mpnn

"""
GNN-MPNN actor-critic for PPO.

Same message-passing design as gnn_dqn.py but:
  - hidden=128  (vs 64 in DQN)
  - n_layers=3  (vs 2)
  - LayerNorm ON  (stabilises PPO gradient scale)
  - Dropout 0.1 active
  - Outputs a continuous Gaussian policy + value, not Q-values
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 2


class MessagePassingLayer(nn.Module):

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e_embed, edge_index, n_edges):
        # h:          (B, max_nodes, hidden)
        # e_embed:    (B, max_edges, hidden)
        # edge_index: (B, 2, max_edges)
        # n_edges:    (B, 1)

        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]

        src_idx = edge_index[:, 0, :]   # (B, max_edges)
        dst_idx = edge_index[:, 1, :]   # (B, max_edges)

        # Gather source embeddings for every edge
        src_exp = src_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        src_h   = h.gather(1, src_exp)  # (B, max_edges, hidden)

        # Compute messages; zero out padding edges
        messages   = self.msg_fn(torch.cat([src_h, e_embed], dim=-1))
        edge_mask  = torch.arange(max_edges, device=h.device).unsqueeze(0) < n_edges
        messages   = messages * edge_mask.unsqueeze(-1).float()

        # Mean-aggregate at destination nodes
        dst_exp = dst_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        agg     = torch.zeros(B, max_nodes, hidden, device=h.device)
        agg.scatter_add_(1, dst_exp, messages)
        counts  = torch.zeros(B, max_nodes, 1, device=h.device)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1), edge_mask.unsqueeze(-1).float())
        agg     = agg / counts.clamp(min=1.0)

        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return self.norm(h_new)


class GNNMPNNEncoder(nn.Module):

    def __init__(self, hidden: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEAT_DIM, hidden)
        self.mp_layers  = nn.ModuleList(
            [MessagePassingLayer(hidden, dropout) for _ in range(n_layers)]
        )

    def forward(self, obs: dict) -> torch.Tensor:
        node_features = obs["node_features"].float()   # (B, max_nodes, 8)
        edge_index    = obs["edge_index"].long()        # (B, 2, max_edges)
        edge_features = obs["edge_features"].float()   # (B, max_edges, 2)
        n_nodes       = obs["n_nodes"].long()           # (B, 1)
        n_edges       = obs["n_edges"].long()           # (B, 1)

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)             # (B, max_nodes, hidden)
        e = self.edge_embed(edge_features)             # (B, max_edges, hidden)

        for layer in self.mp_layers:
            h = layer(h, e, edge_index, n_edges)

        # Masked mean pool over real nodes
        node_mask = torch.arange(max_nodes, device=h.device).unsqueeze(0) < n_nodes
        h = h * node_mask.unsqueeze(-1).float()
        graph_emb = h.sum(dim=1) / n_nodes.float()    # (B, hidden)
        return graph_emb


class GNNMPNNPPOPolicy(BasePPOPolicy):

    def __init__(self, hidden: int = 128, n_layers: int = 3,
                 max_links: int = 4, dropout: float = 0.1,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = GNNMPNNEncoder(hidden, n_layers, dropout)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs)
