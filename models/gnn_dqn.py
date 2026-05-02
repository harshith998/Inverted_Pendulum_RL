#GNN-based DQN policy. Encodes the pendulum graph each timestep, outputs Q-values.

"""
MPNN architecture:

Step 1 — Embed
  node/edge feature vector → Linear → hidden-dim embedding

Step 2 — Message passing (n_layers rounds)
  For every edge: message = MLP(src_node_embed || edge_embed)
  Each node: aggregate mean of incoming messages
  Update:    h_new = LayerNorm( h + MLP(h || mean_messages) )   ← residual + norm

Step 3 — Global pool
  Masked mean over real nodes → single graph embedding

Step 4 — Q-head
  graph_embed → Linear → ReLU → Linear → Q-values (one per action bin)
  Two-layer head gives more capacity than a single linear.

Auxiliary head (training signal only)
  graph_embed → Linear → [length, mass] per rod
  Helps the encoder learn to represent physical params explicitly.
  Both Q and aux share one encoder forward pass per training step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_dqn import BaseDQNPolicy

NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 2


class MessagePassingLayer(nn.Module):
    """One round of: gather src embeddings → compute messages → aggregate at dst → update nodes."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e_embed, edge_index, n_edges):
        # h:          (B, max_nodes, hidden)
        # e_embed:    (B, max_edges, hidden)
        # edge_index: (B, 2, max_edges)
        # n_edges:    (B, 1)

        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]

        src_idx = edge_index[:, 0, :]  # (B, max_edges)
        dst_idx = edge_index[:, 1, :]  # (B, max_edges)

        # Gather source node embedding for every edge
        src_exp = src_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        src_h   = h.gather(1, src_exp)  # (B, max_edges, hidden)

        # Compute messages; zero out padding edges
        messages  = self.msg_fn(torch.cat([src_h, e_embed], dim=-1))
        edge_mask = torch.arange(max_edges, device=h.device).unsqueeze(0) < n_edges
        messages  = messages * edge_mask.unsqueeze(-1).float()

        # Mean-aggregate at destination nodes
        dst_exp = dst_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        agg     = torch.zeros(B, max_nodes, hidden, device=h.device)
        agg.scatter_add_(1, dst_exp, messages)
        counts  = torch.zeros(B, max_nodes, 1, device=h.device)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1), edge_mask.unsqueeze(-1).float())
        agg     = agg / counts.clamp(min=1.0)

        # Residual connection — prevents embedding drift and stabilises gradient flow
        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class GNNEncoder(nn.Module):
    """Graph MPNN encoder: embed → message pass → global pool → embedding."""

    def __init__(self, hidden: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEAT_DIM, hidden)
        self.mp_layers  = nn.ModuleList(
            [MessagePassingLayer(hidden, dropout) for _ in range(n_layers)]
        )

    def forward(self, obs: dict) -> torch.Tensor:
        node_features = obs["node_features"].float()   # (B, max_nodes, 9)
        edge_index    = obs["edge_index"].long()        # (B, 2, max_edges)
        edge_features = obs["edge_features"].float()   # (B, max_edges, 2)
        n_nodes       = obs["n_nodes"].long()           # (B, 1)
        n_edges       = obs["n_edges"].long()           # (B, 1)

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)             # (B, max_nodes, hidden)
        e = self.edge_embed(edge_features)             # (B, max_edges, hidden)

        for layer in self.mp_layers:
            h = layer(h, e, edge_index, n_edges)

        # Masked mean pool over real nodes only
        node_mask = torch.arange(max_nodes, device=h.device).unsqueeze(0) < n_nodes
        h = h * node_mask.unsqueeze(-1).float()
        graph_emb = h.sum(dim=1) / n_nodes.float()    # (B, hidden)
        return graph_emb


class GNNDQNPolicy(BaseDQNPolicy):
    """GNN encoder + two-layer Q-head + auxiliary rod-param prediction head."""

    def __init__(self, n_action_bins: int, hidden: int = 64, n_mp_layers: int = 2,
                 max_links: int = 4, dropout: float = 0.1):
        super().__init__(n_action_bins)
        self.encoder   = GNNEncoder(hidden, n_mp_layers, dropout)
        # Two-layer Q-head: more capacity than a single linear
        self.q_head    = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_action_bins),
        )
        # Auxiliary head: predict [length, mass] per rod from the same embedding
        self.aux_head  = nn.Linear(hidden, max_links * 2)
        self.max_links = max_links

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Single encoder pass → (q_values, aux_predictions). Used during training."""
        emb = self.encoder(obs)
        return self.q_head(emb), self.aux_head(emb)

    def get_q_values(self, obs: dict) -> torch.Tensor:
        """Encoder + Q-head only. Used for action selection and target computation."""
        return self.q_head(self.encoder(obs))

    def get_aux_predictions(self, obs: dict) -> torch.Tensor:
        """Aux head only — kept for API compatibility but training uses forward() instead."""
        return self.aux_head(self.encoder(obs))
