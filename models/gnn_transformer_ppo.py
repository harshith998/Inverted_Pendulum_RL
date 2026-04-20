# Run (via train_ppo.py): python3.12 training/train_ppo.py --policy gnn_transformer

"""
Graph-Attention Transformer actor-critic for PPO.

Each layer uses scaled dot-product attention over graph neighbours:

    logit_ij = (W_q h_i) · (W_k h_j) / √d_head  +  w_e · e_ij
    α_ij     = softmax over neighbours j of node i
    h_i'     = concat_heads( Σ_j α_ij · W_v h_j )
    h_i'     = LayerNorm( W_o h_i' + h_i )          ← residual

Edge features enter as an additive scalar bias to the attention logit,
letting the model learn to modulate neighbour importance by physical params
(rod length, rod mass).

Differences from gnn_mpnn_ppo:
  - Dynamic attention weights instead of fixed mean aggregation
  - Multi-head (n_heads=4)
  - Residual + LayerNorm inside every attention layer
  - Feed-forward sublayer after attention (standard transformer block)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_ppo import BasePPOPolicy

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 2


class GraphAttentionLayer(nn.Module):
    """
    One transformer-style graph attention layer.

    Attention is restricted to graph edges (neighbours only),
    not all node pairs — respects the physical topology.
    """

    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = hidden // n_heads

        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(hidden, hidden)

        # Edge feature → scalar bias per attention head
        self.edge_bias = nn.Linear(EDGE_FEAT_DIM, n_heads, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(hidden)
        self.norm2     = nn.LayerNorm(hidden)

        # Position-wise feed-forward sublayer (standard transformer block)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, h, edge_features_raw, edge_index, n_edges):
        """
        h                : (B, max_nodes, hidden)
        edge_features_raw: (B, max_edges, 2)      raw [length, mass] features
        edge_index       : (B, 2, max_edges)
        n_edges          : (B, 1)
        """
        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]
        H, D = self.n_heads, self.d_head

        src_idx = edge_index[:, 0, :]   # (B, max_edges)
        dst_idx = edge_index[:, 1, :]   # (B, max_edges)

        # Project all nodes
        Q = self.W_q(h).view(B, max_nodes, H, D)   # (B, N, H, D)
        K = self.W_k(h).view(B, max_nodes, H, D)
        V = self.W_v(h).view(B, max_nodes, H, D)

        # Gather Q at dst, K at src for each edge
        src_exp  = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        dst_exp  = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        # Note: attention is dst queries keys from src
        Q_edge   = Q.gather(1, dst_idx.unsqueeze(-1).unsqueeze(-1)
                             .expand(B, max_edges, H, D))  # (B, E, H, D)
        K_edge   = K.gather(1, src_exp)                    # (B, E, H, D)
        V_edge   = V.gather(1, src_exp)                    # (B, E, H, D)

        # Scaled dot-product attention logit per head
        scale    = D ** -0.5
        logits   = (Q_edge * K_edge).sum(-1) * scale      # (B, E, H)

        # Edge feature bias (physical params modulate attention)
        e_bias   = self.edge_bias(edge_features_raw.float())  # (B, E, H)
        logits   = logits + e_bias                             # (B, E, H)

        # Numerically stable scatter softmax over incoming edges per destination node.
        # Naive exp(logit) overflows when logits grow large → NaN via inf/inf.
        # Fix: subtract per-node max before exp (log-sum-exp trick).
        edge_mask = torch.arange(max_edges, device=h.device).unsqueeze(0) < n_edges
        dst_h     = dst_idx.unsqueeze(-1).expand(B, max_edges, H)  # (B, E, H)

        # Mask padding edges so they don't pollute the max
        logits = logits.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))

        # Step 1: max logit at each destination node
        node_max = torch.full((B, max_nodes, H), float("-inf"), device=h.device)
        node_max.scatter_reduce_(1, dst_h, logits, reduce="amax", include_self=True)
        node_max = node_max.clamp(min=-1e9)          # isolated nodes: -inf → safe value
        max_per_edge = node_max.gather(1, dst_h)     # (B, E, H)

        # Step 2: shifted exp — padding edges → -inf → exp = 0
        logits_exp = (logits - max_per_edge).exp()
        logits_exp = logits_exp * edge_mask.unsqueeze(-1).float()

        # Step 3: scatter sum denominator, normalise
        denom = torch.zeros(B, max_nodes, H, device=h.device)
        denom.scatter_add_(1, dst_h, logits_exp)
        denom_per_edge = denom.gather(1, dst_h)
        alpha = logits_exp / denom_per_edge.clamp(min=1e-6)
        alpha = self.attn_drop(alpha)

        # Weighted sum of values at each destination node
        # alpha: (B, E, H), V_edge: (B, E, H, D)
        weighted = alpha.unsqueeze(-1) * V_edge            # (B, E, H, D)
        out = torch.zeros(B, max_nodes, H, D, device=h.device)
        dst_hd = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        out.scatter_add_(1, dst_hd, weighted)              # (B, N, H, D)

        # Merge heads + project
        out   = out.reshape(B, max_nodes, hidden)          # (B, N, hidden)
        out   = self.W_o(out)

        # Residual + LayerNorm (attention sublayer)
        h     = self.norm1(h + out)

        # Feed-forward sublayer + residual
        h     = self.norm2(h + self.ff(h))

        return h


class GNNTransformerEncoder(nn.Module):

    def __init__(self, hidden: int, n_layers: int, n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.attn_layers = nn.ModuleList(
            [GraphAttentionLayer(hidden, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, obs: dict) -> torch.Tensor:
        node_features = obs["node_features"].float()   # (B, max_nodes, 8)
        edge_index    = obs["edge_index"].long()        # (B, 2, max_edges)
        edge_features = obs["edge_features"].float()   # (B, max_edges, 2)
        n_nodes       = obs["n_nodes"].long()           # (B, 1)
        n_edges       = obs["n_edges"].long()           # (B, 1)

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)             # (B, max_nodes, hidden)

        for layer in self.attn_layers:
            h = layer(h, edge_features, edge_index, n_edges)

        # Masked mean pool over real nodes
        node_mask = torch.arange(max_nodes, device=h.device).unsqueeze(0) < n_nodes
        h = h * node_mask.unsqueeze(-1).float()
        graph_emb = h.sum(dim=1) / n_nodes.float()    # (B, hidden)
        return graph_emb


class GNNTransformerPPOPolicy(BasePPOPolicy):

    def __init__(self, hidden: int = 128, n_layers: int = 3, n_heads: int = 4,
                 max_links: int = 4, dropout: float = 0.1,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = GNNTransformerEncoder(hidden, n_layers, n_heads, dropout)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs)
