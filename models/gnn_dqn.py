#GNN-based DQN policy. Encodes the pendulum graph each timestep, outputs Q-values.

"""
This uses message passing neural network MPNN, basically instead of transformer attention across
all pairs, uses graph edges to pass messages between connected nodes.

First:
node/edge feature vector -> linear layer -> 64 dim vector embedding.

Step 2 - message passing
for every edge in the graph, compute message:
message = MLP( source_node_embedding + edge_embedding )

then each node, collect all incoming messages and average them
 (message from top edge and bottom edge for each node)

 each new node embedding is MLP(old node + mean(incoming 2 messages))

 basically embeddings spread/propogate through the graph this way

 Step3 - global pool
 We average all node embeddings into one representative 64 dim vector embedding as for the 'full system'

 Step4 - Q-head
 64 'system vector' -> linear layer -> Q-values of bucket actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_dqn import BaseDQNPolicy

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 2


class MessagePassingLayer(nn.Module):
    #One round of: gather src embeddings → compute messages → aggregate at dst → update nodes.

    def __init__(self, hidden: int):
        super().__init__()
        # message from (src_node, edge) → message vector
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
        )
        # update node from (old_embedding, aggregated_messages)
        self.update_fn = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, h, e_embed, edge_index, n_edges):
        # h:          (B, max_nodes, hidden)
        # e_embed:    (B, max_edges, hidden)
        # edge_index: (B, 2, max_edges)
        # n_edges:    (B, 1)

        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]

        src_idx = edge_index[:, 0, :]  # (B, max_edges)
        dst_idx = edge_index[:, 1, :]  # (B, max_edges)

        # gather source node embedding for every edge
        src_exp = src_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        src_h = h.gather(1, src_exp)  # (B, max_edges, hidden)

        # compute messages and zero out padding edges
        messages = self.msg_fn(torch.cat([src_h, e_embed], dim=-1))  # (B, max_edges, hidden)
        edge_mask = (torch.arange(max_edges, device=h.device).unsqueeze(0) < n_edges)  # (B, max_edges)
        messages = messages * edge_mask.unsqueeze(-1).float()

        # incoming means to destination nodes
        dst_exp = dst_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        agg = torch.zeros(B, max_nodes, hidden, device=h.device)
        agg.scatter_add_(1, dst_exp, messages)

        counts = torch.zeros(B, max_nodes, 1, device=h.device)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1), edge_mask.unsqueeze(-1).float())
        agg = agg / counts.clamp(min=1.0)

        # update node embeddings
        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return h_new


class GNNEncoder(nn.Module):
    #Graph MPNN encoder: embed → message pass → global pool → embedding.

    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEAT_DIM, hidden)
        self.mp_layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(n_layers)])

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

        # mean pool over real nodes only
        node_mask = torch.arange(max_nodes, device=h.device).unsqueeze(0) < n_nodes  # (B, max_nodes)
        h = h * node_mask.unsqueeze(-1).float()
        graph_emb = h.sum(dim=1) / n_nodes.float()    # (B, hidden)

        return graph_emb


class GNNDQNPolicy(BaseDQNPolicy):
    #GNN encoder + Q-head. Auxiliary head predicts rod [length, mass] pairs for richer training signal.

    def __init__(self, n_action_bins: int, hidden: int = 64, n_mp_layers: int = 2, max_links: int = 4):
        super().__init__(n_action_bins)
        self.encoder  = GNNEncoder(hidden, n_mp_layers)
        self.q_head   = nn.Linear(hidden, n_action_bins)
        # auxiliary: predict [length, mass] for every rod (max_links rods)
        self.aux_head = nn.Linear(hidden, max_links * 2)
        self.max_links = max_links

    def get_q_values(self, obs: dict) -> torch.Tensor:
        emb = self.encoder(obs)
        return self.q_head(emb)  # (B, n_action_bins)

    def get_aux_predictions(self, obs: dict) -> torch.Tensor:
        #Predict rod params [L0,m0, L1,m1, ...] — used as auxiliary loss during training.
        emb = self.encoder(obs)
        return self.aux_head(emb)  # (B, max_links * 2)
